#!/usr/bin/env julia
# Universal forecast BHP optimization driver.
#
# Loads a .DATA deck, simulates its history, then builds an N-month forecast
# and optimizes per-period BHP per individual well to maximize forecast NPV via
# JutulDarcy adjoint gradients + L-BFGS.
#
# Well roles (producer/injector) are derived from the deck's historical control
# types (last non-Disabled control wins), NOT from well-name prefixes, so the
# driver is deck-agnostic like jutul_run.jl.
#
# CLI:
#   jutul_optimize.jl --case=<path> --out=<dir>
#       --months=N
#       --oil-price= --gas-price= --water-price=
#       --water-cost= --gas-cost= --discount-rate=
#       --bhp-prod-min= --bhp-prod-max=
#       --bhp-inj-min=  --bhp-inj-max=   (all BHP in bar)
#       --max-it=N      (optional, defaults to Jutul unit_box_bfgs default: 25)
# Prices/costs are $/m3 (oil, water) and $/m3 (gas); see liquid_unit/gas_unit=1.
#
# Env L-BFGS caps: OPTI_MAX_IT, OPTI_MAX_INITIAL_UPDATE,
#                  OPTI_LINE_SEARCH_MAX_IT, OPTI_GRAD_TOL.
#
# Writes <out>/optimal_bhp.csv, <out>/production.csv, <out>/summary.json.

using JutulDarcy
using JutulDarcy: replace_target, BottomHolePressureTarget, InjectorControl,
    ProducerControl, DisabledControl, well_symbols, report_timesteps,
    npv_objective, compute_well_qoi
import Jutul
import Jutul: JutulCase
using Dates
using JSON3

# tNavigator-specific keyword unknown to GeoEnergyIO: skip instead of failing
# (mirrors geocode/bin/jutul_run.jl).
JutulDarcy.GeoEnergyIO.InputParser.skip_kw!(:RUNCTRL, 1)

const BAR = 1.0e5  # bar -> Pa

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
function parse_args(argv)
    d = Dict{String,Any}()
    floats = Set(["oil-price", "gas-price", "water-price", "water-cost",
                  "gas-cost", "discount-rate", "bhp-prod-min", "bhp-prod-max",
                  "bhp-inj-min", "bhp-inj-max"])
    for a in argv
        startswith(a, "--") || continue
        kv = split(a[3:end], "=", limit = 2)
        length(kv) == 2 || continue
        k, v = String(kv[1]), String(kv[2])
        if k in ("case", "out", "history-cache")
            d[k] = v
        elseif k in ("months", "max-it")
            d[k] = parse(Int, v)
        elseif k in floats
            d[k] = parse(Float64, v)
        end
    end
    haskey(d, "case") || error("--case=<path> is required")
    haskey(d, "out") || error("--out=<dir> is required")
    required = ("months", "oil-price", "gas-price", "water-price", "water-cost",
                "gas-cost", "discount-rate", "bhp-prod-min", "bhp-prod-max",
                "bhp-inj-min", "bhp-inj-max")
    missing = filter(k -> !haskey(d, k), required)
    isempty(missing) ||
        error("Missing required options: " * join(("--" * k for k in missing), ", "))
    return d
end

# ---------------------------------------------------------------------------
# Role classification: scan the deck's historical forces, last non-Disabled
# control wins (matches JutulDarcy's set_active_controls! semantics).
# ---------------------------------------------------------------------------
# Classify wells by their control at the END of history (the forecast's
# starting state). The forecast continues from there, so we optimize the wells
# actually operating at that point. Wells shut (Disabled) or absent at the
# forecast start are left shut -- we do NOT re-open them, because a well shut
# for depletion / lost perforations cannot be revived just by setting a BHP
# target, and reopening an economically-shut well would ignore workover cost.
function classify_roles(case)
    forces = case.forces isa Vector ? case.forces : [case.forces]
    ctrls = forces[end][:Facility].control
    producers = Symbol[]
    injectors = Symbol[]
    shut = Symbol[]
    for w in collect(well_symbols(case.model))
        c = haskey(ctrls, w) ? ctrls[w] : nothing
        if c isa ProducerControl
            push!(producers, w)
        elseif c isa InjectorControl
            push!(injectors, w)
        else
            push!(shut, w)
        end
    end
    return producers, injectors, shut
end

# ---------------------------------------------------------------------------
# Forecast construction
# ---------------------------------------------------------------------------
function history_end_date(case, total_dt_s)
    start = try
        DateTime(case.input_data["RUNSPEC"]["START"])
    catch
        DateTime(2000, 1, 1)
    end
    return start + Second(round(Int, total_dt_s))
end

# Wells passed here are active at the forecast start, so their control is a
# valid Producer/InjectorControl whose target we can swap for a BHP target.
function set_bhp_target!(fac, well, bhp)
    fac.control[well] = replace_target(fac.control[well], BottomHolePressureTarget(bhp))
    fac.limits[well] = JutulDarcy.default_limits(fac.control[well])
end

function main(argv)
    args = parse_args(argv)
    outdir = args["out"]
    mkpath(outdir)
    months = args["months"]

    println("="^70)
    println("  Forecast BHP optimization: $(basename(args["case"]))")
    println("  months=$months")
    println("="^70)

    # -- Load history -------------------------------------------------------
    case = setup_case_from_data_file(args["case"]; backend = :csr)
    history_cache = get(args, "history-cache", nothing)
    if history_cache !== nothing && isdir(history_cache)
        println("\n[1/5] Loading cached history ...")
        result_hist = simulate_reservoir(case;
            output_path = history_cache, restart = true)
    else
        println("\n[1/5] Simulating history ...")
        result_hist = simulate_reservoir(case)
    end
    state0_forecast = Jutul.setup_state(case.model, result_hist.result.states[end])

    producers, injectors, shut = classify_roles(case)
    isempty(producers) && isempty(injectors) &&
        error("No wells are operating at the forecast start; nothing to optimize.")
    if !isempty(shut)
        println("       Shut at forecast start (left shut, not optimized): $shut")
    end
    println("       Producers: $producers")
    println("       Injectors: $injectors")

    # -- Build N-month forecast forces from the last history force ----------
    println("\n[2/5] Building $months-month forecast ...")
    nperiods = months
    hist_end = history_end_date(case, sum(Float64.(case.dt)))
    fdates = [hist_end + Month(i) for i in 0:nperiods]
    dt_forecast = Float64[
        Dates.value(Millisecond(fdates[i+1] - fdates[i])) / 1000.0
        for i in 1:nperiods
    ]

    # The base forecast continues the final historical controls unchanged.
    last_force = case.forces isa Vector ? case.forces[end] : case.forces
    base_forces = [deepcopy(last_force) for _ in 1:nperiods]
    case_base = JutulCase(case.model, dt_forecast, base_forces;
        state0 = state0_forecast, parameters = case.parameters,
        input_data = case.input_data)

    # control_specs: (label, wells, bhp_min, bhp_max, x0_norm, role).
    # Start BHP optimization from the final historical BHP, clipped to the
    # requested bounds. One control per individual well.
    function spec_for(label, wells, role)
        prefix = role == "producer" ? "bhp-prod" : "bhp-inj"
        bmin = args["$prefix-min"] * BAR
        bmax = args["$prefix-max"] * BAR
        bhp0 = sum(Float64(compute_well_qoi(case.model,
            result_hist.result.states[end], last_force, w, :bhp)) for w in wells) / length(wells)
        x0n = clamp((bhp0 - bmin) / (bmax - bmin), 0.0, 1.0)
        return (label, wells, bmin, bmax, x0n, role)
    end
    specs = Tuple{Symbol,Vector{Symbol},Float64,Float64,Float64,String}[]
    for w in producers
        push!(specs, spec_for(w, [w], "producer"))
    end
    for w in injectors
        push!(specs, spec_for(w, [w], "injector"))
    end
    ncontrols = length(specs)
    println("       Variables: $(ncontrols * nperiods) ($ncontrols controls x $nperiods periods)")

    # Build clean forecast forces with setup_reservoir_forces (controls only,
    # no historical per-well perforation masks, which the adjoint cannot
    # vectorize). Active wells get their historical control with an initial BHP
    # target; wells shut at the forecast start stay disabled.
    orig_fac = last_force[:Facility]
    init_controls = Dict{Symbol,Any}()
    for (_, wells, bmin, bmax, x0n, _) in specs
        for w in wells
            init_controls[w] = replace_target(orig_fac.control[w],
                BottomHolePressureTarget(bmin + x0n * (bmax - bmin)))
        end
    end
    for w in shut
        init_controls[w] = DisabledControl()
    end
    forecast_forces = [setup_reservoir_forces(case.model; control = deepcopy(init_controls))
                       for _ in 1:nperiods]

    case_forecast = JutulCase(case.model, dt_forecast, forecast_forces;
        state0 = state0_forecast, parameters = case.parameters,
        input_data = case.input_data)

    # -- Optimization setup -------------------------------------------------
    x0 = zeros(ncontrols * nperiods)
    for i in 1:nperiods, (c, spec) in enumerate(specs)
        x0[(i-1)*ncontrols+c] = spec[5]
    end

    function set_forecast_bhp!(forces, x)
        xm = reshape(x, ncontrols, nperiods)
        for s in 1:nperiods
            fac = forces[s][:Facility]
            for (c, (_, wells, bmin, bmax, _, _)) in enumerate(specs)
                bhp = bmin + clamp(xm[c, s], 0.0, 1.0) * (bmax - bmin)
                for w in wells
                    set_bhp_target!(fac, w, bhp)
                end
            end
        end
    end

    econ = (oil_price = args["oil-price"], gas_price = args["gas-price"],
            water_price = args["water-price"], water_cost = args["water-cost"],
            gas_cost = args["gas-cost"], discount_rate = args["discount-rate"],
            liquid_unit = 1.0, gas_unit = 1.0)  # 1.0 -> prices are $/m3

    function evaluate_npv(case_eval, r)
        dt_mini = report_timesteps(r.reports, ministeps = true)
        npv_obj(m, st, dt, si, fo) = npv_objective(m, st, dt, si, fo;
            injectors = injectors, producers = producers, timesteps = dt_mini, econ...)
        obj = Jutul.evaluate_objective(npv_obj, case_eval.model,
            r.states, case_eval.dt, case_eval.forces)
        return obj, npv_obj, dt_mini
    end

    cache = Dict{Symbol,Any}()
    function bhp_objective!(x; grad = true)
        set_forecast_bhp!(case_forecast.forces, x)
        sim = simulate_reservoir(case_forecast; output_substates = true, info_level = -1)
        r = sim.result
        obj, npv_obj, dt_mini = evaluate_npv(case_forecast, r)
        grad || return obj
        forces = case_forecast.forces

        targets = Jutul.force_targets(case_forecast.model)
        targets[:Facility][:control] = :control
        targets[:Facility][:limits] = nothing
        key = (length(r.states), length(dt_mini))
        if get(cache, :key, nothing) != key
            cache[:storage] = Jutul.setup_adjoint_forces_storage(
                case_forecast.model, r.states, forces, case_forecast.dt, npv_obj;
                state0 = case_forecast.state0, targets = targets,
                parameters = case_forecast.parameters, eachstep = true, di_sparse = true)
            cache[:key] = key
        end
        dforces, _, _ = Jutul.solve_adjoint_forces!(cache[:storage],
            case_forecast.model, r.states, r.reports, npv_obj, forces;
            state0 = case_forecast.state0, parameters = case_forecast.parameters)
        df = zeros(ncontrols, nperiods)
        for s in 1:nperiods, (c, (_, wells, bmin, bmax, _, _)) in enumerate(specs)
            g = 0.0
            for w in wells
                g += dforces[s][:Facility].control[w].target.value
            end
            df[c, s] = g * (bmax - bmin)
        end
        return (obj, vec(df))
    end

    # -- Optimize -----------------------------------------------------------
    println("\n[3/5] Simulating base and optimizing (L-BFGS + adjoint) ...")
    base_sim = simulate_reservoir(case_base; info_level = -1)
    base_npv, _, _ = evaluate_npv(case_base, base_sim.result)
    max_it = get(args, "max-it", parse(Int, get(ENV, "OPTI_MAX_IT", "25")))
    _, _, opt_history = Jutul.unit_box_bfgs(x0, bhp_objective!;
        maximize = true, max_it = max_it,
        max_initial_update = parse(Float64, get(ENV, "OPTI_MAX_INITIAL_UPDATE", "0.05")),
        line_searchmax_it = parse(Int, get(ENV, "OPTI_LINE_SEARCH_MAX_IT", "5")),
        grad_tol = parse(Float64, get(ENV, "OPTI_GRAD_TOL", "1.0e-3")),
        print = true)
    best_ix = argmax(opt_history.val)
    x_best = opt_history.u[best_ix]
    opt_npv = bhp_objective!(x_best; grad = false)

    converged = opt_npv >= base_npv
    if !converged
        println("       Optimizer did not improve on continued historical controls.")
    end
    improvement = (opt_npv - base_npv) / max(abs(base_npv), eps()) * 100.0
    println("       base NPV=$(round(base_npv/1e6, digits=3)) MM, " *
            "opt NPV=$(round(opt_npv/1e6, digits=3)) MM ($(round(improvement, digits=1))%)")

    # -- Simulate optimized forecast for production export -------------------
    println("\n[4/5] Simulating optimized forecast ...")
    prod_base = extract_production(base_sim, producers, injectors, nperiods)
    set_forecast_bhp!(case_forecast.forces, x_best)
    prod_opt = extract_production(simulate_reservoir(case_forecast; info_level = -1),
        producers, injectors, nperiods)

    # -- Export -------------------------------------------------------------
    println("\n[5/5] Writing CSV + summary ...")
    fmt(x) = round(x, digits = 3)
    open(joinpath(outdir, "production.csv"), "w") do io
        println(io, "period,start_date,end_date,well,role," *
            "base_oil_rate_m3_day,opt_oil_rate_m3_day," *
            "base_gas_rate_m3_day,opt_gas_rate_m3_day," *
            "base_water_rate_m3_day,opt_water_rate_m3_day," *
            "base_water_inj_m3_day,opt_water_inj_m3_day")
        for i in 1:nperiods
            for w in [producers; injectors]
                base = prod_base[w]
                opt = prod_opt[w]
                println(io, "$i,$(fdates[i]),$(fdates[i+1]),$w,$(base.role)," *
                    "$(fmt(base.oil[i])),$(fmt(opt.oil[i]))," *
                    "$(fmt(base.gas[i])),$(fmt(opt.gas[i]))," *
                    "$(fmt(base.water[i])),$(fmt(opt.water[i]))," *
                    "$(fmt(base.winj[i])),$(fmt(opt.winj[i]))")
            end
        end
    end

    xb = reshape(x_best, ncontrols, nperiods)
    open(joinpath(outdir, "optimal_bhp.csv"), "w") do io
        println(io, "period,start_date,end_date,control,role,bhp_bar")
        for i in 1:nperiods, (c, (label, _, bmin, bmax, _, role)) in enumerate(specs)
            bhp = bmin + clamp(xb[c, i], 0.0, 1.0) * (bmax - bmin)
            println(io, "$i,$(fdates[i]),$(fdates[i+1]),$label,$role,$(fmt(bhp/BAR))")
        end
    end

    summary = Dict{String,Any}(
        "case" => args["case"],
        "months" => months, "periods" => nperiods,
        "prices" => Dict("oil" => args["oil-price"], "gas" => args["gas-price"],
            "water" => args["water-price"], "water_inj" => args["water-cost"],
            "gas_inj" => args["gas-cost"], "unit" => "USD/m3"),
        "discount_rate" => args["discount-rate"],
        "n_wells_producer" => length(producers),
        "n_wells_injector" => length(injectors),
        "n_wells_shut" => length(shut),
        "shut_wells" => string.(shut),
        "n_variables" => ncontrols * nperiods,
        "base_npv" => base_npv, "opt_npv" => opt_npv,
        "improvement_pct" => improvement, "max_it" => max_it,
        "iterations" => length(opt_history.val),
        "converged" => converged,
    )
    open(joinpath(outdir, "summary.json"), "w") do io
        JSON3.pretty(io, summary)
    end
    println("\nDone. Wrote production.csv, optimal_bhp.csv, summary.json to $outdir")
    return 0
end

# Per-well rates in m3/day.
function extract_production(sim, producers, injectors, nstep)
    ws = sim.wells
    production = Dict{Symbol,Any}()
    n(v) = Float64.(v) .* 86400.0
    for w in producers
        wd = ws[w]
        oil = haskey(wd, :orat) ? max.(-n(wd[:orat]), 0.0)[1:nstep] : zeros(nstep)
        gas = haskey(wd, :grat) ? max.(-n(wd[:grat]), 0.0)[1:nstep] : zeros(nstep)
        water = haskey(wd, :wrat) ? max.(-n(wd[:wrat]), 0.0)[1:nstep] : zeros(nstep)
        production[w] = (role = "producer", oil = oil, gas = gas, water = water,
            winj = zeros(nstep))
    end
    for w in injectors
        wd = ws[w]
        winj = haskey(wd, :wrat) ? max.(n(wd[:wrat]), 0.0)[1:nstep] : zeros(nstep)
        production[w] = (role = "injector", oil = zeros(nstep),
            gas = zeros(nstep),
            water = zeros(nstep), winj = winj)
    end
    return production
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
