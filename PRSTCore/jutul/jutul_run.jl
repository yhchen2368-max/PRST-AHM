#!/usr/bin/env julia
# JutulDarcy driver: setup_case_from_data_file -> simulate_reservoir ->
# export states.h5, wells.h5, cell_indices.h5 and manifest.json.
# The JLD2 cache written by simulate_reservoir is kept under <out>/jutul_state/
# for native JutulDarcy restart.
#
# CLI:
#   jutul_run.jl --case=<path> --out=<dir> [--restart=none|latest|step:N]

using HDF5
using JSON3
using Dates
using JutulDarcy

# tNavigator-specific keywords unknown to GeoEnergyIO: skip instead of failing.
JutulDarcy.GeoEnergyIO.InputParser.skip_kw!(:RUNCTRL, 1)

# Production and injection rates are split into separate non-negative columns.
const WELLS_COLUMNS = ["time_days", "WBHP", "WOPR", "WWPR", "WGPR", "WWIR", "WGIR"]
const WELLS_UNITS = Dict(
    "time_days" => "d", "WBHP" => "bar",
    "WOPR" => "sm3/d", "WWPR" => "sm3/d", "WGPR" => "sm3/d",
    "WWIR" => "sm3/d", "WGIR" => "sm3/d",
)

function parse_args(argv)
    out = Dict{String,Any}("restart" => "none")
    for a in argv
        if startswith(a, "--case=")        out["case"] = a[8:end]
        elseif startswith(a, "--out=")     out["out"] = a[7:end]
        elseif startswith(a, "--restart=") out["restart"] = a[11:end]
        end
    end
    haskey(out, "case") || error("--case=<path> is required")
    haskey(out, "out")  || error("--out=<dir> is required")
    return out
end

# Translate the restart arg: "none" -> false; "latest" -> true; "step:N" -> N.
function parse_restart(s::String)
    s == "none" && return false
    s == "latest" && return true
    startswith(s, "step:") && return parse(Int, s[6:end])
    error("unrecognised --restart value: $s")
end

# Extract a (ncells, nsteps) matrix from a vector of per-step JutulDarcy state dicts.
function stack_state(states, var::Symbol, nphases_idx::Union{Nothing,Int}=nothing)
    nsteps = length(states)
    first = states[1][var]
    if nphases_idx === nothing
        ncells = length(first)
        m = zeros(Float64, ncells, nsteps)
        for s in 1:nsteps
            m[:, s] = Vector{Float64}(states[s][var])
        end
        return m
    else
        # Saturations is (nphases, ncells); take the requested phase row.
        ncells = size(first, 2)
        m = zeros(Float64, ncells, nsteps)
        for s in 1:nsteps
            row = states[s][var][nphases_idx, :]
            m[:, s] = Vector{Float64}(row)
        end
        return m
    end
end

# Map JutulDarcy phase to the dataset name.
function saturation_dataset_name(phase)
    s = string(phase)
    occursin("Aqueous", s)  && return "/swat"
    occursin("Liquid", s)   && return "/soil"
    occursin("Vapor", s)    && return "/sgas"
    occursin("Vapour", s)   && return "/sgas"
    return nothing
end

function run_simulation(args)
    @info "Setting up case from $(args["case"])..."
    case = setup_case_from_data_file(args["case"])

    jutul_state_dir = joinpath(args["out"], "jutul_state")
    mkpath(jutul_state_dir)

    restart_arg = parse_restart(args["restart"])
    @info "Running simulate_reservoir(output_path=$jutul_state_dir, restart=$restart_arg)..."
    result = simulate_reservoir(case; output_path = jutul_state_dir, restart = restart_arg)
    return case, result.wells, result.states
end

function export_states(case, states, outpath::String)
    pressure = stack_state(states, :Pressure)
    ncells = size(pressure, 1)

    phases = try
        case.model.models.Reservoir.system.phases
    catch
        []
    end

    # Prepend state0 so the exported shape is (ncells, nsteps + 1).
    p0_vec = Vector{Float64}(case.state0[:Reservoir][:Pressure])
    pressure = hcat(p0_vec, pressure)

    sat_datasets = Dict{String,Matrix{Float64}}()
    if !isempty(phases)
        s0_mat = Matrix{Float64}(case.state0[:Reservoir][:Saturations])
        for (i, p) in enumerate(phases)
            name = saturation_dataset_name(p)
            name === nothing && continue
            sat = stack_state(states, :Saturations, i)
            s0 = Vector{Float64}(s0_mat[i, :])
            sat_datasets[name] = hcat(s0, sat)
        end
    end

    total_steps = size(pressure, 2)
    # state0 is at t=0, step k ends sum(case.dt[1:k]) seconds after start;
    # full DateTime so sub-day report steps stay distinct.
    start = DateTime(case.input_data["RUNSPEC"]["START"])
    cum_s = vcat(0.0, cumsum(Vector{Float64}(case.dt)))
    @assert length(cum_s) == total_steps "dt length inconsistent with solved steps"

    h5open(outpath, "w") do f
        f["/pressure"] = pressure
        for (name, data) in sat_datasets
            f[name] = data
        end
        f["/dates_iso8601"] = String[string(start + Second(round(Int, cum_s[s]))) for s in 1:total_steps]
    end
    return ncells, total_steps, collect(keys(sat_datasets))
end

# Look up the (nx, ny, nz) DIMENS from a parsed RUNSPEC block; GeoEnergyIO
# may wrap the vector in a dict.
function _grid_dims(case)
    dims = case.input_data["RUNSPEC"]["DIMENS"]
    if isa(dims, AbstractDict)
        dims = dims["DIMENS"]
    end
    return Int(dims[1]), Int(dims[2]), Int(dims[3])
end

function export_wells(ws, outpath::String)
    names = collect(keys(ws.wells))
    nsteps = length(ws.time)
    ncols = length(WELLS_COLUMNS)
    h5open(outpath, "w") do f
        g = create_group(f, "wells")
        for w in names
            wd = ws[w]
            mat = zeros(Float64, ncols, nsteps)
            for s in 1:nsteps
                # JutulDarcy surface rates are signed (negative = out of the
                # reservoir = production). Split into non-negative production
                # and injection columns so both plot as positive numbers.
                orat = haskey(wd, :orat) ? Float64(wd[:orat][s]) * 86400.0 : 0.0
                wrat = haskey(wd, :wrat) ? Float64(wd[:wrat][s]) * 86400.0 : 0.0
                grat = haskey(wd, :grat) ? Float64(wd[:grat][s]) * 86400.0 : 0.0
                mat[1, s] = Float64(ws.time[s] / 86400.0) # seconds -> days
                mat[2, s] = haskey(wd, :bhp) ? Float64(wd[:bhp][s]) / 1e5 : NaN  # Pa -> bar
                mat[3, s] = max(-orat, 0.0)  # WOPR
                mat[4, s] = max(-wrat, 0.0)  # WWPR
                mat[5, s] = max(-grat, 0.0)  # WGPR
                mat[6, s] = max(wrat, 0.0)   # WWIR
                mat[7, s] = max(grat, 0.0)   # WGIR
            end
            ds = create_dataset(g, string(w), datatype(Float64), dataspace(size(mat)))
            write(ds, mat)
            attrs(ds)["columns"] = join(WELLS_COLUMNS, ",")
            attrs(ds)["units"]   = join([WELLS_UNITS[c] for c in WELLS_COLUMNS], ",")
        end
    end
    return [string(n) for n in names]
end

function export_cell_indices(case, outpath::String)
    # cell_map[i] is the 1-based natural-grid index of the i-th active cell;
    # rebase to 0 for numpy consumers.
    cm = case.model.models.Reservoir.data_domain.representation.cell_map
    h5open(outpath, "w") do f
        f["/active_to_natural"] = Int64.(cm .- 1)
    end
end

function main(argv)
    args = parse_args(argv)
    outdir = args["out"]
    mkpath(outdir)

    case, ws, states = run_simulation(args)
    ncells, nsteps, sat_names = export_states(case, states, joinpath(outdir, "states.h5"))
    well_names = export_wells(ws, joinpath(outdir, "wells.h5"))
    export_cell_indices(case, joinpath(outdir, "cell_indices.h5"))
    nx, ny, nz = _grid_dims(case)

    manifest = Dict{String,Any}(
        "simulator" => "jutul",
        "jutuldarcy" => string(pkgversion(JutulDarcy)),
        "case" => args["case"],
        "grid" => Dict("nx" => nx, "ny" => ny, "nz" => nz, "n_active" => ncells),
        "states" => Dict("file" => "states.h5", "datasets" => sat_names, "n_steps" => nsteps),
        "wells" => Dict("file" => "wells.h5", "names" => well_names,
                        "columns" => WELLS_COLUMNS, "units" => WELLS_UNITS),
        "restart" => args["restart"],
        "completed_at" => string(now(UTC)) * "Z",
    )
    open(joinpath(outdir, "manifest.json"), "w") do io
        JSON3.pretty(io, manifest)
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
