#!/usr/bin/env python
"""从 ECLIPSE case 构建、训练 CGNet，并进行注采梯度优化。

CGNet 在本仓库中指由细网格模型上采样得到、可重新标定的粗网格
``GenericBlackOilModel``。脚本完成下面的闭环：

1. 读入 ``.DATA``，并可选读入 ECLIPSE 二进制 ``.UNRST`` 结果；
2. 对细网格做逻辑均匀分区，生成粗网格（CGNet）；
3. 用压力、饱和度及（若有）井解训练三个全局 CGNet 系数：PV、
   地层传导率和井指数；
4. 在已训练 CGNet 上，用投影梯度法优化分期注入/生产控制以提高 NPV。

``.UNRST`` 由 ``PRSTCore.deckformat.resultinput`` 中的 reader 转换为
PRSTCore/MRST 风格状态，并会按照 DATA 文件的单位制转换压力。UNSMRY
仅有汇总曲线，不足以训练网格状态；请将 ``--results`` 指向 UNRST、
其无扩展名前缀或其所在目录。

示例（先用较少迭代验证流程）：

    python scripts/eclipse_to_cgnet_train_optimize.py \
        --deck examples/SPE9/SPE9_CP.DATA \
        --results examples/SPE9/RESULTS/SPE9_CP.UNRST \
        --coarse-dims 4 4 2 --train-iters 2 --control-iters 2

未给 ``--results`` 时，脚本会先模拟细网格模型，并将该模拟结果作为训练
参考数据。这适合尚未运行 ECLIPSE、但希望先验证 CGNet 训练流程的场景。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


# 允许从仓库根目录以外运行 ``python scripts/...py``。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.simulators.simulate_schedule_ad import simulate_schedule_ad
from PRSTCore.ad_core.upscale import upscale_model_tpfa, upscale_schedule, upscale_state
from PRSTCore.coarsegrid import compress_partition, partition_ui
from PRSTCore.deckformat.deckinput import read_eclipse_deck
from PRSTCore.deckformat.resultinput import convert_restart_to_states


def _resolve_unrst(path_text: str) -> Path:
    """Resolve a UNRST path, a result prefix, or a result directory."""
    candidate = Path(path_text).expanduser().resolve()
    if candidate.is_dir():
        files = sorted(candidate.glob("*.UNRST")) + sorted(candidate.glob("*.FUNRST"))
        if len(files) != 1:
            raise ValueError(
                f"{candidate} contains {len(files)} restart files; please pass one explicit .UNRST file"
            )
        return files[0]
    if candidate.suffix.upper() in {".UNRST", ".FUNRST"}:
        return candidate
    for suffix in (".UNRST", ".FUNRST"):
        possible = candidate.with_suffix(suffix)
        if possible.exists():
            return possible
    raise FileNotFoundError(
        f"Could not find an ECLIPSE restart file from {candidate}. Expected .UNRST or .FUNRST."
    )


def _eclipse_unit_name(deck: dict[str, Any]) -> str:
    runspec = deck.get("RUNSPEC", {})
    for name in ("METRIC", "FIELD", "LAB", "PVT_M", "SI"):
        if runspec.get(name, False):
            return name
    return "METRIC"


def load_unrst_states(path_text: str, raw_deck: dict[str, Any], fine_model: Any) -> list[dict[str, Any]]:
    """Read pressure/saturation snapshots through PRSTCore resultinput."""
    source = _resolve_unrst(path_text)
    states, _ = convert_restart_to_states(
        str(source),
        fine_model.G,
        include_well_sols=False,
        include_fluxes=False,
        unit_system=_eclipse_unit_name(raw_deck),
    )
    if not states:
        raise ValueError(f"{source}: no PRESSURE snapshots were found in the restart file")
    return [_normalise_restart_state(state, fine_model) for state in states]


def _normalise_restart_state(state: dict[str, Any], model: Any) -> dict[str, Any]:
    output = dict(state)
    nc = _num_cells(model)
    pressure = np.asarray(output["pressure"], dtype=float).ravel()[:nc]
    output["pressure"] = pressure
    if "s" in output:
        sat = np.asarray(output["s"], dtype=float)
        if sat.ndim == 1:
            sat = sat.reshape(-1, 1)
        output["s"] = sat[:nc, :]
        output["sW"] = output["s"][:, 0]
        output["sG"] = output["s"][:, 2] if output["s"].shape[1] >= 3 else np.zeros(nc)
    else:
        sw = np.asarray(output.get("sW", np.zeros(nc)), dtype=float).ravel()[:nc]
        sg = np.asarray(output.get("sG", np.zeros(nc)), dtype=float).ravel()[:nc]
        so = np.maximum(1.0 - sw - sg, 0.0)
        output["s"] = np.column_stack([sw, so, sg])
        output["sW"] = sw
        output["sG"] = sg
    output.setdefault("wellSol", [])
    return output


def _num_cells(model: Any) -> int:
    grid = getattr(model, "G", None)
    if isinstance(grid, dict):
        return int(grid["cells"]["num"])
    raise ValueError("Model has no grid cell count")


def _with_saturation_matrix(state: dict[str, Any], model: Any) -> dict[str, Any]:
    """Return a copy containing an MRST-style ``s`` matrix for upscaling."""
    output = dict(state)
    nc = _num_cells(model)
    if "s" in output:
        sat = np.asarray(output["s"], dtype=float)
        if sat.ndim == 1:
            sat = sat.reshape(-1, 1)
        output["s"] = sat
        return output

    sw = np.asarray(output.get("sW", np.zeros(nc)), dtype=float).ravel()
    sg = np.asarray(output.get("sG", np.zeros(nc)), dtype=float).ravel()
    oil = np.maximum(1.0 - sw - sg, 0.0)
    if bool(getattr(model, "gas", False)):
        output["s"] = np.column_stack([sw, oil, sg])
    else:
        output["s"] = np.column_stack([sw, oil])
    return output


def _as_simulator_state(state: dict[str, Any], model: Any) -> dict[str, Any]:
    """Convert a coarse upscaled saturation matrix back to simulator fields."""
    output = dict(state)
    if "s" in output:
        sat = np.asarray(output["s"], dtype=float)
        if sat.ndim == 1:
            sat = sat.reshape(-1, 1)
        output["sW"] = sat[:, 0]
        output["sG"] = sat[:, 2] if sat.shape[1] >= 3 else np.zeros(sat.shape[0])
    # ``upscale_state`` copies unknown fields before upscaling the fields it
    # knows about.  In particular an inactive ``rv`` field can otherwise
    # retain fine-grid length and make the coarse simulator fail validation.
    nc = _num_cells(model)
    for field in ("pressure", "sW", "sG", "rs", "rv"):
        if field not in output:
            continue
        values = np.asarray(output[field])
        if values.ndim and values.shape[0] != nc:
            output.pop(field)
    output.setdefault("wellSol", [])
    return output


def _upscale_schedule_for_zero_based_wells(coarse_model: Any, schedule: dict[str, Any]) -> dict[str, Any]:
    """Use ``upscale_schedule`` while retaining PRSTCore's zero-based cells.

    The deck converter and GenericBlackOilModel use zero-based active-cell
    indices, whereas the generic upscaler follows MRST's one-based API.
    """
    one_based = copy.deepcopy(schedule)
    for control in one_based.get("control", []):
        for well in control.get("W", []):
            if "cells" in well:
                cells = np.asarray(well["cells"], dtype=int).ravel()
                well["cells"] = (cells + 1).tolist()
    coarse_schedule = upscale_schedule(coarse_model, one_based)
    for control in coarse_schedule.get("control", []):
        for well in control.get("W", []):
            if "cells" in well:
                well["cells"] = (np.asarray(well["cells"], dtype=int).ravel() - 1).tolist()
    return coarse_schedule


def _automatic_coarse_dims(cart_dims: Iterable[int], target_cells: int) -> list[int]:
    """Choose a balanced logical partition with at most target_cells blocks."""
    fine = np.asarray(list(cart_dims), dtype=int)
    coarse = np.ones_like(fine)
    target_cells = max(1, int(target_cells))
    while int(np.prod(coarse)) < target_cells:
        possible = [
            i for i in range(fine.size)
            if coarse[i] < fine[i] and int(np.prod(coarse)) * (coarse[i] + 1) // coarse[i] <= target_cells
        ]
        if not possible:
            break
        # Refine the direction that remains most under-resolved.
        axis = max(possible, key=lambda i: fine[i] / coarse[i])
        coarse[axis] += 1
    return coarse.astype(int).tolist()


def make_cgnet(
    fine_model: Any,
    fine_state0: dict[str, Any],
    fine_schedule: dict[str, Any],
    coarse_dims: list[int],
) -> tuple[Any, dict[str, Any], dict[str, Any], np.ndarray]:
    """Build the upscaled CGNet model, state and schedule."""
    fine_grid = fine_model.G
    # ``init_eclipse_grid`` stores the face arrays but older imported decks
    # do not always retain the redundant MRST ``faces.num`` scalar.  The
    # coarse-grid routines use it when validating a partition.
    if "faces" in fine_grid and "num" not in fine_grid["faces"]:
        neighbors = np.asarray(fine_grid["faces"].get("neighbors", []))
        fine_grid["faces"]["num"] = int(neighbors.shape[0])
    partition = compress_partition(partition_ui(fine_grid, coarse_dims))
    coarse_model = upscale_model_tpfa(fine_model, partition, trans_from_rock=False)
    coarse_state0 = _as_simulator_state(
        upscale_state(coarse_model, fine_model, _with_saturation_matrix(fine_state0, fine_model)),
        coarse_model,
    )
    coarse_schedule = _upscale_schedule_for_zero_based_wells(coarse_model, fine_schedule)
    return coarse_model, coarse_state0, coarse_schedule, partition


def _clone_model(model: Any) -> Any:
    clone = copy.copy(model)
    operators = getattr(model, "operators", None)
    if isinstance(operators, dict):
        clone.operators = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value)
            for key, value in operators.items()
        }
    if hasattr(model, "porevolume") and isinstance(model.porevolume, np.ndarray):
        clone.porevolume = np.array(model.porevolume, copy=True)
    return clone


def _scale_well_indices(schedule: dict[str, Any], multiplier: float) -> dict[str, Any]:
    result = copy.deepcopy(schedule)
    for control in result.get("control", []):
        for well in control.get("W", []):
            wi = np.asarray(well.get("WI", 1.0), dtype=float)
            well["WI"] = (wi * multiplier).tolist() if wi.ndim else float(wi * multiplier)
    return result


def _scaled_cgnet_setup(
    base_model: Any,
    base_state0: dict[str, Any],
    base_schedule: dict[str, Any],
    unit_parameters: np.ndarray,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, float]]:
    """Apply three log-scaled, global CGNet calibration parameters."""
    u = np.clip(np.asarray(unit_parameters, dtype=float), 0.0, 1.0)
    if u.size != 3:
        raise ValueError("CGNet calibration uses exactly [porevolume, transmissibility, WI] parameters")
    lower = np.array([0.25, 0.10, 0.10])
    upper = np.array([4.0, 10.0, 10.0])
    factors = lower * np.power(upper / lower, u)
    model = _clone_model(base_model)
    ops = model.operators
    ops["pv"] = np.asarray(ops["pv"], dtype=float) * factors[0]
    model.porevolume = np.asarray(ops["pv"], dtype=float).copy()
    for name in ("T", "T_all"):
        if name in ops:
            ops[name] = np.asarray(ops[name], dtype=float) * factors[1]
    schedule = _scale_well_indices(base_schedule, float(factors[2]))
    return model, copy.deepcopy(base_state0), schedule, {
        "porevolume": float(factors[0]),
        "transmissibility": float(factors[1]),
        "well_index": float(factors[2]),
    }


def _align_reference_states(states: list[dict[str, Any]], nsteps: int) -> list[dict[str, Any]]:
    valid = [state for state in states if "pressure" in state]
    if not valid:
        raise ValueError("Reference results contain no pressure states")
    if len(valid) == nsteps:
        return valid
    indices = np.rint(np.linspace(0, len(valid) - 1, nsteps)).astype(int)
    print(
        f"[reference] {len(valid)} snapshots are mapped to {nsteps} schedule steps "
        f"(nearest report-step mapping)."
    )
    return [valid[i] for i in indices]


def _coarse_reference_states(
    fine_states: list[dict[str, Any]], fine_model: Any, coarse_model: Any
) -> list[dict[str, Any]]:
    reference: list[dict[str, Any]] = []
    nc_fine = _num_cells(fine_model)
    for state in fine_states:
        if np.asarray(state.get("pressure", [])).size != nc_fine:
            continue
        coarse = upscale_state(coarse_model, fine_model, _with_saturation_matrix(state, fine_model))
        reference.append(_as_simulator_state(coarse, coarse_model))
    if not reference:
        raise ValueError("No reference state matches the active-cell count of the DATA model")
    return reference


def _simulation_states(model: Any, state0: dict[str, Any], schedule: dict[str, Any], solver: Any = None):
    _, states = simulate_schedule_ad(
        copy.deepcopy(state0), model, schedule, nonlinear_solver=solver, verbose=False
    )
    return states


def _safe_scale(values: np.ndarray, floor: float) -> float:
    return max(float(np.mean(np.abs(values))), floor)


def _trajectory_mismatch(
    predicted: list[dict[str, Any]],
    observed: list[dict[str, Any]],
    pressure_weight: float,
    saturation_weight: float,
    well_weight: float,
) -> float:
    """Dimensionless pressure/saturation/well mismatch for CGNet training."""
    if len(predicted) != len(observed):
        raise ValueError("Predicted and observed trajectories have different lengths")
    value = 0.0
    for pred, obs in zip(predicted, observed):
        if pressure_weight:
            p_obs = np.asarray(obs["pressure"], dtype=float)
            p_pre = np.asarray(pred["pressure"], dtype=float)
            value += pressure_weight * float(np.mean(((p_pre - p_obs) / _safe_scale(p_obs, 1.0e5)) ** 2))
        if saturation_weight:
            for field in ("sW", "sG"):
                if field in obs and field in pred:
                    s_obs = np.asarray(obs[field], dtype=float)
                    s_pre = np.asarray(pred[field], dtype=float)
                    value += saturation_weight * float(np.mean((s_pre - s_obs) ** 2))
        if well_weight and obs.get("wellSol") and pred.get("wellSol"):
            for w_pre, w_obs in zip(pred["wellSol"], obs["wellSol"]):
                for field in ("qWs", "qOs", "bhp"):
                    ref = float(w_obs.get(field, 0.0))
                    est = float(w_pre.get(field, 0.0))
                    floor = 1.0e5 if field == "bhp" else 1.0e-8
                    value += well_weight * ((est - ref) / max(abs(ref), floor)) ** 2
    return value / max(len(predicted), 1)


def _forward_gradient(
    objective: Callable[[np.ndarray], float], u: np.ndarray, base_value: float, epsilon: float
) -> np.ndarray:
    gradient = np.zeros_like(u, dtype=float)
    for index in range(u.size):
        candidate = u.copy()
        candidate[index] = min(1.0, candidate[index] + epsilon)
        delta = candidate[index] - u[index]
        if delta == 0.0:
            candidate[index] = max(0.0, candidate[index] - epsilon)
            delta = candidate[index] - u[index]
        if delta == 0.0:
            continue
        gradient[index] = (objective(candidate) - base_value) / delta
    return gradient


def projected_gradient_minimize(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    max_iterations: int,
    gradient_epsilon: float,
    initial_step: float,
    label: str,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    """Box-constrained gradient descent with Armijo backtracking.

    The CGNet has only three training parameters and the default control
    parameterisation is small.  Explicit finite differences therefore give
    a dependable gradient even for decks whose current AD path has no
    derivatives through every facility control.
    """
    u = np.clip(np.asarray(initial, dtype=float), 0.0, 1.0)
    value = float(objective(u))
    history = [{"iteration": 0, "value": value, "u": u.copy()}]
    print(f"[{label}] iter=0 objective={value:.6e}")

    for iteration in range(1, max_iterations + 1):
        gradient = _forward_gradient(objective, u, value, gradient_epsilon)
        norm = float(np.linalg.norm(gradient, ord=np.inf))
        if not np.isfinite(norm) or norm < 1.0e-8:
            print(f"[{label}] stopping: projected gradient norm={norm:.3e}")
            break
        direction = -gradient / norm
        step = initial_step
        accepted = False
        directional_derivative = float(np.dot(gradient, direction))
        for _ in range(8):
            candidate = np.clip(u + step * direction, 0.0, 1.0)
            if np.array_equal(candidate, u):
                step *= 0.5
                continue
            candidate_value = float(objective(candidate))
            if np.isfinite(candidate_value) and candidate_value <= value + 1.0e-4 * step * directional_derivative:
                u, value = candidate, candidate_value
                accepted = True
                break
            step *= 0.5
        history.append(
            {
                "iteration": iteration,
                "value": value,
                "u": u.copy(),
                "gradient_inf_norm": norm,
                "accepted_step": step if accepted else 0.0,
            }
        )
        print(f"[{label}] iter={iteration} objective={value:.6e}, |g|inf={norm:.3e}, step={step if accepted else 0.0:.3e}")
        if not accepted:
            print(f"[{label}] stopping: no improving step found")
            break
    return u, value, history


def _explode_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    """Give every report step its own control record before optimisation."""
    output = copy.deepcopy(schedule)
    values = np.asarray(output["step"]["val"], dtype=float)
    old_controls = output["control"]
    old_indices = np.asarray(output["step"]["control"], dtype=int)
    output["control"] = [copy.deepcopy(old_controls[int(index)]) for index in old_indices]
    output["step"]["control"] = np.arange(values.size, dtype=int)
    return output


def _make_control_layout(schedule: dict[str, Any], periods: int, per_well: bool) -> list[dict[str, Any]]:
    signatures: dict[str, int] = {}
    for control in schedule.get("control", []):
        for index, well in enumerate(control.get("W", [])):
            name = str(well.get("name", f"W{index}"))
            signatures.setdefault(name, int(np.sign(float(well.get("sign", 0.0)))))
    categories = []
    if any(sign > 0 for sign in signatures.values()):
        categories.append("injector")
    if any(sign < 0 for sign in signatures.values()):
        categories.append("producer")
    if not categories:
        raise ValueError("No injector (sign=1) or producer (sign=-1) wells were found in the schedule")

    layout: list[dict[str, Any]] = []
    for period in range(periods):
        for category in categories:
            if per_well:
                wanted = 1 if category == "injector" else -1
                entities = [name for name, sign in signatures.items() if sign == wanted]
            else:
                entities = ["all"]
            for entity in entities:
                layout.append({"period": period, "category": category, "entity": entity})
    return layout


def _control_multiplier(parameter: dict[str, Any], unit_value: float) -> float:
    if parameter["category"] == "injector":
        return 0.50 + unit_value  # 0.5x .. 1.5x
    return 0.85 + 0.30 * unit_value  # 0.85x .. 1.15x


def apply_control_vector(
    base_schedule: dict[str, Any], layout: list[dict[str, Any]], unit_vector: np.ndarray
) -> dict[str, Any]:
    schedule = _explode_schedule(base_schedule)
    u = np.asarray(unit_vector, dtype=float)
    if u.size != len(layout):
        raise ValueError("Control vector does not match the selected control layout")
    nsteps = len(schedule["step"]["val"])
    periods = max(item["period"] for item in layout) + 1
    parameter_map = {
        (item["period"], item["category"], item["entity"]): _control_multiplier(item, float(u[index]))
        for index, item in enumerate(layout)
    }

    for step, control in enumerate(schedule["control"]):
        period = min(periods - 1, step * periods // max(nsteps, 1))
        for index, well in enumerate(control.get("W", [])):
            sign = int(np.sign(float(well.get("sign", 0.0))))
            category = "injector" if sign > 0 else "producer" if sign < 0 else None
            if category is None:
                continue
            name = str(well.get("name", f"W{index}"))
            multiplier = parameter_map.get(
                (period, category, name), parameter_map.get((period, category, "all"))
            )
            if multiplier is None or "val" not in well:
                continue
            try:
                well["val"] = float(well["val"]) * multiplier
            except (TypeError, ValueError):
                continue
    return schedule


def net_present_value(states: list[dict[str, Any]], schedule: dict[str, Any], args: argparse.Namespace) -> float:
    """Compute a sign-safe field NPV from positive-magnitude well rates."""
    cumulative_time = 0.0
    total = 0.0
    dts = np.asarray(schedule["step"]["val"], dtype=float)
    for dt, state in zip(dts, states):
        cumulative_time += dt
        discount = (1.0 + args.discount_rate) ** (-cumulative_time / (365.25 * 86400.0))
        for well in state.get("wellSol", []):
            if not well.get("status", True):
                continue
            sign = float(well.get("sign", 0.0))
            qw = max(float(well.get("qWs", 0.0)), 0.0)
            qo = max(float(well.get("qOs", 0.0)), 0.0)
            qg = max(float(well.get("qGs", 0.0)), 0.0)
            if sign < 0.0:
                cashflow = args.oil_price * qo + args.gas_price * qg - args.water_production_cost * qw
            else:
                cashflow = -args.water_injection_cost * qw - args.gas_injection_cost * qg
            total += discount * dt * cashflow
    return float(total)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck", required=True, help="ECLIPSE .DATA 文件路径")
    parser.add_argument("--results", help="可选 .UNRST、无扩展名的重启前缀或只含一个 UNRST 的目录")
    parser.add_argument("--coarse-dims", nargs=3, type=int, metavar=("NX", "NY", "NZ"), help="CGNet 逻辑粗网格维度")
    parser.add_argument("--target-coarse-cells", type=int, default=32, help="未指定 --coarse-dims 时的最大粗块数（默认 32）")
    parser.add_argument("--train-fraction", type=float, default=0.5, help="用于训练的前段时间比例（默认 0.5）")
    parser.add_argument("--train-steps", type=int, help="覆盖 --train-fraction 的训练报告步数")
    parser.add_argument("--train-iters", type=int, default=5, help="CGNet 训练的最大梯度迭代数")
    parser.add_argument("--control-periods", type=int, default=4, help="注采控制划分的时间段数")
    parser.add_argument("--control-iters", type=int, default=5, help="NPV 优化的最大梯度迭代数")
    parser.add_argument("--optimization-steps", type=int, help="仅优化前 N 个报告步；默认优化完整生产期")
    parser.add_argument("--per-well-controls", action="store_true", help="为每口井、每个时段单独优化控制；默认按所有注井/采井聚合")
    parser.add_argument("--gradient-epsilon", type=float, default=0.02, help="有限差分梯度的 unit-box 扰动量")
    parser.add_argument("--gradient-step", type=float, default=0.20, help="投影梯度法的初始步长")
    parser.add_argument("--pressure-weight", type=float, default=1.0, help="训练压力失配权重")
    parser.add_argument("--saturation-weight", type=float, default=1.0, help="训练饱和度失配权重")
    parser.add_argument("--well-weight", type=float, default=0.1, help="存在井解参考时的井数据失配权重")
    parser.add_argument("--oil-price", type=float, default=1.0)
    parser.add_argument("--gas-price", type=float, default=0.1)
    parser.add_argument("--water-production-cost", type=float, default=0.1)
    parser.add_argument("--water-injection-cost", type=float, default=0.1)
    parser.add_argument("--gas-injection-cost", type=float, default=0.1)
    parser.add_argument("--discount-rate", type=float, default=0.0, help="年贴现率")
    parser.add_argument("--output-dir", default="results/cgnet_train_optimize", help="产物目录")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    deck_path = Path(args.deck).expanduser().resolve()
    if not deck_path.is_file():
        raise FileNotFoundError(f"DATA file does not exist: {deck_path}")
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train-fraction must be in (0, 1]")
    if args.control_periods < 1:
        raise ValueError("--control-periods must be positive")

    print(f"[1/5] Reading ECLIPSE DATA: {deck_path}")
    raw_deck = read_eclipse_deck(str(deck_path))
    fine_state0, fine_model, fine_schedule, fine_solver = init_eclipse_problem_ad(str(deck_path))
    fine_grid = fine_model.G
    fine_dims = np.asarray(fine_grid["cartDims"], dtype=int)
    coarse_dims = args.coarse_dims or _automatic_coarse_dims(fine_dims, args.target_coarse_cells)
    if any(dim < 1 for dim in coarse_dims) or any(c > f for c, f in zip(coarse_dims, fine_dims)):
        raise ValueError(f"Invalid coarse dimensions {coarse_dims}; fine grid dimensions are {fine_dims.tolist()}")

    print(f"[2/5] Building CGNet: fine={fine_dims.tolist()}, coarse={coarse_dims}")
    coarse_model, coarse_state0, coarse_schedule, partition = make_cgnet(
        fine_model, fine_state0, fine_schedule, list(map(int, coarse_dims))
    )
    print(f"       active fine cells={_num_cells(fine_model)}, CGNet cells={_num_cells(coarse_model)}")

    print("[3/5] Loading reference trajectory")
    if args.results:
        fine_reference = load_unrst_states(args.results, raw_deck, fine_model)
        reference_source = str(_resolve_unrst(args.results))
    else:
        print("       No --results supplied; simulating the fine model as reference.")
        fine_reference = _simulation_states(fine_model, fine_state0, fine_schedule, fine_solver)
        reference_source = "fine-model simulation"
    fine_reference = _align_reference_states(fine_reference, len(fine_schedule["step"]["val"]))
    observed = _coarse_reference_states(fine_reference, fine_model, coarse_model)

    nsteps = len(coarse_schedule["step"]["val"])
    ntrain = args.train_steps or int(math.ceil(nsteps * args.train_fraction))
    ntrain = max(1, min(int(ntrain), nsteps))
    train_schedule = copy.deepcopy(coarse_schedule)
    train_schedule["step"]["val"] = np.asarray(train_schedule["step"]["val"], dtype=float)[:ntrain]
    train_schedule["step"]["control"] = np.asarray(train_schedule["step"]["control"], dtype=int)[:ntrain]
    train_observed = observed[:ntrain]
    print(f"       reference={reference_source}; training steps={ntrain}/{nsteps}")

    def training_objective(u: np.ndarray) -> float:
        try:
            model, state0, schedule, _ = _scaled_cgnet_setup(
                coarse_model, coarse_state0, train_schedule, u
            )
            predicted = _simulation_states(model, state0, schedule)
            return _trajectory_mismatch(
                predicted, train_observed, args.pressure_weight, args.saturation_weight, args.well_weight
            )
        except Exception as exc:
            print(f"[training] rejected non-convergent trial: {type(exc).__name__}: {exc}")
            return 1.0e30

    print("[4/5] Training CGNet global PV/T/WI parameters")
    train_u, train_loss, train_history = projected_gradient_minimize(
        training_objective,
        np.full(3, 0.5),
        args.train_iters,
        args.gradient_epsilon,
        args.gradient_step,
        "training",
    )
    trained_model, trained_state0, trained_schedule, trained_factors = _scaled_cgnet_setup(
        coarse_model, coarse_state0, coarse_schedule, train_u
    )
    print(f"       calibrated factors: {trained_factors}")

    print("[5/5] Optimising injection/production controls by projected gradient")
    optimization_schedule = copy.deepcopy(trained_schedule)
    if args.optimization_steps is not None:
        nopt = max(1, min(int(args.optimization_steps), nsteps))
        optimization_schedule["step"]["val"] = np.asarray(
            optimization_schedule["step"]["val"], dtype=float
        )[:nopt]
        optimization_schedule["step"]["control"] = np.asarray(
            optimization_schedule["step"]["control"], dtype=int
        )[:nopt]
        print(f"       optimisation horizon restricted to {nopt}/{nsteps} report steps")
    control_layout = _make_control_layout(optimization_schedule, args.control_periods, args.per_well_controls)
    base_control_schedule = _explode_schedule(optimization_schedule)
    baseline_states = _simulation_states(trained_model, trained_state0, base_control_schedule)
    baseline_npv = net_present_value(baseline_states, base_control_schedule, args)
    npv_scale = max(abs(baseline_npv), 1.0)

    def control_objective(u: np.ndarray) -> float:
        try:
            schedule = apply_control_vector(optimization_schedule, control_layout, u)
            states = _simulation_states(trained_model, trained_state0, schedule)
            return -net_present_value(states, schedule, args) / npv_scale
        except Exception as exc:
            print(f"[control] rejected non-convergent trial: {type(exc).__name__}: {exc}")
            return 1.0e30

    control_u, control_loss, control_history = projected_gradient_minimize(
        control_objective,
        np.full(len(control_layout), 0.5),
        args.control_iters,
        args.gradient_epsilon,
        args.gradient_step,
        "control",
    )
    optimized_schedule = apply_control_vector(optimization_schedule, control_layout, control_u)
    optimized_states = _simulation_states(trained_model, trained_state0, optimized_schedule)
    optimized_npv = net_present_value(optimized_states, optimized_schedule, args)
    print(f"       NPV: baseline={baseline_npv:.6e}, optimized={optimized_npv:.6e}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "cgnet_model.npz",
        partition=partition,
        porevolume=np.asarray(trained_model.operators["pv"], dtype=float),
        transmissibility=np.asarray(trained_model.operators["T"], dtype=float),
        training_unit_parameters=train_u,
        control_unit_parameters=control_u,
    )
    report = {
        "deck": deck_path,
        "reference_source": reference_source,
        "fine_cart_dims": fine_dims,
        "coarse_dims": coarse_dims,
        "fine_cells": _num_cells(fine_model),
        "cgnet_cells": _num_cells(trained_model),
        "training_steps": ntrain,
        "optimization_steps": len(optimized_schedule["step"]["val"]),
        "training_loss": train_loss,
        "training_unit_parameters": train_u,
        "calibrated_factors": trained_factors,
        "baseline_npv": baseline_npv,
        "optimized_npv": optimized_npv,
        "npv_improvement": optimized_npv - baseline_npv,
        "control_layout": control_layout,
        "control_unit_parameters": control_u,
        "training_history": train_history,
        "control_history": control_history,
    }
    (output_dir / "report.json").write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    (output_dir / "optimized_schedule.json").write_text(
        json.dumps(_json_ready(optimized_schedule), indent=2), encoding="utf-8"
    )
    print(f"[done] Wrote CGNet model and optimisation outputs to {output_dir}")
    return report


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
