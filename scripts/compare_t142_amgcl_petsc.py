"""Compare T142 AMGCL block CPR vs PETSc baseline: speed + results.

Speed:  timing_amgcl.csv  vs  timing_baseline.csv
Result: well_rates_amgcl.csv vs well_rates_baseline.csv
"""
import csv
import sys
import numpy as np

BASE = r"results\T142_full"


def read_timing(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({"step": int(r["step"]), "wall": float(r["wall_s"]),
                             "newton": int(r["newton_iters"])})
            except (KeyError, ValueError):
                continue
    return rows


import re

_AMGCL_LOG = re.compile(r"step (\d+)/\d+: conv=(\w+) iters=(\d+) wall=([\d.]+) s")


def read_amgcl_timing_from_log(path):
    """The AMGCL runner's timing.csv is buffered (short rows, never flushed),
    but the log is flushed per step -- parse it for wall/iterations instead."""
    rows = []
    with open(path) as f:
        for line in f:
            m = _AMGCL_LOG.search(line)
            if m:
                rows.append({"step": int(m.group(1)), "wall": float(m.group(4)),
                             "newton": int(m.group(3))})
    return rows


def read_rates(path):
    """{step: {well: dict}} with rates in Sm3/d and bhp in bar."""
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                step = int(r["step"])
            except ValueError:
                continue
            w = r["well"]
            data.setdefault(step, {})[w] = {
                "qO": float(r["qO_sm3d"]), "qW": float(r["qW_sm3d"]),
                "qG": float(r["qG_sm3d"]), "bhp": float(r["bhp_bar"]),
                "status": int(r["status"]),
            }
    return data


def field_totals(rates, step):
    qO = qW = qG = 0.0
    for w, d in rates.get(step, {}).items():
        if d["status"]:
            qO += d["qO"]
            qW += d["qW"]
            qG += d["qG"]
    return qO, qW, qG


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    ta = read_amgcl_timing_from_log(BASE + r"\full_run_amgcl_log.txt")
    tp = read_timing(BASE + (r"\timing_%s.csv" % ref))
    if not ta:
        print("AMGCL log timing missing/empty")
        return 1
    print("PETSc reference: %s" % ref)

    # restrict to the first 188 steps for a partial-run comparison
    ta = [t for t in ta if t["step"] <= 188]
    tp = [t for t in tp if t["step"] <= 188]

    # ---- speed ----
    sum_a = sum(t["wall"] for t in ta)
    sum_p = sum(t["wall"] for t in tp)
    it_a = sum(t["newton"] for t in ta)
    it_p = sum(t["newton"] for t in tp)
    print("=== SPEED ===")
    print("steps: AMGCL=%d  PETSc=%d" % (len(ta), len(tp)))
    print("total wall : AMGCL=%.1f s (%.1f min)   PETSc=%.1f s (%.1f min)" % (sum_a, sum_a / 60, sum_p, sum_p / 60))
    print("speedup    : %.2fx" % (sum_p / max(sum_a, 1e-9)))
    print("newton iters: AMGCL=%d  PETSc=%d  (ratio %.2fx)" % (it_a, it_p, it_a / max(it_p, 1)))
    print("avg wall/step: AMGCL=%.2f s  PETSc=%.2f s" % (sum_a / len(ta), sum_p / len(tp)))

    # per-step comparison over the common range
    common = min(len(ta), len(tp))
    walls_a = np.array([t["wall"] for t in ta[:common]])
    walls_p = np.array([t["wall"] for t in tp[:common]])
    print("\nfirst %d steps: AMGCL total=%.1f s  PETSc total=%.1f s  ratio=%.2fx"
          % (common, walls_a.sum(), walls_p.sum(), walls_p.sum() / max(walls_a.sum(), 1e-9)))
    print("  AMGCL faster in %d/%d steps" % ((walls_a < walls_p).sum(), common))

    # ---- results ----
    print("\n=== RESULTS (well_rates) ===")
    ra = read_rates(BASE + r"\well_rates_amgcl.csv")
    rp = read_rates(BASE + r"\well_rates_baseline.csv")
    common_steps = sorted(set(ra) & set(rp))
    if not common_steps:
        print("no common steps to compare rates")
        return 1
    # field totals per step
    dO = []
    dW = []
    dG = []
    for s in common_steps:
        qOa, qWa, qGa = field_totals(ra, s)
        qOp, qWp, qGp = field_totals(rp, s)
        dO.append(qOa - qOp)
        dW.append(qWa - qWp)
        dG.append(qGa - qGp)
    dO, dW, dG = np.asarray(dO), np.asarray(dW), np.asarray(dG)
    # relative to PETSc magnitude
    print("common steps=%d" % len(common_steps))
    for name, d, label in (("qO", dO, "oil rate"), ("qW", dW, "water rate"), ("qG", dG, "gas rate")):
        scale = np.abs(np.array([field_totals(rp, s)[{"qO":0,"qW":1,"qG":2}[name]] for s in common_steps]))
        scale = np.maximum(scale, 1e-12)
        rel = np.abs(d) / scale
        print("  %-4s field diff: max abs=%.4e  mean abs=%.4e  max rel=%.3f%%"
              % (name, np.abs(d).max(), np.abs(d).mean(), 100 * rel.max()))
    # per-well final-step comparison
    last = common_steps[-1]
    wells = sorted(set(ra[last]) & set(rp[last]))
    maxdiff = 0.0
    for w in wells:
        for key in ("qO", "qW", "qG", "bhp"):
            va = ra[last][w].get(key, 0.0)
            vp = rp[last][w].get(key, 0.0)
            maxdiff = max(maxdiff, abs(va - vp))
    print("  final step %d: per-well max |AMGCL-PETSc| over qO/qW/qG/bhp = %.6e" % (last, maxdiff))
    return 0


if __name__ == "__main__":
    sys.exit(main())
