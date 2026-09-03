"""Compare SPE9 first-20 steps: AMGCL block CPR vs PETSc CPR (from logs)."""
import re
import sys

logs = {
    "AMGCL block CPR": r"results\spe9_amgcl_block_full.log",
    "PETSc CPR       ": r"results\spe9_petsc_full.log",
}
pat = re.compile(r"step (\d+)/\d+: conv=(\w+) iters=(\d+) wall=([\d.]+) s")

data = {}
for label, path in logs.items():
    steps = {}
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps[int(m.group(1))] = (bool(m.group(2) == "True"),
                                          int(m.group(3)), float(m.group(4)))
    data[label] = steps

N = 20
print("%-16s %-28s %-28s" % ("step", "AMGCL iters/wall(s)", "PETSc iters/wall(s)"))
totals = {k: [0, 0.0, 0] for k in data}   # iters, wall, converged
for s in range(1, N + 1):
    row = []
    for label in data:
        if s in data[label]:
            conv, it, wall = data[label][s]
            row.append("%3d / %6.2f" % (it, wall))
            totals[label][0] += it
            totals[label][1] += wall
            totals[label][2] += 1 if conv else 0
        else:
            row.append("   -- /    -- ")
    print("step %-11d %-28s %-28s" % (s, row[0], row[1]))

print("-" * 72)
print("sum steps 1-%d: AMGCL  iters=%d wall=%.1f s | PETSc iters=%d wall=%.1f s"
      % (N, totals["AMGCL block CPR"][0], totals["AMGCL block CPR"][1],
         totals["PETSc CPR       "][0], totals["PETSc CPR       "][1]))
a = totals["AMGCL block CPR"]
p = totals["PETSc CPR       "]
print("PETSc/AMGCL wall ratio: %.2fx  (iters ratio %.2fx)"
      % (p[1] / max(a[1], 1e-9), p[0] / max(a[0], 1e-9)))
