"""Quick per-deck SPE1-variant check with a subprocess timeout."""
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DECKS = [
    "examples/SpE1/SPE1CASE1_INF.DATA",
    "examples/SpE1/SPE1CASE1_MID.DATA",
    "examples/SpE1/SPE1CASE2_2P.DATA",
    "examples/SpE1/SPE1CASE2_OILGAS.DATA",
    "examples/SpE1/SPE1CASE2_SLGOF.DATA",
    "examples/SpE1/SPE1CASE2_NOWELLS.DATA",
]

CHILD = r'''
import sys
sys.path.insert(0, r"__ROOT__")
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
from PRSTCore.ad_core.solvers import AMGCL_CPRSolverBlockAD
from PRSTCore.ad_core.solvers.simulate_schedule import simulate_schedule
deck = r"__DECK__"
try:
    state0, model, schedule, solver = init_eclipse_problem_ad(deck, RemoveZeroPoreVolume=True)
except Exception as exc:
    print("INIT-ERROR %s: %s" % (type(exc).__name__, exc)); sys.exit(0)
print("cells=%d steps=%d" % (len(state0["pressure"]), len(schedule["step"]["val"])))
solver.useLinesearch = True; solver.enforceResidualDecrease = True; solver.acceptanceFactor = 2.0
solver.LinearSolver = AMGCL_CPRSolverBlockAD(tolerance=1e-4, maxIterations=50,
                                             strategy="mrst", decoupling="trueIMPES")
try:
    res = simulate_schedule(model, state0, schedule, solver, max_steps=4)
    print("OK steps=%d" % len(res.get("steps", [])))
    for i, st in enumerate(res.get("steps", [])):
        print("  step %d conv=%s" % (i + 1, st.get("converged")))
except Exception as exc:
    print("RUN-ERROR %s: %s" % (type(exc).__name__, exc))
'''

for rel in DECKS:
    deck = os.path.join(ROOT, rel).replace("\\", "/")
    code = CHILD.replace("__ROOT__", ROOT.replace("\\", "/")).replace("__DECK__", deck)
    fd, path = tempfile.mkstemp(suffix=".py", dir=ROOT)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=45, cwd=ROOT)
        out = (p.stdout + p.stderr).strip().replace("\n", " | ")
    except subprocess.TimeoutExpired:
        out = "TIMEOUT (>45s)"
    os.remove(path)
    print("%-38s %5.1fs  %s" % (os.path.basename(rel)[:38], time.time() - t0, out),
          flush=True)
