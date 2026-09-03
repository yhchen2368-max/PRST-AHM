"""Display helper functions.

1:1 Python translations of MRST solvers/adjoint/dispControls.m and dispSchedule.m
"""

import numpy as np


def disp_controls(controls, schedule):
    """Display control variables."""
    print("\n----------------- DISPLAYING CONTROL VARIABLES ----------------")
    num_c = len(controls["well"])
    cw = [w["wellNum"] for w in controls["well"]]
    names = [schedule[0]["names"][wn] for wn in cw]
    types = [w["type"] for w in controls["well"]]
    min_max = np.array([w["minMax"] for w in controls["well"]])

    print(f"{'Var':>9s}{'Name':>9s}{'Type':>9s}{'MaxMin':>15s}")
    for k in range(num_c):
        print(f"{'u_'+str(k):>9s}{names[k]:>9s}{types[k]:>9s}{str(min_max[k]):>15s}")

    ec = controls.get("linEqConst")
    if ec is not None:
        print("\nLinear equality constraints:")
        A = np.atleast_2d(ec["A"])
        b = np.atleast_1d(ec["b"])
        for k in range(A.shape[0]):
            terms = []
            for k1 in range(A.shape[1]):
                if A[k, k1] != 0:
                    terms.append(f"{A[k,k1]}*u_{k1}")
            print(" + ".join(terms) + f" = {b[k]}")


def disp_schedule(schedule):
    """Display schedule."""
    print("\n----------------- DISPLAYING SCHEDULE ----------------")
    print(f"{'Step':>6s}{'Start':>10s}{'End':>10s}")
    for k, s in enumerate(schedule):
        ti = s["timeInterval"]
        vals = ", ".join(f"{v:.3g}" for v in s["values"])
        print(f"{k+1:6d}{ti[0]:10.3g}{ti[1]:10.3g}  vals=[{vals}]")
