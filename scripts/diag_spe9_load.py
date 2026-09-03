"""Phase-timed deck load for SPE9 to find where loading hangs."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

deck_path = os.path.join(ROOT, "examples", "SPE9", "SPE9.DATA")

def stage(name):
    print("[%6.1fs] %s" % (time.time() - t0, name), flush=True)

t0 = time.time()
from PRSTCore.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
deck = read_eclipse_deck(deck_path)
stage("read_eclipse_deck done (%d sections)" % len(deck))

from PRSTCore.deckformat.deckinput.convert_deck_units import convert_deck_units
deck = convert_deck_units(deck)
stage("convert_deck_units done")

import PRSTCore.ad_core.initialization.init_eclipse_problem_ad as mod
opts = dict(
    useMex=False, useMexGeometry=None, useMexProcessGrid=None,
    TimestepStrategy='iteration', useCPR=True, rowMajorAD=False,
    AutoDiffBackend=None, UniformFacilityModel=False, maxIterations=12,
    useRelaxation=True, model=None, G=None, getSchedule=True,
    getInitialState=True, RemoveZeroPoreVolume=True)
model = mod._initialize_model(deck, opts)
stage("_initialize_model done")

state0 = mod._init_state_deck(model, deck)
stage("_init_state_deck done")

G = getattr(model, "G", None)
rock = getattr(model, "rock", None)
schedule = mod._convert_deck_schedule_to_mrst(model, deck, G=G, rock=rock)
stage("_convert_deck_schedule_to_mrst done (%d controls)" % len(schedule["control"]))

print("nc = %d" % len(state0["pressure"]), flush=True)
