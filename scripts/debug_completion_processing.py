"""Debug: check if _process_well_completions is called and completions exist."""
import sys
sys.path.insert(0, '.')
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

# Monkey-patch to add debug output
import PRSTCore.ad_core.initialization.init_eclipse_problem_ad as init_mod

original_process = init_mod._process_well_completions

def debug_process(wells, G, rock):
    print(f'_process_well_completions called with {len(wells)} wells')
    for wname, w in list(wells.items())[:3]:
        print(f'  {wname}: completions={len(w.get("completions", []))} cells_before={len(w.get("cells", []))}')
    result = original_process(wells, G, rock)
    for wname, w in list(wells.items())[:3]:
        print(f'  {wname}: cells_after={len(w.get("cells", []))} WI_after={len(w.get("WI", []))}')
    return result

init_mod._process_well_completions = debug_process

s, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
print('\nFinal wells in schedule:')
for w in schedule['control'][0]['W'][:3]:
    print(f'  {w["name"]}: cells={len(w.get("cells", []))} WI={len(w.get("WI", []))}')
