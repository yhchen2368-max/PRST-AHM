"""Debug: check wells dict keys vs control well names."""
import sys
sys.path.insert(0, '.')

# Monkey-patch to intercept rebuild
import PRSTCore.ad_core.initialization.init_eclipse_problem_ad as init_mod

original_convert = init_mod._convert_deck_schedule_to_mrst

def debug_convert(model, deck, G=None, rock=None):
    import numpy as np
    # Call original but intercept before return
    sched = deck.get('SCHEDULE', {})
    wells = {}
    controls = []
    steps = []
    order = sched.get('_order', [])
    
    # Simplified: just check after completion processing
    result = original_convert(model, deck, G, rock)
    
    # Now check wells dict in memory (can't access from here, need different approach)
    return result

# Better: directly modify _process_well_completions to print
original_process = init_mod._process_well_completions

def debug_process(wells, G, rock):
    print(f'Before processing: wells.keys()={list(wells.keys())[:5]}')
    result = original_process(wells, G, rock)
    print(f'After processing: sample wells[INJE1].cells={len(wells.get("INJE1", {}).get("cells", []))}')
    print(f'After processing: sample wells[PRODU10].cells={len(wells.get("PRODU10", {}).get("cells", []))}')
    return result

init_mod._process_well_completions = debug_process

from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
s, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
print(f'\nFinal schedule control[0] well names: {[w["name"] for w in schedule["control"][0]["W"][:5]]}')
print(f'INJE1 in schedule: cells={len(schedule["control"][0]["W"][0].get("cells", []))}')
