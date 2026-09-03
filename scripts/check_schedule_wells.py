"""Check schedule['control'][0]['W'] directly."""
import sys
sys.path.insert(0, '.')
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

s, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
print(f'Total controls: {len(schedule["control"])}')
print(f'Control 0 has {len(schedule["control"][0]["W"])} wells')
for w in schedule['control'][0]['W'][:3]:
    print(f'{w.get("name"):10s}: cells={len(w.get("cells", []))} WI={len(w.get("WI", []))}')
    if len(w.get("cells", [])) > 0:
        print(f'  sample: {w["cells"][:3]}')
