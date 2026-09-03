"""Check well perforation cells from deck."""
import sys
sys.path.insert(0, '.')
from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad

s, model, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
forces = model.getDrivingForces(schedule['control'][0])
wells = [w for w in forces.get('W', []) if w.get('status')]
print(f'Total wells: {len(wells)}')
for w in wells[:5]:
    cells = w.get('cells', [])
    WI = w.get('WI', [])
    print(f'{w.get("name"):10s}: cells={len(cells)} WI_len={len(WI)} sign={w.get("sign")} val={w.get("val")}')
    if len(cells) > 0:
        print(f'  sample cells: {cells[:3]}')
