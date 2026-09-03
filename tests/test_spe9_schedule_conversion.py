from PRSTCore.ad_core.initialization.init_eclipse_problem_ad import init_eclipse_problem_ad
import numpy as np


def test_spe9_schedule_has_controls_and_wells():
    _, _, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
    assert len(schedule['step']['val']) == 90
    assert len(schedule['control']) == 3
    assert len(schedule['control'][0]['W']) >= 20


def test_spe9_wellspecs_mapping_fields():
    _, _, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
    wells = schedule['control'][0]['W']
    inj = next(w for w in wells if w['name'] == 'INJE1')
    assert inj['i'] == 24
    assert inj['j'] == 25
    # SPE9_CP.DATA is a FIELD-unit deck; refDepth is converted to SI (metres)
    # like every other length in this pipeline, so 9110 ft -> 9110*0.3048 m.
    assert abs(inj['refDepth'] - 9110.0 * 0.3048) < 1e-9
    assert inj['phase'] == 'WATER'


def test_spe9_control_sequence_changes():
    _, _, schedule, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
    # MRST-like structure should reference all 3 controls over 90 report-steps
    used = sorted(set(int(v) for v in schedule['step']['control']))
    assert used == [0, 1, 2]


def test_spe9_stage1_pvt_rs_rv_is_wired_into_model():
    state0, model, _, _ = init_eclipse_problem_ad('examples/SPE9/SPE9_CP.DATA')
    p = np.asarray(state0['pressure'], dtype=float).ravel()

    assert hasattr(model, 'bo') and callable(model.bo)
    assert hasattr(model, 'bw') and callable(model.bw)
    assert hasattr(model, 'bg') and callable(model.bg)
    assert hasattr(model, 'rs') and callable(model.rs)
    assert hasattr(model, 'rv') and callable(model.rv)

    bo = np.asarray(model.bo(p), dtype=float).ravel()
    bw = np.asarray(model.bw(p), dtype=float).ravel()
    bg = np.asarray(model.bg(p), dtype=float).ravel()
    rs = np.asarray(model.rs(p), dtype=float).ravel()
    rv = np.asarray(model.rv(p), dtype=float).ravel()

    assert bo.size == p.size
    assert bw.size == p.size
    assert bg.size == p.size
    assert rs.size == p.size
    assert rv.size == p.size
    assert np.all(np.isfinite(bo)) and np.all(bo > 0)
    assert np.all(np.isfinite(bw)) and np.all(bw > 0)
    assert np.all(np.isfinite(bg)) and np.all(bg > 0)
    assert np.all(np.isfinite(rs)) and np.all(rs >= 0)
    assert np.all(np.isfinite(rv)) and np.all(rv >= 0)
    # MRST black-oil initialization: oil-present/no-free-gas cells start at RsSat.
    assert np.max(rs) > 0.0
