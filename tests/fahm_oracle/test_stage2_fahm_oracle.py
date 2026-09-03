from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.fahm_oracle.tools.extract_fahm_static_oracle import extract
from tests.fahm_oracle.tools.verify_fahm_oracle import validate_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "fahm_oracle" / "v1"
STATIC_ROOT = FIXTURE_ROOT / "static"
GOLDEN_ROOT = FIXTURE_ROOT / "golden"
CONTRACT_ROOT = FIXTURE_ROOT / "contracts"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _array(manifest: dict[str, object], name: str) -> np.ndarray:
    item = next(item for item in manifest["arrays"] if item["name"] == name)
    values = np.fromfile(GOLDEN_ROOT / item["path"], dtype=np.dtype(item["dtype"]))
    return values.reshape(tuple(item["shape"]), order=item["order"])


def test_mlapp_and_m_fingerprint_is_frozen():
    fingerprint = _json(STATIC_ROOT / "source_fingerprint.json")
    assert fingerprint["embedded_code_matches_fahm_m"] is True
    assert fingerprint["embedded_code_normalized_sha256"] == (
        "136abf2f91b5f1573314c015d07de22e305c480313c4986bdd5f323453e96367"
    )
    assert fingerprint["mlapp"]["sha256"] == (
        "fa1fbccb885136fd96a407b475b4dcaabb9189b00a992fe945e674e6ae481668"
    )
    assert fingerprint["fahm_m"]["sha256"] == (
        "a1c29aedd7620438b94bbbcb529c7d9fb6fd62fe909def6d419a10385fb493c7"
    )
    assert fingerprint["fahm_m"]["logical_lines"] == 4404
    assert fingerprint["function_count"] == 119
    assert fingerprint["public_component_count"] == 373
    assert fingerprint["callback_binding_count"] == 95
    assert fingerprint["app_metadata"] == {
        "created_utc": "2023-06-13T01:27:03Z",
        "minimum_supported_matlab_release": "R2018a",
        "mlapp_version": "2",
        "modified_utc": "2026-01-06T07:28:00Z",
        "saved_by_matlab_release": "R2024b",
        "screenshot_height": 736,
        "screenshot_width": 915,
        "title": "FROSIT",
        "uuid": "f78846df-4381-4897-8e25-fd9e7235e3c7",
    }


def test_static_oracle_regenerates_byte_identically(tmp_path):
    regenerated = tmp_path / "static"
    extract(WORKSPACE_ROOT, regenerated)
    assert _tree_bytes(regenerated) == _tree_bytes(STATIC_ROOT)


def test_ui_control_inventory_contains_fahm_product_shell():
    inventory = _json(STATIC_ROOT / "ui_controls.json")
    components = {item["name"]: item for item in inventory["components"]}
    required = {
        "UIFigure",
        "MainTabGroup",
        "SetUpTab",
        "RunTab",
        "MismatchTab",
        "ViewTab",
        "ModelTab",
        "ObjectiveTab",
        "ParameterTab",
        "CreatProjectButton",
        "StartButton",
        "TerminateButton",
    }
    assert required <= components.keys()
    assert len(components) == 373
    assert all(item["declared_type"].startswith("matlab.ui.") for item in components.values())
    assert all(item["assignments"] for item in components.values())


def test_callback_bindings_and_key_event_flows_are_frozen():
    trace = _json(STATIC_ROOT / "event_trace.json")
    bindings = {
        (item["control"], item["event"], item["handler"])
        for item in trace["callback_bindings"]
    }
    assert ("CreatProjectButton", "ButtonPushedFcn", "CreatProjectButtonPushed") in bindings
    assert ("StartButton", "ButtonPushedFcn", "StartButtonPushed") in bindings
    assert not any(item[0] == "TerminateButton" for item in bindings)
    assert all(item["handler_line"] is not None for item in trace["callback_bindings"])
    flows = {item["id"]: item["classification"] for item in trace["key_flows"]}
    assert flows == {
        "FAHM-EVENT-STARTUP": "PARITY",
        "FAHM-EVENT-CREATE-PROJECT": "PARITY",
        "FAHM-EVENT-START-OPTIMIZATION": "PARITY",
        "FAHM-EVENT-PRST-EXTENSIONS": "PRST_EXTENSION",
    }


def test_deviation_registry_has_only_explicit_classifications():
    registry = _json(CONTRACT_ROOT / "deviations.json")
    allowed = set(registry["allowed_classifications"])
    entries = registry["entries"]
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids))
    assert {entry["classification"] for entry in entries} <= allowed
    assert {
        "FAHM-FIX-001",
        "FAHM-FIX-002",
        "FAHM-FIX-003",
        "FAHM-FIX-004",
        "FAHM-FIX-005",
        "FAHM-FIX-006",
        "FAHM-FIX-007",
        "FAHM-FIX-008",
        "FAHM-FIX-009",
        "FAHM-FIX-010",
        "FAHM-FIX-011",
        "FAHM-EXT-001",
        "FAHM-EXT-002",
        "FAHM-EXT-003",
        "FAHM-EXT-004",
    } <= set(ids)
    assert all(entry["source"] and entry["prst_action"] for entry in entries)


def test_golden_fixture_schema_files_and_hashes_are_valid():
    manifest = validate_fixture(GOLDEN_ROOT)
    assert manifest["oracle_id"] == "FAHM-SPE1-WOG-FIRST-3-STEPS"
    assert manifest["active_phase_order"] == "WOG"
    assert manifest["model_class"] == "GenericBlackOilModel"
    assert len(manifest["arrays"]) == 105
    assert manifest["parameter_order"] == ["porevolume", "permx", "permy", "permz"]
    assert [row["name"] for row in manifest["config_rows"]] == [
        "porevolume",
        "permx",
        "permy",
        "permz",
        "krw",
        "kro",
        "krg",
        "swl",
        "swcr",
        "swu",
        "sowcr",
        "sgl",
        "sgcr",
        "sgu",
        "sogcr",
    ]
    assert [row["include"] for row in manifest["config_rows"]] == [True] * 4 + [False] * 11
    assert [row["scaling"] for row in manifest["config_rows"][:4]] == [
        "linear",
        "log",
        "log",
        "log",
    ]
    assert manifest["history_fields"] == [
        "val",
        "u",
        "pg",
        "alpha",
        "lsit",
        "lsfl",
        "hess",
        "rho",
        "r",
        "params",
    ]
    names = {item["name"] for item in manifest["arrays"]}
    for step in range(1, 4):
        assert f"forward/step_{step:02d}/pressure" in names
        assert f"forward/step_{step:02d}/saturation_WOG" in names
        assert f"observed/step_{step:02d}/well/qWs" in names
        assert f"observed/step_{step:02d}/well/bhp" in names


def test_index_column_major_and_per_cell_parameter_contract():
    manifest = validate_fixture(GOLDEN_ROOT)
    active1 = _array(manifest, "grid/active_index_map_1based")
    active0 = _array(manifest, "grid/active_index_map_0based")
    np.testing.assert_array_equal(active1 - 1, active0)
    cells1 = _array(manifest, "schedule/well_cells_1based")
    cells0 = _array(manifest, "schedule/well_cells_0based")
    np.testing.assert_array_equal(cells1 - 1, cells0)
    p2w1 = _array(manifest, "schedule/perforation_to_well_1based")
    p2w0 = _array(manifest, "schedule/perforation_to_well_0based")
    np.testing.assert_array_equal(p2w1 - 1, p2w0)

    permeability = _array(manifest, "rock/permeability")
    assert permeability.shape == (300, 3)
    item = next(item for item in manifest["arrays"] if item["name"] == "rock/permeability")
    flat = np.fromfile(GOLDEN_ROOT / item["path"], dtype=item["dtype"])
    np.testing.assert_array_equal(flat.reshape((300, 3), order="F"), permeability)

    np.testing.assert_array_equal(_array(manifest, "parameters/nparam").ravel(), [300] * 4)
    np.testing.assert_array_equal(
        _array(manifest, "parameters/slices_1based_inclusive"),
        [[1, 300], [301, 600], [601, 900], [901, 1200]],
    )
    np.testing.assert_array_equal(
        _array(manifest, "parameters/slices_0based_half_open"),
        [[0, 300], [300, 600], [600, 900], [900, 1200]],
    )
    pvec = _array(manifest, "parameters/pvec_unit_box")
    assert pvec.shape == (1200, 1)
    assert np.all((0 <= pvec) & (pvec <= 1))
    np.testing.assert_allclose(pvec, 0.5, rtol=0, atol=6e-16)


def test_objective_sign_and_per_step_values_are_frozen():
    manifest = validate_fixture(GOLDEN_ROOT)
    per_step = _array(manifest, "objective/per_step_positive_misfit")
    expected = np.array(
        [
            1.1852198324654768e-05,
            1.4880441550909137e-04,
            4.3973923158078626e-04,
        ]
    ).reshape((-1, 1))
    np.testing.assert_array_equal(per_step, expected)
    total = _array(manifest, "objective/total_positive_misfit")
    returned = _array(manifest, "objective/evaluator_return_value")
    np.testing.assert_array_equal(total, per_step.sum(keepdims=True))
    np.testing.assert_array_equal(returned, -total)


def test_history_schema_and_repeated_export_report_are_frozen():
    manifest = validate_fixture(GOLDEN_ROOT)
    np.testing.assert_array_equal(
        _array(manifest, "history/val"),
        [[-0.4050000000000001, -0.24499999999999997, -0.0]],
    )
    report = _json(FIXTURE_ROOT / "reproducibility.json")
    assert report["status"] == "PASS"
    assert report["run_count"] == 2
    assert report["exact_byte_comparison"] is True
    manifest_hash = hashlib.sha256((GOLDEN_ROOT / "manifest.json").read_bytes()).hexdigest()
    assert manifest_hash == report["manifest_sha256"]


def test_math_contract_forbids_scalar_parameter_and_c_order_substitutions():
    contract = _json(CONTRACT_ROOT / "math_contract.json")
    assert contract["reshape_and_storage"]["oracle_binary_order"] == "F"
    assert "order='F'" in contract["reshape_and_storage"]["numpy_required"]
    assert contract["unit_box"]["bounds"] == [0.0, 1.0]
    assert "never one scalar multiplier" in contract["parameterization"]["dimension"]
    assert contract["objective_and_gradient_sign"]["matchObservedOWGProfile"].startswith(
        "returns one non-negative"
    )
