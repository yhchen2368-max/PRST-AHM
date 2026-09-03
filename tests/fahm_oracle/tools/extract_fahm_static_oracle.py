"""Extract deterministic source, UI and callback facts from ``FAHM.mlapp``.

This tool is deliberately independent of PRST's implementation.  It reads the
authoritative MRST App Designer artifact and its extracted ``FAHM.m`` source,
then writes canonical JSON files that can be reviewed and compared byte for
byte.  It does not import or modify any PRST algorithm module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree


_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise_matlab_source(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    path.write_bytes(encoded)


def _extract_document_code(document_xml: bytes) -> str:
    root = ElementTree.fromstring(document_xml)
    text_nodes = root.findall(f".//{{{_W_NS}}}t")
    if not text_nodes:
        raise ValueError("FAHM.mlapp matlab/document.xml has no w:t code node")
    return "".join(node.text or "" for node in text_nodes)


def _xml_leaf_values(payload: bytes) -> dict[str, str]:
    root = ElementTree.fromstring(payload)
    values: dict[str, str] = {}
    for node in root.iter():
        if len(node) == 0 and node.text and node.text.strip():
            values[node.tag.rsplit("}", 1)[-1]] = node.text.strip()
    return values


def _function_lines(lines: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    pattern = re.compile(
        r"^\s*function\s+(?:(?:\[[^]]+\]|[A-Za-z_]\w*)\s*=\s*)?"
        r"([A-Za-z_]\w*)\s*\("
    )
    for line_number, line in enumerate(lines, 1):
        match = pattern.match(line)
        if match:
            result.setdefault(match.group(1), line_number)
    return result


def _public_components(lines: list[str]) -> list[dict[str, object]]:
    start = next(
        index
        for index, line in enumerate(lines)
        if line.strip() == "properties (Access = public)"
    )
    declarations: list[dict[str, object]] = []
    declaration = re.compile(
        r"^\s{8}([A-Za-z_]\w*)\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$"
    )
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() == "end":
            break
        match = declaration.match(line)
        if match:
            declarations.append(
                {
                    "declared_type": match.group(2),
                    "name": match.group(1),
                    "source_line": index + 1,
                }
            )

    create_start = next(
        index for index, line in enumerate(lines) if "function createComponents(app)" in line
    )
    create_end = next(
        index
        for index in range(create_start + 1, len(lines))
        if re.search(r"\bfunction\s+app\s*=\s*FAHM(?:\s*\(|\s*$)", lines[index])
    )
    assignment = re.compile(
        r"^\s*app\.([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?\s*=\s*(.+);\s*$"
    )
    assignments: dict[str, list[dict[str, object]]] = {}
    for index in range(create_start, create_end):
        match = assignment.match(lines[index])
        if not match:
            continue
        name, attribute, expression = match.groups()
        assignments.setdefault(name, []).append(
            {
                "attribute": attribute or "<constructor>",
                "expression": expression.strip(),
                "source_line": index + 1,
            }
        )

    for item in declarations:
        item["assignments"] = assignments.get(str(item["name"]), [])
    return declarations


def _callback_trace(lines: list[str]) -> dict[str, object]:
    handlers = _function_lines(lines)
    binding_pattern = re.compile(
        r"^\s*app\.([A-Za-z_]\w*)\.([A-Za-z_]\w*Fcn)\s*=\s*"
        r"createCallbackFcn\(app,\s*@([A-Za-z_]\w*),\s*(true|false)\);"
    )
    bindings: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        match = binding_pattern.match(line)
        if match:
            control, event, handler, passes_event = match.groups()
            bindings.append(
                {
                    "binding_line": line_number,
                    "control": control,
                    "event": event,
                    "handler": handler,
                    "handler_line": handlers.get(handler),
                    "passes_event": passes_event == "true",
                }
            )

    flows = [
        {
            "id": "FAHM-EVENT-STARTUP",
            "classification": "PARITY",
            "steps": [
                "FAHM constructor",
                "createComponents",
                "registerApp",
                "runStartupFcn(startupFcn)",
                "StartupPanel.Visible=on",
                "MainTabGroup.Visible=off",
                "pause(20)",
                "StartupPanel.Visible=off",
                "MainTabGroup.Visible=on",
            ],
        },
        {
            "id": "FAHM-EVENT-CREATE-PROJECT",
            "classification": "PARITY",
            "handler": "CreatProjectButtonPushed",
            "steps": [
                "read/process Eclipse deck",
                "recreate baseCase",
                "write NOSIM/tNavigator deck",
                "run external simulator",
                "read EGRID/INIT/UNRST",
                "construct G/rock/fluid/model/state0",
                "validate model and enable setup navigation",
            ],
        },
        {
            "id": "FAHM-EVENT-START-OPTIMIZATION",
            "classification": "PARITY",
            "handler": "StartButtonPushed",
            "steps": [
                "process monitoring data",
                "merge observation and report times",
                "construct observed/config/ModelParameter/pvec",
                "construct evaluateMatchFromEclipseRun callback",
                "call optimizeBoundConstrainedForFAHM",
                "for each trial: external forward + objective + adjoint gradient",
                "save caseN and history.mat",
            ],
        },
        {
            "id": "FAHM-EVENT-PRST-EXTENSIONS",
            "classification": "PRST_EXTENSION",
            "steps": [
                "Terminate may use PRST behavior",
                "Mismatch result loading may use PRST behavior",
                "View may use PRST behavior",
            ],
        },
    ]
    return {"callback_bindings": bindings, "key_flows": flows}


def extract(workspace_root: Path, output_dir: Path) -> None:
    app_dir = workspace_root / "MRST" / "dev" / "APP"
    mlapp_path = app_dir / "FAHM.mlapp"
    matlab_path = app_dir / "FAHM.m"
    matlab_raw = matlab_path.read_bytes()
    matlab_text = matlab_raw.decode("utf-8-sig")

    with zipfile.ZipFile(mlapp_path) as archive:
        names = sorted(archive.namelist())
        payloads = {name: archive.read(name) for name in names}
        members = []
        for name in names:
            payload = payloads[name]
            info = archive.getinfo(name)
            members.append(
                {
                    "crc32": f"{info.CRC:08x}",
                    "path": name,
                    "sha256": _sha256_bytes(payload),
                    "size": info.file_size,
                }
            )
        document_xml = payloads["matlab/document.xml"]

    app_metadata = _xml_leaf_values(payloads["metadata/appMetadata.xml"])
    core_metadata = _xml_leaf_values(payloads["metadata/coreProperties.xml"])
    release_metadata = _xml_leaf_values(payloads["metadata/mwcorePropertiesReleaseInfo.xml"])
    screenshot = payloads["metadata/appScreenshot.png"]
    if screenshot[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("FAHM app screenshot is not a PNG")
    screenshot_width, screenshot_height = struct.unpack(">II", screenshot[16:24])

    document_code = _extract_document_code(document_xml)
    normal_m = _normalise_matlab_source(matlab_text)
    normal_mlapp = _normalise_matlab_source(document_code)
    if normal_m != normal_mlapp:
        raise ValueError("FAHM.m does not match code embedded in FAHM.mlapp")

    lines = normal_m.split("\n")
    components = _public_components(lines)
    event_trace = _callback_trace(lines)
    fingerprint = {
        "artifact": "MRST/dev/APP/FAHM.mlapp",
        "embedded_code_matches_fahm_m": True,
        "embedded_code_normalized_sha256": _sha256_bytes(normal_mlapp.encode("utf-8")),
        "fahm_m": {
            "logical_lines": len(lines),
            "path": "MRST/dev/APP/FAHM.m",
            "sha256": _sha256_bytes(matlab_raw),
            "size": len(matlab_raw),
        },
        "format": "MATLAB App Designer MLAPPVersion 2",
        "function_count": sum(1 for line in lines if re.match(r"^\s*function\b", line)),
        "app_metadata": {
            "created_utc": core_metadata["created"],
            "minimum_supported_matlab_release": app_metadata["minimumSupportedMATLABRelease"],
            "mlapp_version": app_metadata["MLAPPVersion"],
            "modified_utc": core_metadata["modified"],
            "saved_by_matlab_release": release_metadata["release"],
            "screenshot_height": screenshot_height,
            "screenshot_width": screenshot_width,
            "title": core_metadata["title"],
            "uuid": app_metadata["uuid"],
        },
        "mlapp": {
            "path": "MRST/dev/APP/FAHM.mlapp",
            "sha256": _sha256_bytes(mlapp_path.read_bytes()),
            "size": mlapp_path.stat().st_size,
            "zip_members": members,
        },
        "public_component_count": len(components),
        "callback_binding_count": len(event_trace["callback_bindings"]),
        "reference_policy": "FAHM.mlapp is the sole product reference; FAHM.m is its exact searchable code extraction.",
    }

    _write_json(output_dir / "source_fingerprint.json", fingerprint)
    _write_json(
        output_dir / "ui_controls.json",
        {
            "components": components,
            "source": "MRST/dev/APP/FAHM.m public App component properties and createComponents assignments",
        },
    )
    _write_json(output_dir / "event_trace.json", event_trace)


def main() -> int:
    default_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=default_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            default_root
            / "RSTCore-main"
            / "tests"
            / "fixtures"
            / "fahm_oracle"
            / "v1"
            / "static"
        ),
    )
    args = parser.parse_args()
    extract(args.workspace_root.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
