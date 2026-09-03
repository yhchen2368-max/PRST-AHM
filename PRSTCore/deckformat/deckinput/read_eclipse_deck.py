"""Read ECLIPSE input deck (.DATA file).

1:1 Python translation of MRST model-io/deckformat/deckinput/readEclipseDeck.m
"""

import os
import re

from .initialize_deck import initialize_deck
from .read_grid import read_grid
from .read_props import read_props
from .read_regions import read_regions
from .read_runspec import read_runspec
from .read_schedule import read_schedule
from .read_solution import read_solution
from .read_summary import read_summary


def read_eclipse_deck(fn):
    """Read a simplified ECLIPSE input deck.

    Parameters
    ----------
    fn : str
        Path to .DATA file.

    Returns
    -------
    dict
        Deck structure with RUNSPEC, GRID, PROPS, REGIONS, SOLUTION, SCHEDULE.
    """
    content = _read_text_with_fallback(fn)

    dirname = os.path.dirname(os.path.abspath(fn))

    # Initialize deck
    deck = initialize_deck()

    # Process INCLUDE files
    content = _process_includes(content, dirname)

    # Tokenize into keyword blocks
    kw_blocks = _tokenize_deck(content)

    for kw, block in kw_blocks:
        kw_upper = kw.upper().strip()

        if kw_upper == "RUNSPEC":
            deck["RUNSPEC"] = read_runspec(block)
        elif kw_upper in ("GRID", "GRIDFILE"):
            deck["GRID"] = read_grid(
                block, deck.get("GRID", {}),
                cart_dims=deck.get("RUNSPEC", {}).get("cartDims"),
            )
        elif kw_upper == "PROPS":
            deck["PROPS"] = read_props(
                block, cart_dims=deck.get("RUNSPEC", {}).get("cartDims")
            )
        elif kw_upper == "REGIONS":
            deck["REGIONS"] = read_regions(
                block, cart_dims=deck.get("RUNSPEC", {}).get("cartDims")
            )
        elif kw_upper == "SOLUTION":
            deck["SOLUTION"] = read_solution(block)
        elif kw_upper == "SCHEDULE":
            # START turns DATES records into step lengths, exactly as
            # readSCHEDULE's `readDATES` uses deck.RUNSPEC.START.
            deck["SCHEDULE"] = read_schedule(
                block, start=deck.get("RUNSPEC", {}).get("START"))
        elif kw_upper == "SUMMARY":
            deck["SUMMARY"] = read_summary(block)
        elif kw_upper in ("ECHO", "NOECHO"):
            continue
        elif kw_upper == "END":
            break

    # Post-process
    if "GRID" in deck and "RUNSPEC" in deck and "cartDims" in deck["RUNSPEC"]:
        deck["GRID"]["cartDims"] = deck["RUNSPEC"]["cartDims"]
        if "ACTNUM" in deck["GRID"]:
            import numpy as np
            deck["GRID"]["ACTNUM"] = np.array(deck["GRID"]["ACTNUM"], dtype=np.int32).ravel()

    # Default unit system
    if not any(k in deck.get("RUNSPEC", {}) for k in ("METRIC", "FIELD", "LAB")):
        deck["RUNSPEC"]["METRIC"] = True

    # If GRID lacks COORD/ZCORN (common when GRDECL included), attempt to
    # extract them directly from the full file content as a fallback.
    if "GRID" in deck:
        if "COORD" not in deck["GRID"]:
            vals = _extract_grdecl_keyword(content, 'COORD')
            if vals:
                deck["GRID"]["COORD"] = vals
        if "ZCORN" not in deck["GRID"]:
            vals = _extract_grdecl_keyword(content, 'ZCORN')
            if vals:
                deck["GRID"]["ZCORN"] = vals
    # Extract SPECGRID dims if present
    if 'RUNSPEC' in deck and (not deck['RUNSPEC'].get('cartDims')):
        import re
        m = re.search(r'\bSPECGRID\b\s*(.*?)\/', content, re.IGNORECASE | re.DOTALL)
        if m:
            nums = re.findall(r"\d+", m.group(1))
            if len(nums) >= 3:
                try:
                    nx, ny, nz = int(nums[0]), int(nums[1]), int(nums[2])
                    deck['RUNSPEC']['cartDims'] = [nx, ny, nz]
                except Exception:
                    pass

    return deck


def _extract_grdecl_keyword(content: str, keyword: str):
    import re
    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b(.*?)/', re.IGNORECASE | re.DOTALL)
    m = pattern.search(content)
    if not m:
        return None
    block = m.group(1)
    # Tokenize numeric values from block
    toks = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[EeDd][-+]?[0-9]+)?", block)
    vals = [float(t.replace('D', 'E').replace('d', 'e')) for t in toks]
    return vals


# ECLIPSE itself allows nested includes; this only guards a cycle.
_MAX_INCLUDE_DEPTH = 16


def _initialize_deck():
    return {"RUNSPEC": {}, "GRID": {}, "PROPS": {}, "REGIONS": {}, "SOLUTION": {}, "SCHEDULE": {}}


def _process_includes(content, dirname, _depth=0, _seen=None):
    """Process INCLUDE statements with support for multiline syntax.

    Expansion recurses: an included file may itself contain INCLUDE
    statements, which real decks routinely use -- a GRID include that
    pulls in separate .grdecl, NTG and PERM files, for instance. Nested
    paths resolve against the *deck root*, not against the including
    file's directory, which is what ECLIPSE does.

    ``_depth``/``_seen`` guard against an include cycle rather than
    recursing until the stack gives out.
    """
    if _depth > _MAX_INCLUDE_DEPTH:
        raise ValueError('INCLUDE nesting deeper than %d levels; the deck '
                         'probably has an include cycle' % _MAX_INCLUDE_DEPTH)
    _seen = set() if _seen is None else _seen

    lines = content.splitlines()
    out = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            out.append(line)
            i += 1
            continue

        if stripped.upper().startswith('INCLUDE'):
            rest = stripped[len('INCLUDE'):].strip()
            inc_ref = None

            # Form 1: INCLUDE filename /
            if rest:
                # Remove a trailing statement terminator slash, but keep
                # internal path separators used by relative include paths.
                m = re.match(r"^\s*(['\"]?)(.+?)\1\s*/?\s*$", rest)
                if m:
                    rest = m.group(2)
                inc_ref = rest.strip().strip("'").strip('"')
            else:
                # Form 2:
                # INCLUDE
                # filename
                # /
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                if j < n:
                    cand = lines[j].strip()
                    m = re.match(r"^\s*(['\"]?)(.+?)\1\s*/?\s*$", cand)
                    if m:
                        cand = m.group(2)
                    inc_ref = cand.strip().strip("'").strip('"')
                    i = j
                    # Consume optional trailing slash-only line.
                    if i + 1 < n and lines[i + 1].strip() == '/':
                        i += 1

            if inc_ref:
                inc_path = os.path.join(dirname, inc_ref)
                key = os.path.normcase(os.path.abspath(inc_path))
                if os.path.exists(inc_path) and key not in _seen:
                    nested = _read_text_with_fallback(inc_path)
                    out.append(_process_includes(nested, dirname,
                                                 _depth + 1, _seen | {key}))
                elif os.path.exists(inc_path):
                    # Already on this include chain: expanding again would
                    # loop. Leave the statement in place rather than
                    # silently dropping the reference.
                    out.append(line)
                else:
                    out.append(line)
            else:
                out.append(line)

            i += 1
            continue

        out.append(line)
        i += 1

    return '\n'.join(out)


def _read_text_with_fallback(path):
    # Norne/legacy decks may not match platform default codepage.
    encodings = ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'gbk')
    last_err = None
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    with open(path, 'r') as f:
        return f.read()


def _tokenize_deck(content):
    """Split deck content into keyword blocks."""
    import re
    lines = content.split("\n")
    blocks = []
    current_kw = None
    current_block = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("--"):
            continue  # comment

        # Check if this line starts a new keyword *section*.  ECLIPSE has
        # several ordinary RUNSPEC keywords beginning with ``GRID``
        # (notably GRIDOPTS and GRIDFILE).  MRST's readRUNSPEC leaves that
        # section only when it receives the exact GRID keyword; a prefix
        # match wrongly drops all following phase flags in Norne.
        head = stripped.split(None, 1)[0].upper()
        for section in ("RUNSPEC", "GRID", "PROPS", "REGIONS", "SOLUTION", "SCHEDULE",
                         "SUMMARY", "END", "GRIDFILE"):
            if head == section:
                if current_kw:
                    blocks.append((current_kw, "\n".join(current_block)))
                current_kw = section
                current_block = [stripped]
                break
        else:
            # ECHO/NOECHO toggle lines inside sections: treat as continuation
            if stripped.upper() in ('ECHO', 'NOECHO'):
                if current_kw:
                    current_block.append(stripped)
                continue
            if current_kw:
                current_block.append(stripped)

    if current_kw:
        blocks.append((current_kw, "\n".join(current_block)))

    return blocks


