"""Create a 2-phase SPE10_MODEL2 deck (the deck declares GAS only for the
3-phase 'Flo' simulator; there is no gas PVT, so the gas phase is degenerate
and makes the 3x3 cell blocks near-singular)."""
import re

SRC = 'examples/spe10model2/SPE10_MODEL2.DATA'
DST = 'examples/spe10model2/SPE10_MODEL2_2P.DATA'

with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.read().splitlines()

out = []
i = 0
skipped = []
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    if re.match(r'^GAS\s*$', stripped):
        skipped.append(('GAS', i + 1))
        i += 1
        continue
    if re.match(r'^SGOF\s*$', stripped):
        # skip through the terminating '/' (the last / line before DENSITY)
        start = i
        i += 1
        while i < len(lines) and not re.search(r'/\s*$', lines[i]):
            i += 1
        if i < len(lines):
            i += 1  # consume the '/' line too
        skipped.append(('SGOF', start + 1))
        continue
    if re.match(r'^DENSITY', stripped):
        # two-phase DENSITY: keep only the first two entries
        new = re.sub(r'\s+[0-9.eE+-]+\s*/\s*$', '  /', stripped)
        new = re.sub(r'^(\s*DENSITY\s+\S+\s+\S+).*$', r'\1  /', stripped)
        out.append(new)
        i += 1
        continue
    out.append(line)
    i += 1

with open(DST, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')

print('wrote %s (%d lines, %d skipped: %s)'
      % (DST, len(out), len(skipped), skipped))
