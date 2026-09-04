"""Port of MRST ``readSaturationTest.m`` (mrst-2026a/hm/utils/observed).

Reads a saturation log into one table per well.

Expected columns::

    name  date  top  bottom  water  oil  gas

Unlike the profile survey, the depth interval arrives already split into
its own ``top``/``bottom`` columns.
"""

import numpy as _np

from ._tables import (group_by_well, parse_dates, read_sheets,
                      solve_key_similarities)

POSSIBLE_KEYS = (
    ('name', ('井号', '井名', 'wellname', 'name')),
    ('date', ('日期', '生产日期', '年月', 'date')),
    ('top', ('顶部深度', '测试顶深', '顶深', 'top')),
    ('bottom', ('底部深度', '测试底深', '底深', 'bottom')),
    ('water', ('含水饱和度', '含水', 'water', 'sw')),
    ('oil', ('含油饱和度', '含油', 'oil', 'so')),
    ('gas', ('含气饱和度', '含气', 'gas', 'sg')),
)


def readSaturationTest(fn):
    """Return ``[(well_name, table), ...]``."""
    out = []
    for sheet in read_sheets(fn):
        table = solve_key_similarities(sheet, POSSIBLE_KEYS)
        if 'name' not in table:
            continue
        if _np.asarray(table['name']).size == 0:
            continue
        table['date'] = parse_dates(table['date'])
        out.extend(group_by_well(table))
    return out
