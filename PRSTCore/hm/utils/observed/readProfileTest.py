"""Port of MRST ``readProfileTest.m`` (mrst-2026a/hm/utils/observed).

Reads a production-logging (profile) survey into one table per well.

Expected columns::

    name  date  depth  rate...

where ``depth`` is a text interval such as ``1200-1250``, split into
``top``/``bottom``. A blank date carries the previous one forward -- the
sheets mark a multi-interval survey by writing the date only on its first
row. Missing rates become zero.
"""

import numpy as _np

from ._tables import (fill_missing_with, group_by_well, parse_dates,
                      read_sheets, solve_key_similarities,
                      split_depth_interval)

POSSIBLE_KEYS = (
    ('name', ('井号', '井名', '标准井号', '标准井名')),
    ('date', ('测井时间', '测井日期', '时间', '日期')),
    ('depth', ('解释井段', '测试井段', '顶底深')),
    ('qW', ('实产水', '日产水', '实注', '实注水', '日注', '日注水')),
    ('qO', ('实产油', '日产油')),
    ('qG', ('实产气', '日产气')),
    ('cqW', ('绝对产水量', '相对产水量', '绝对吸水量', '相对吸水量')),
    ('cqO', ('绝对产油量', '相对产油量')),
    ('cqG', ('绝对产气量', '相对产气量')),
)

RATE_COLUMNS = ('qW', 'qO', 'qG', 'cqW', 'cqO', 'cqG')


def readProfileTest(fn):
    """Return ``[(well_name, table), ...]``."""
    out = []
    for sheet in read_sheets(fn):
        table = solve_key_similarities(
            sheet, POSSIBLE_KEYS, text_columns=('name', 'date', 'depth'))
        if 'name' not in table:
            continue
        if _np.asarray(table['name']).size == 0:
            continue
        table['date'] = parse_dates(table['date'], forward_fill=True)
        fill_missing_with(table, RATE_COLUMNS, 0.0)
        table['top'], table['bottom'] = split_depth_interval(table['depth'])
        out.extend(group_by_well(table))
    return out
