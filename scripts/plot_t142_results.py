"""Plot T142 full-run results: daily liquid rate, oil rate, water cut.

Reads results/T142_full/well_rates.csv (produced by run_t142_full.py).
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

CSV_PATH = os.path.join('results', 'T142_full', 'well_rates.csv')
PNG_PATH = os.path.join('results', 'T142_full', 'well_rates.png')


def load_data(path):
    steps, times, dates, wells = [], [], [], []
    qO, qW, qG, bhp, status = [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row['step']))
            times.append(float(row['time_days']))
            dates.append(row['date'])
            wells.append(row['well'])
            status.append(bool(int(row['status'])))
            qO.append(float(row['qO_sm3d']))
            qW.append(float(row['qW_sm3d']))
            qG.append(float(row['qG_sm3d']))
            bhp.append(float(row['bhp_bar']))
    return (np.asarray(steps), np.asarray(times), dates, wells,
            np.asarray(status), np.asarray(qO), np.asarray(qW),
            np.asarray(qG), np.asarray(bhp))


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f'CSV not found: {CSV_PATH} (run scripts/run_t142_full.py first)')

    steps, times, dates, wells, status, qO, qW, qG, bhp = load_data(CSV_PATH)
    step_list = np.unique(steps)
    print(f'loaded {len(step_list)} report steps, {len(np.unique(wells))} wells')

    # Aggregate producing wells per step (producer: qO<0 or qW<0)
    t = np.unique(times)
    qO_tot = np.zeros_like(t)
    qW_tot = np.zeros_like(t)
    qL_tot = np.zeros_like(t)
    fw = np.zeros_like(t)
    nprod = np.zeros(len(t), dtype=int)

    for i, st in enumerate(step_list):
        mask = steps == st
        prod = mask & status & ((qO < 0) | (qW < 0))
        inj = mask & status & (qW > 0) & (qO >= 0)
        qO_tot[i] = float(np.sum(-qO[prod]))
        qW_tot[i] = float(np.sum(-qW[prod]))
        nprod[i] = int(np.sum(prod))
        qL_tot[i] = qO_tot[i] + qW_tot[i]
        fw[i] = qW_tot[i] / qL_tot[i] if qL_tot[i] > 0 else np.nan

    days = t / 1.0
    years = t / 365.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Top: liquid & oil production rates
    ax1.plot(years, qL_tot, '-', color='tab:blue', lw=1.6, label='Total liquid rate')
    ax1.plot(years, qO_tot, '-', color='tab:green', lw=1.6, label='Oil rate')
    ax1.set_ylabel('Rate (Sm³/day)')
    ax1.set_title('T142 — Daily Liquid & Oil Production Rates')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(years.min(), years.max())

    # Bottom: water cut
    ax2.plot(years, fw * 100.0, '-', color='tab:red', lw=1.6)
    ax2.set_ylabel('Water cut (%)')
    ax2.set_xlabel('Time (years)')
    ax2.set_title('T142 — Water Cut (field)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    print(f'saved: {PNG_PATH}')

    # textual summary at key points
    print('\nfield totals (producers only):')
    for i in [0, len(t) // 4, len(t) // 2, 3 * len(t) // 4, len(t) - 1]:
        print(f'  t={days[i]:7.0f} d ({days[i]/365.25:5.2f} yr): '
              f'qL={qL_tot[i]:9.1f}  qO={qO_tot[i]:9.1f}  qW={qW_tot[i]:9.1f}  '
              f'fw={fw[i]*100:5.2f}%  nprod={nprod[i]}')


if __name__ == '__main__':
    main()
