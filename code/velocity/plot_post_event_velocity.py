# ROI velocity in the weeks after the 14 August 2021 event

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

PREP_DIR = os.path.join(config.DATA_DIR, 'velocity')
EVENT_DATE = pd.Timestamp('2021-08-14')

# A4 portrait page width (~17 cm)
fig, axes = plt.subplots(5, 2, figsize=(7, 9), sharex=True)

for i in range(len(config.ROI_LABELS)):
    lbl = config.ROI_LABELS[i]
    ax = axes.flat[i]

    df = pd.read_csv(os.path.join(PREP_DIR, f'ROI_{lbl}_timeseries.csv'), parse_dates=['date'])
    df = df[(df['date'] >= '2021-07-01') & (df['date'] <= '2021-12-15')]

    ax.errorbar(df['date'], df['velocity'], yerr=df['velocity_error'], fmt='o-', color='dodgerblue', ms=3, lw=0.8, elinewidth=0.7)
    ax.axvline(EVENT_DATE, color='black', linestyle='--', linewidth=1.2)
    ax.set_title(f'L{lbl}', loc='left', fontweight='bold', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

for ax in axes[:, 0]:
    ax.set_ylabel('Ice speed (m yr$^{-1}$)', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(config.FIG_DIR, 'FigS8_post_event_velocity.png'), dpi=200)
