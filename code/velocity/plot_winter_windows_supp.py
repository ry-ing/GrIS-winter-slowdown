# Example of the manually picked winter windows for one ROI

import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

PREP_DIR = os.path.join(config.DATA_DIR, 'velocity')
WINDOWS_PATH = os.path.join(PREP_DIR, 'winter_windows.json')
EVENT_DATE = pd.Timestamp('2021-08-14')
EXAMPLE_ROI = 3

WINDOW_COLOR = '#BBBBBB'
FS_LABEL = 28
FS_TICK = 26

with open(WINDOWS_PATH) as f:
    windows = json.load(f)[str(EXAMPLE_ROI)]

df = pd.read_csv(os.path.join(PREP_DIR, f'ROI_{EXAMPLE_ROI}_timeseries.csv'), parse_dates=['date'])
df = df.sort_values('date')

fig, ax = plt.subplots(figsize=(24, 6))

# winter windows
for start, end in windows:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=WINDOW_COLOR, alpha=0.6, linewidth=0, zorder=1)

ax.errorbar(df['date'], df['velocity'], yerr=df['velocity_error'], fmt='o', color='#444444',
            markersize=3, alpha=0.70, elinewidth=0.5, zorder=2)
ax.plot(df['date'], df['velocity_smooth'], color='dodgerblue', linewidth=2.5, zorder=3)
ax.axvline(EVENT_DATE, color='r', alpha=0.9, linestyle='--', linewidth=1.8, zorder=4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=FS_TICK)
# minor ticks every month
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.tick_params(which='major', length=10)
ax.tick_params(which='minor', length=5)
ax.set_xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2025-10-01'))
ax.set_ylim(40, 320)
ax.set_xlabel('Year', fontsize=FS_LABEL)
ax.set_ylabel('Ice speed (m yr$^{-1}$)', fontsize=FS_LABEL)
ax.set_title(f'L{EXAMPLE_ROI}', loc='left', fontsize=25, fontweight='bold', y=0.86, x=0.005)

p = os.path.join(config.FIG_DIR, 'FigS2_winter_windows.png')
fig.savefig(p, dpi=200, bbox_inches='tight')
plt.close(fig)
