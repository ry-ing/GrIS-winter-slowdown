# Watson River discharge, summer 2021

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

# read data, negatives = missing
df = pd.read_csv(config.WATSON_PATH, sep='\t')
df.columns = df.columns.str.strip()
Q_COL  = 'WaterFluxDivers&Temperature (m3/s)'
df[Q_COL] = pd.to_numeric(df[Q_COL], errors='coerce')
df.loc[df[Q_COL] < 0, Q_COL] = np.nan
df['date'] = pd.to_datetime(
    df['Year'].astype(str) + '-' +
    df['MonthOfYear'].astype(str) + '-' +
    df['DayOfMonth'].astype(str))
df = df.set_index('date').sort_index()
df['pre3_mean'] = df[Q_COL].shift(1).rolling(3).mean()
df['increase']  = df[Q_COL] - df['pre3_mean']

# percentiles of other years, Jun-Sep
MELT_S = 152   # 1 Jun
MELT_E = 274   # 1 Oct
doy_range   = np.arange(MELT_S, MELT_E + 1)
other_years = [y for y in sorted(df['Year'].unique()) if y != 2021]
mat = np.full((len(other_years), len(doy_range)), np.nan)
for i, yr in enumerate(other_years):
    sub = df[(df['Year']==yr) & df['DayOfYear'].between(MELT_S, MELT_E)][Q_COL].values
    n = min(len(sub), len(doy_range))
    mat[i, :n] = sub[:n]

q10  = np.nanpercentile(mat, 10, axis=0)
q25  = np.nanpercentile(mat, 25, axis=0)
q75  = np.nanpercentile(mat, 75, axis=0)
q90  = np.nanpercentile(mat, 90, axis=0)
qmed = np.nanpercentile(mat, 50, axis=0)
doy_dates = [pd.Timestamp('2021-01-01') + pd.Timedelta(days=int(d)-1)
             for d in doy_range]

df2021  = df[(df['Year']==2021) & df['DayOfYear'].between(MELT_S, MELT_E)].copy()
dates21 = df2021.index
q21     = df2021[Q_COL].values

# scatter, coloured by density
scatter_df  = df[df['MonthOfYear'].between(6, 9)][
    [Q_COL, 'pre3_mean', 'increase', 'Year', 'MonthOfYear']
].dropna()
scatter_pos = scatter_df[scatter_df['increase'] > 0].copy()
sx = scatter_pos[Q_COL].values
sy = scatter_pos['increase'].values

kde          = gaussian_kde(np.vstack([sx, sy]))
density      = kde(np.vstack([sx, sy]))
density_norm = (density - density.min()) / (density.max() - density.min())
order        = np.argsort(density_norm)   # dense first, sparse on top
sx_s = sx[order]
sy_s = sy[order]
dn_s = density_norm[order]

# the 3 largest increases in the record
top3 = scatter_pos.sort_values('increase', ascending=False).head(3)

# date as e.g. 14 Aug 2021
def fmt_date(idx):
    ts = pd.Timestamp(idx)
    return f"{ts.day} {ts.strftime('%b %Y')}"

r1_x = float(top3.iloc[0][Q_COL])
r1_y = float(top3.iloc[0]['increase'])
r1_l = fmt_date(top3.index[0])
r2_x = float(top3.iloc[1][Q_COL])
r2_y = float(top3.iloc[1]['increase'])
r2_l = fmt_date(top3.index[1])
r3_x = float(top3.iloc[2][Q_COL])
r3_y = float(top3.iloc[2]['increase'])
r3_l = fmt_date(top3.index[2])

# Figure
FONT = 9
FSM = 8
fig = plt.figure(figsize=(13, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                        width_ratios=[1.45, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Panel (a): timeseries
ax1.fill_between(doy_dates, q10, q90,
                 color='#BFCBDD', alpha=0.45, zorder=1,
                 label='10th–90th percentile')
ax1.fill_between(doy_dates, q25, q75,
                 color='#7B9EC8', alpha=0.60, zorder=2,
                 label='IQR (25th–75th percentile)')
ax1.plot(doy_dates, qmed,
         color='#2C5F8A', lw=1.2, ls='--', zorder=3,
         label='Median')
ax1.plot(dates21, q21,
         color='#E84855', lw=1.8, zorder=5,
         label='2021')
ax1.axvline(pd.Timestamp('2021-08-14'),
            color='black', lw=1.0, ls=':', zorder=6,
            label='14 August 2021')

ax1.set_xlim(pd.Timestamp('2021-06-01'), pd.Timestamp('2021-09-30'))
ax1.set_ylim(bottom=0)
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax1.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
ax1.set_ylabel('Discharge (m³ s⁻¹)', fontsize=FONT)
ax1.tick_params(labelsize=FSM)
ax1.grid(True, axis='y', alpha=0.18, lw=0.5)
ax1.spines[['top', 'right']].set_visible(False)
ax1.legend(fontsize=FSM, loc='upper right',
           framealpha=0.92, edgecolor='lightgray')
ax1.text(0.015, 0.97, '(a)', transform=ax1.transAxes,
         fontsize=11, fontweight='bold', va='top')

# Panel (b): scatter
sc = ax2.scatter(sx_s, sy_s,
                 c=dn_s, cmap='YlGnBu_r',
                 s=16, alpha=0.85,
                 linewidths=0.1, edgecolors='black',
                 zorder=2)

# Rank 1 - black circle
ax2.scatter(r1_x, r1_y, s=300, facecolors='none',
            edgecolors='black', linewidths=2.0, zorder=7)
ax2.annotate(r1_l, xy=(r1_x, r1_y),
             xytext=(r1_x - 1600, r1_y - 50),
             fontsize=FSM - 0.5, color='black',
             ha='left', va='center',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.1),
             bbox=dict(facecolor='white', alpha=0.90, edgecolor='none', pad=1.5))

# Rank 2 - black circle
ax2.scatter(r2_x, r2_y, s=300, facecolors='none',
            edgecolors='black', linewidths=1.0, zorder=7)
ax2.annotate(r2_l, xy=(r2_x, r2_y),
             xytext=(r2_x + 350, r2_y + 80),
             fontsize=FSM - 0.5, color='black',
             ha='left', va='bottom',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.1),
             bbox=dict(facecolor='white', alpha=0.90, edgecolor='none', pad=1.5))

# Rank 3 (Aug 2021) - red circle
ax2.scatter(r3_x, r3_y, s=300, facecolors='none',
            edgecolors='#E84855', linewidths=2.2, zorder=7)
ax2.annotate(r3_l, xy=(r3_x, r3_y),
             xytext=(r3_x + 600, r3_y + 120),
             fontsize=FSM - 0.5, color='#C0202E',
             ha='left', va='bottom',
             arrowprops=dict(arrowstyle='->', color='#C0202E', lw=1.1),
             bbox=dict(facecolor='white', alpha=0.90, edgecolor='none', pad=1.5))

ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.set_xlabel('Total daily discharge (m³ s⁻¹)', fontsize=FONT)
ax2.set_ylabel('Increase from 3-day preceding mean (m³ s⁻¹)', fontsize=FONT)
ax2.tick_params(labelsize=FSM)
ax2.grid(True, alpha=0.18, lw=0.5)
ax2.spines[['top', 'right']].set_visible(False)
cbar = fig.colorbar(sc, ax=ax2, fraction=0.040, pad=0.02)
cbar.set_label('Density', fontsize=FSM)
cbar.ax.tick_params(labelsize=FSM - 1)
ax2.text(0.015, 0.97, '(b)', transform=ax2.transAxes,
         fontsize=11, fontweight='bold', va='top')

out = os.path.join(config.FIG_DIR, 'FigS5_watson_discharge.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')