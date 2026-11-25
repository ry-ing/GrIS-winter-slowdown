# Winter discharge deficit at each gate, needs calculate ice discharge to be run first

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

OUTPUT_DIR = os.path.join(config.DATA_DIR, 'ice_discharge')
FIGURE_DIR = config.FIG_DIR
WINDOWS_PATH = os.path.join(config.DATA_DIR, 'velocity', 'winter_windows.json')

BASELINE_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]  # year the Nov-Apr winter ends
EVENT_DATE = pd.Timestamp('2021-08-14')
SEC_YR = 3.154e7
PLOT_XMIN = pd.Timestamp('2016-01-01')
PLOT_XMAX = pd.Timestamp('2025-06-01')

def season_year(t):
    if t.month >= 11:
        return t.year + 1
    return t.year

# error on
def quad_sigma(errs, n):
    errs = errs[np.isfinite(errs) & (errs >= 0)]
    if len(errs) == 0 or n == 0:
        return np.nan
    return float(np.sqrt(np.sum(errs**2))) / n


# read discharge csvs
all_dfs = {}
for lbl in config.ROI_LABELS:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, f'discharge_ROI_{lbl}.csv'))
    df['time'] = pd.to_datetime(df['time'])
    all_dfs[lbl] = df.sort_values('time').reset_index(drop=True)
roi_ids = sorted(all_dfs.keys())

# winter windows
with open(WINDOWS_PATH) as f:
    windows_json = json.load(f)
windows_all = {}
for k in windows_json:
    windows_all[int(k)] = []
    for s, e in windows_json[k]:
        windows_all[int(k)].append((pd.Timestamp(s), pd.Timestamp(e)))


# deficit for each ROI
summary_rows = []
season_rows = []
counted_def = {}   # deficit seasons up to recovery for the timeseries plot

for lbl in roi_ids:
    df = all_dfs[lbl]
    print('ROI', lbl)

    windows = windows_all[lbl]
    base_windows = []
    post_windows = []
    for t0, t1 in windows:
        if season_year(t1) in BASELINE_YEARS:
            base_windows.append((t0, t1))
        if season_year(t1) >= 2022:
            post_windows.append((t0, t1))

    # baseline
    base_obs = []
    base_err = []
    for t0, t1 in sorted(base_windows):
        sub = df[(df['time'] >= t0) & (df['time'] <= t1)].copy()
        base_obs.extend(sub['discharge_gt_yr'].values)
        base_err.extend(sub['u_error'].fillna(0.0).values)

    base_D = float(np.median(base_obs))
    base_mean = float(np.mean(base_obs))
    base_std = float(np.std(base_obs))
    n_base = len(base_obs)
    sig_base = quad_sigma(np.array(base_err), n_base)
    print('  baseline', round(base_D, 4), 'Gt/yr, n =', n_base)

    # post-event winters
    seasons = []
    for t0, t1 in sorted(post_windows):
        sub = df[(df['time'] >= t0) & (df['time'] <= t1)].copy()
        sy = season_year(t1)
        if sub.empty:
            continue
        d = sub['discharge_gt_yr']
        n = len(sub)
        D_med = float(d.median())
        sig_s = quad_sigma(sub['u_error'].fillna(0).values, n)
        if n >= 2:
            dur_yr = (sub['time'].max() - sub['time'].min()).total_seconds() / SEC_YR
        else:
            dur_yr = 0.5
        # deficit = below 99% of baseline
        is_def = D_med < 0.99 * base_D
        anom = 100.0 * (D_med - base_D) / base_D
        print(' ', sy, round(anom, 1), '%', is_def)

        seasons.append({'sy': sy, 't0': t0, 't1': t1, 'D_med': D_med,
                        'sigma_season': sig_s, 'dur_yr': dur_yr, 'is_deficit': is_def})
        season_rows.append({'roi_label': lbl,
                            'season_year': sy,
                            'season_median_gt_yr': D_med,
                            'baseline_median_gt_yr': base_D,
                            'anomaly_pct': anom,
                            'is_deficit': is_def,
                            'duration_yr': dur_yr,
                            'n_obs': n})

    # deficit winters until recovery
    seasons = sorted(seasons, key=lambda x: x['sy'])
    def_seasons = []
    rec_seasons = []
    for s in seasons:
        if s['is_deficit']:
            def_seasons.append(s)
        else:
            rec_seasons.append(s)
            if len(def_seasons) > 0:
                break

    if len(def_seasons) == 0:
        continue

    counted_def[lbl] = [s['sy'] for s in def_seasons]
    cumdef = 0.0
    cumdef_u = 0.0
    cum_cf = 0.0
    for s in def_seasons:
        cum_cf += base_D * s['dur_yr']
        rate = max(base_D - s['D_med'], 0.0)
        sig_rate = float(np.sqrt(sig_base**2 + s['sigma_season']**2))
        def_gt = rate * s['dur_yr']
        sig_gt = sig_rate * s['dur_yr']
        cumdef += def_gt
        cumdef_u += sig_gt
    print('  deficit', round(cumdef, 4), '+/-', round(cumdef_u, 4), 'Gt')

    # deficit per observation
    subs = []
    for s in def_seasons:
        sub = df[(df['time'] >= s['t0']) & (df['time'] <= s['t1'])].copy()
        sub['baseline_D'] = base_D
        sub['deficit_rate'] = np.maximum(base_D - sub['discharge_gt_yr'], 0.0)
        sub['season_year'] = s['sy']
        subs.append(sub)
    pd.concat(subs).to_csv(os.path.join(OUTPUT_DIR, f'discharge_deficit_ROI_{lbl}.csv'), index=False)

    deficit_pct = 100.0 * cumdef / cum_cf
    deficit_pct_u = 100.0 * cumdef_u / cum_cf

    summary_rows.append({
        'roi_label': lbl,
        'baseline_median': base_D,
        'baseline_mean': base_mean,
        'baseline_std': base_std,
        'sigma_baseline': sig_base,
        'n_baseline_obs': n_base,
        'n_deficit_win': len(def_seasons),
        'n_recovered_win': len(rec_seasons),
        'first_deficit_season': def_seasons[0]['sy'],
        'last_deficit_season': def_seasons[-1]['sy'],
        'cumulative_deficit_gt': cumdef,
        'cumulative_deficit_u': cumdef_u,
        'counterfactual_gt': cum_cf,
        'deficit_pct': deficit_pct,
        'deficit_pct_u': deficit_pct_u,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'discharge_deficit_summary.csv'), index=False)
season_df = pd.DataFrame(season_rows)
season_df.to_csv(os.path.join(OUTPUT_DIR, 'discharge_season_anomalies.csv'), index=False)

total = summary_df['cumulative_deficit_gt'].sum()
total_u = summary_df['cumulative_deficit_u'].sum()
total_cf = summary_df['counterfactual_gt'].sum()
print('total deficit', round(total, 3), '+/-', round(total_u, 3), 'Gt,', round(100 * total / total_cf, 1), '% of baseline')


# heatmap and bars
winters = []
for roi in roi_ids:
    for t0, t1 in windows_all.get(roi, []):
        w = f'{season_year(t1)-1}/{season_year(t1)}'
        if t1 > EVENT_DATE and w not in winters:
            winters.append(w)
winters = sorted(winters, key=lambda s: int(s.split('/')[0]))
labels = [f'L{r}' for r in roi_ids]

heat = np.full((len(roi_ids), len(winters)), np.nan)
for i, r in season_df.iterrows():
    w = f"{int(r['season_year'])-1}/{int(r['season_year'])}"
    if int(r['roi_label']) in roi_ids and w in winters:
        heat[roi_ids.index(int(r['roi_label'])), winters.index(w)] = r['anomaly_pct']

fig = plt.figure(figsize=(8.3, 4.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.22)
ax_hm = fig.add_subplot(gs[0, 0])
ax_bar = fig.add_subplot(gs[0, 1])

vmax = max(float(np.nanpercentile(np.abs(heat), 95)), 10.0)
im = ax_hm.imshow(heat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
ax_hm.set_yticks(np.arange(len(roi_ids)))
ax_hm.set_yticklabels(labels, fontsize=8)
ax_hm.set_xticks(np.arange(len(winters)))
ax_hm.set_xticklabels(winters, rotation=0, ha='center', fontsize=8)
ax_hm.set_xlabel('Winter season', fontsize=9)
for i in range(len(roi_ids)):
    for j in range(len(winters)):
        if np.isfinite(heat[i, j]):
            ax_hm.text(j, i, f'{heat[i, j]:+.1f}', fontsize=6.6, fontweight='bold', va='top', ha='center',
                       path_effects=[pe.withStroke(linewidth=2, foreground='white')])

pos = ax_hm.get_position()
cax = fig.add_axes([pos.x0, pos.y1 + 0.010, pos.width, 0.035])
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Winter discharge anomaly (%)', fontsize=8)
cbar.ax.tick_params(labelsize=8)
cax.xaxis.set_label_position('top')
cax.xaxis.tick_top()
ax_hm.text(-0.11, 1.03, '(a)', transform=ax_hm.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')

# discharge anomaly % (negative = deficit)
s = summary_df.set_index('roi_label').reindex(roi_ids)
defs_gt = s['cumulative_deficit_gt'].to_numpy(dtype=float)
cf = s['counterfactual_gt'].to_numpy(dtype=float)
bars = np.zeros(len(roi_ids))
for i in range(len(roi_ids)):
    bars[i] = -100.0 * defs_gt[i] / cf[i]

y = np.arange(len(roi_ids))
ax_bar.barh(y, bars, color='dodgerblue', alpha=0.85, edgecolor='black', linewidth=0.6)
ax_bar.axvline(0, color='black', lw=0.8)
ax_bar.set_yticks(y)
ax_bar.set_yticklabels(labels, fontsize=8)
ax_bar.set_xlabel('Discharge anomaly (%)', fontsize=9)
ax_bar.grid(True, axis='x', alpha=0.25)
ax_bar.invert_yaxis()
ax_bar.text(0.01, 1.04, '(b)', transform=ax_bar.transAxes, ha='left', va='bottom', fontsize=10, fontweight='bold')
ax_bar.set_xlim(-max(20.0, np.ceil((np.nanmax(-bars) + 5.0) / 5.0) * 5.0), 0.0)

for i in range(len(roi_ids)):
    roi = roi_ids[i]
    n_win = int(s.loc[roi, 'n_deficit_win'])
    txt = f'{-defs_gt[i]:.3f} Gt'
    ax_bar.text(0.02, i, f'n={n_win}\n{txt}', fontsize=7, ha='left', va='center',
                transform=ax_bar.get_yaxis_transform(),
                bbox=dict(facecolor='white', edgecolor='0.5', alpha=0.9, boxstyle='round,pad=0.15'),
                clip_on=False, zorder=7)

p = os.path.join(FIGURE_DIR, 'Fig06_discharge_heatmap.png')
fig.savefig(p, dpi=300, bbox_inches='tight')
plt.close(fig)


# timeseries plot
base_map = {}
for i, r in summary_df.iterrows():
    base_map[int(r['roi_label'])] = float(r['baseline_median'])
is_def = {}
for i, r in season_df.iterrows():
    is_def[(int(r['roi_label']), int(r['season_year']))] = bool(r['is_deficit'])

n = len(roi_ids)
fig, axes = plt.subplots(n, 1, figsize=(9.0, n * 1.2), sharex=True, gridspec_kw={'hspace': 0.04})

for i in range(n):
    ax = axes[i]
    roi = roi_ids[i]
    df = all_dfs[roi]
    ax.text(0.013, 0.84, f'L{roi}', transform=ax.transAxes, fontsize=8, fontweight='bold', va='top')
    ax.set_xlim(PLOT_XMIN, PLOT_XMAX)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18, lw=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='both'))

    # weekly means, 3 week rolling median for the line
    d = df.set_index('time').resample('W').agg({'discharge_gt_yr': 'mean', 'u_error': 'mean', 'l_error': 'mean'})
    d = d.dropna(subset=['discharge_gt_yr']).reset_index()
    d['smooth'] = d['discharge_gt_yr'].rolling(3, center=True, min_periods=1).median()

    for t0, t1 in windows_all.get(roi, []):
        sy = season_year(t1)
        if sy in BASELINE_YEARS:
            ax.axvspan(t0, t1, color='#2196F3', alpha=0.18, linewidth=0, zorder=1)
        elif t1 > EVENT_DATE and (roi, sy) in is_def:
            # deficit until the first recovered winter, recovered from then on
            if sy in counted_def.get(roi, []):
                colour = '#F44336'
            else:
                colour = '#4CAF50'
            ax.axvspan(t0, t1, color=colour, alpha=0.20, linewidth=0, zorder=1)

    ax.fill_between(d['time'], d['discharge_gt_yr'] - d['l_error'].fillna(0),
                    d['discharge_gt_yr'] + d['u_error'].fillna(0),
                    color='#4C9BE8', alpha=0.18, linewidth=0, zorder=2)
    ax.plot(d['time'], d['smooth'], color='#1a6fbf', lw=0.9, zorder=3)
    if roi in base_map:
        ax.axhline(base_map[roi], color='black', lw=0.7, ls='--', alpha=0.7, zorder=4)
    ax.axvline(EVENT_DATE, color='#c0392b', lw=0.9, ls='--', alpha=0.85, zorder=4)

axes[-1].set_xlabel('Date', fontsize=9)
fig.text(0.06, 0.5, 'Ice discharge (Gt yr$^{-1}$)', rotation='vertical', va='center', fontsize=11)
fig.tight_layout(rect=[0.07, 0.02, 1.0, 1.0])
legend_patches = [Patch(facecolor='#2196F3', alpha=0.5, label='Baseline winter'),
                  Patch(facecolor='#F44336', alpha=0.5, label='Deficit season'),
                  Patch(facecolor='#4CAF50', alpha=0.5, label='Recovered season')]
axes[0].legend(handles=legend_patches, fontsize=7, loc='upper right', framealpha=0.85, ncols=3,
               bbox_to_anchor=(1.0, 1.3))

p = os.path.join(FIGURE_DIR, 'FigS7_ice-discharge_tseries.png')
fig.savefig(p, dpi=250, bbox_inches='tight')
plt.close(fig)
print('done')
