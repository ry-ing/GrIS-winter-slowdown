# Summer runoff vs winter velocity
# needs the MAR daily runoff csv from fig_return_period_runoff_increase.py

import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

warnings.filterwarnings("ignore")

PREP_DIR = os.path.join(config.DATA_DIR, 'velocity')
OUTPUT_DIR = config.FIG_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAR_DAILY_CSV = os.path.join(config.DATA_DIR, 'met', 'mar_hist_runoff_daily_1985_2025_rp.csv')

# Mankoff et al. (2020) MAR runoff for each basin
MANKOFF_BASINS = os.path.join(config.MANKOFF_DIR, 'basins.gpkg')
MANKOFF_DISCHARGE_DIR = os.path.join(config.MANKOFF_DIR, 'discharge')
CATCHMENT_RADIUS_M = 10_000
RHO_WATER = 1000.0   # discharge is m3/s of water

SELECTED_ROI_LABELS = config.ROI_LABELS
YEARS = list(range(2016, 2025))
BASELINE_YEARS = [2016, 2017, 2018, 2019, 2020]
EVENT_YEAR = 2021

POST_WINTER_COLORS = {
    2021: '#0C2C84',
    2022: '#1D91C0',
    2023: '#41B6C4',
    2024: '#C7E9B4',
}
POST_WINTER_LABELS = {
    2021: 'W1 (2021/22)',
    2022: 'W2 (2022/23)',
    2023: 'W3 (2023/24)',
    2024: 'W4 (2024/25)',
}

EXTREME_EVENT_PCTL = 90   # 3-day runoff increase above the 90th percentile = extreme melt day

PRE_COLOR = '#AAAAAA'
FONT = 28
FONT_SM = 24
FONT_PL = 34


# Mankoff basins within 10 km of each ROI
rois = gpd.read_file(config.ROI_SHAPEFILE_PATH).to_crs('EPSG:3413')
rois = rois[rois['id'].astype(int).isin(SELECTED_ROI_LABELS)]
basins = gpd.read_file(MANKOFF_BASINS).to_crs('EPSG:3413')

roi_basins = {}
for i, row in rois.iterrows():
    lbl = int(row['id'])
    centroid = row.geometry.centroid
    hits = basins[basins.centroid.within(centroid.buffer(CATCHMENT_RADIUS_M))]
    roi_basins[lbl] = list(hits['cat'].values)

# May-Sep runoff for each ROI
runoff = {}
for lbl in SELECTED_ROI_LABELS:
    runoff[lbl] = {}
for yr in YEARS:
    da = xr.open_dataset(os.path.join(MANKOFF_DISCHARGE_DIR, f'MAR_{yr}.nc'))['discharge']
    da = da.sel(time=slice(f'{yr}-05-01', f'{yr}-09-30'))
    for lbl in SELECTED_ROI_LABELS:
        cats = [b for b in roi_basins[lbl] if b in da['station'].values]
        daily_gt = da.sel(station=cats).sum(dim='station').values.astype(float) * RHO_WATER * 86400.0 / 1e12
        if np.sum(daily_gt > 0) < 5:
            runoff[lbl][yr] = np.nan
        else:
            runoff[lbl][yr] = float(np.sum(daily_gt))


# winter and autumn velocity anomalies
winter_pct = {}
autumn_pct = {}
for lbl in SELECTED_ROI_LABELS:
    df = pd.read_csv(os.path.join(PREP_DIR, f'ROI_{lbl}_timeseries.csv'), parse_dates=['date'])
    winter_med = {}
    autumn_min = {}
    for yr in YEARS:
        v = df.loc[(df['date'] >= f'{yr}-11-01') & (df['date'] <= f'{yr + 1}-04-30'), 'velocity'].dropna()
        if len(v) >= 3:
            winter_med[yr] = float(v.median())
        else:
            winter_med[yr] = np.nan
        a = df.loc[(df['date'] >= f'{yr}-09-01') & (df['date'] <= f'{yr}-10-31'), 'velocity_smooth'].dropna()
        if len(a) >= 3:
            autumn_min[yr] = float(a.min())
        else:
            autumn_min[yr] = np.nan

    base = float(np.nanmedian([winter_med[yr] for yr in BASELINE_YEARS]))
    winter_pct[lbl] = {}
    autumn_pct[lbl] = {}
    for yr in YEARS:
        winter_pct[lbl][yr] = (winter_med[yr] / base - 1.0) * 100.0
        autumn_pct[lbl][yr] = (autumn_min[yr] / base - 1.0) * 100.0


# standardise each ROI
norm = {}            # lbl -> (x mean, x std, y mean, y std)
x_pre_norm = []
y_pre_norm = []
scatter_all = []     # (x norm, y norm, year, roi) for every year

for lbl in SELECTED_ROI_LABELS:
    x_roi = []
    y_roi = []
    for yr in BASELINE_YEARS:
        if np.isfinite(runoff[lbl][yr]) and np.isfinite(winter_pct[lbl][yr]):
            x_roi.append(runoff[lbl][yr])
            y_roi.append(winter_pct[lbl][yr])

    x_roi = np.array(x_roi)
    y_roi = np.array(y_roi)
    x_mu = x_roi.mean()
    x_sd = x_roi.std(ddof=1)
    y_mu = y_roi.mean()
    y_sd = y_roi.std(ddof=1)
    norm[lbl] = (x_mu, x_sd, y_mu, y_sd)

    x_pre_norm.extend((x_roi - x_mu) / x_sd)
    y_pre_norm.extend((y_roi - y_mu) / y_sd)

    for yr in YEARS:
        if np.isfinite(runoff[lbl][yr]) and np.isfinite(winter_pct[lbl][yr]):
            xn = (runoff[lbl][yr] - x_mu) / x_sd
            yn = (winter_pct[lbl][yr] - y_mu) / y_sd
            scatter_all.append((xn, yn, yr, lbl))

x_pre_norm = np.array(x_pre_norm)
y_pre_norm = np.array(y_pre_norm)

# pooled OLS fit
reg = stats.linregress(x_pre_norm, y_pre_norm)
slope = reg.slope
intercept = reg.intercept
r_val = reg.rvalue
p_val = reg.pvalue
print('runoff vs velocity R2', round(r_val**2, 3), 'p', round(p_val, 4), 'n', len(x_pre_norm))

# 95% confidence interval on the fitted line
n = len(x_pre_norm)
x_bar = x_pre_norm.mean()
Sxx = np.sum((x_pre_norm - x_bar) ** 2)
MSE = np.sum((y_pre_norm - (slope * x_pre_norm + intercept)) ** 2) / (n - 2)
s = np.sqrt(MSE)

all_xn = np.array([p[0] for p in scatter_all])
x_line = np.linspace(all_xn.min() * 1.15, all_xn.max() * 1.15, 300)
y_line = slope * x_line + intercept
t_crit = stats.t.ppf(0.975, df=n - 2)
se_fit = s * np.sqrt(1.0 / n + (x_line - x_bar) ** 2 / Sxx)
ci_hi = y_line + t_crit * se_fit
ci_lo = y_line - t_crit * se_fit


# residuals (%)
res_mean = {}
res_se = {}
for yr in YEARS:
    res = []
    for lbl in norm:
        if np.isfinite(runoff[lbl][yr]) and np.isfinite(winter_pct[lbl][yr]):
            x_mu, x_sd, y_mu, y_sd = norm[lbl]
            xn = (runoff[lbl][yr] - x_mu) / x_sd
            y_pred = (slope * xn + intercept) * y_sd + y_mu
            res.append(winter_pct[lbl][yr] - y_pred)
    res_mean[yr] = float(np.mean(res))
    res_se[yr] = float(np.std(res, ddof=1) / np.sqrt(len(res)))

print('residuals (%)')
for yr in YEARS:
    print(yr, round(res_mean[yr], 1))


# autumn minimum, mean over the ROIs
aut_mean = []
aut_se = []
for yr in YEARS:
    vals = [autumn_pct[lbl][yr] for lbl in SELECTED_ROI_LABELS if np.isfinite(autumn_pct[lbl][yr])]
    aut_mean.append(float(np.mean(vals)))
    aut_se.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
aut_mean = np.array(aut_mean)
aut_se = np.array(aut_se)


# (d) runoff and extreme melt days
domain_runoff_total = []
for yr in YEARS:
    domain_runoff_total.append(float(np.sum([runoff[lbl][yr] for lbl in SELECTED_ROI_LABELS])))
domain_runoff_total = np.array(domain_runoff_total)

mar_daily = pd.read_csv(MAR_DAILY_CSV, parse_dates=['date'])
pos_delta = mar_daily.loc[mar_daily['delta_gt'] > 0, 'delta_gt']
extreme_thresh = float(np.percentile(pos_delta, EXTREME_EVENT_PCTL))
mar_daily['is_extreme'] = mar_daily['delta_gt'] > extreme_thresh
n_extreme_by_year = mar_daily.groupby('year')['is_extreme'].sum()
n_extreme = np.array([float(n_extreme_by_year[yr]) for yr in YEARS])
print('extreme melt threshold', round(extreme_thresh, 2), 'Gt/day')


# figure
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': FONT,
    'axes.labelsize': FONT,
    'xtick.labelsize': FONT,
    'ytick.labelsize': FONT,
    'legend.fontsize': FONT_SM,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

fig = plt.figure(figsize=(18, 18), facecolor='white')
gs = gridspec.GridSpec(4, 1, figure=fig, left=0.16, right=0.97, top=0.97, bottom=0.05,
                       height_ratios=[1.5, 1, 1, 1], hspace=0.63)
ax_scat = fig.add_subplot(gs[0])
ax_res = fig.add_subplot(gs[1])
ax_aut = fig.add_subplot(gs[2])
ax_d = fig.add_subplot(gs[3])


# (a) scatter in normalised units
ax_scat.spines[['top', 'right']].set_visible(False)
ax_scat.grid(True, axis='y', alpha=0.20, linewidth=0.5, color='#999999')
ax_scat.tick_params(which='both', direction='out')
ax_scat.grid(True, axis='x', alpha=0.20, linewidth=0.5, color='#999999')

ax_scat.fill_between(x_line, ci_lo, ci_hi, alpha=0.18, color='#555555', zorder=1)
ax_scat.plot(x_line, y_line, color='black', linewidth=1.8, zorder=4, label='OLS fit (pre-event)')

# pre-event points
ax_scat.scatter(x_pre_norm, y_pre_norm, s=90, color='grey', alpha=0.6,
                linewidths=0.5, edgecolors='#444444', zorder=3)

# post-event points for each ROI
for yr in POST_WINTER_COLORS:
    xs = [p[0] for p in scatter_all if p[2] == yr]
    ys = [p[1] for p in scatter_all if p[2] == yr]
    ax_scat.scatter(xs, ys, color=POST_WINTER_COLORS[yr], s=80, marker='D',
                    edgecolors='black', linewidths=0.6, alpha=0.80, zorder=5)

# ROI means
post_legend = []
for yr in POST_WINTER_COLORS:
    xs = [p[0] for p in scatter_all if p[2] == yr]
    ys = [p[1] for p in scatter_all if p[2] == yr]
    ax_scat.scatter(float(np.mean(xs)), float(np.mean(ys)), color=POST_WINTER_COLORS[yr], s=280, marker='D',
                    edgecolors='black', linewidths=1.2, zorder=7)
    post_legend.append(Line2D([0], [0], marker='D', color='w', markerfacecolor=POST_WINTER_COLORS[yr],
                              markeredgecolor='black', markersize=12, label=POST_WINTER_LABELS[yr]))

ax_scat.axhline(0, color='#888888', linewidth=0.8, linestyle=':', zorder=1)
ax_scat.axvline(0, color='#888888', linewidth=0.8, linestyle=':', zorder=1)

ax_scat.text(0.97, 0.04, f'$R^{{2}}$ = {r_val**2:.2f},   $p$ = {p_val:.4f}',
             transform=ax_scat.transAxes, fontsize=FONT_SM, ha='right', va='bottom', style='italic',
             bbox=dict(facecolor='white', alpha=0.80, edgecolor='none', pad=2))

ax_scat.set_xlabel('Summer runoff anomaly (normalised)', fontsize=FONT)
ax_scat.set_ylabel('Winter velocity\nanomaly (normalised)', fontsize=FONT)

pre_h = Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markeredgecolor='#444444',
               markersize=10, label='Pre-event (per ROI, normalised)')
fit_h = Line2D([0], [0], color='black', linewidth=1.8, label='OLS fit (pre-event)')
ci_h = mpatches.Patch(facecolor='#555555', alpha=0.20, edgecolor='none', label='95% CI')

ax_scat.set_xlim(x_line.min(), x_line.max())
all_yn = np.concatenate([y_pre_norm, np.array([p[1] for p in scatter_all])])
y_pad = (all_yn.max() - all_yn.min()) * 0.02
ax_scat.set_ylim(all_yn.min() - y_pad, all_yn.max() + y_pad)

ax_scat.legend(handles=[pre_h, fit_h, ci_h] + post_legend,
               fontsize=FONT_SM - 2, framealpha=0.95, edgecolor='lightgray',
               loc='lower left', bbox_to_anchor=(0.0, 1.01),
               ncols=4, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
ax_scat.text(0.02, 0.97, '(a)', transform=ax_scat.transAxes, fontsize=FONT_PL, fontweight='bold', va='top')


# (b) residuals in %
ax_res.spines[['top', 'right']].set_visible(False)
ax_res.grid(True, axis='y', alpha=0.20, linewidth=0.5, color='#999999')
ax_res.tick_params(which='both', direction='out')

res_means = np.array([res_mean[yr] for yr in YEARS])
res_ses = np.array([res_se[yr] for yr in YEARS])
res_labels = [f"{yr}/{str(yr + 1)[2:]}" for yr in YEARS]
bar_cols = []
for yr in YEARS:
    if yr > 2020:
        bar_cols.append(POST_WINTER_COLORS[yr])
    else:
        bar_cols.append(PRE_COLOR)

x_pos = np.arange(len(YEARS))
ax_res.bar(x_pos, res_means, color=bar_cols, edgecolor='k', linewidth=0.8, width=0.70, zorder=2)
ax_res.errorbar(x_pos, res_means, yerr=res_ses * 1.96, fmt='none', color='#222222', linewidth=1.4,
                capsize=5, capthick=1.4, zorder=4)
ax_res.axhline(0, color='black', linewidth=1.0, zorder=3)

pre_n = len(BASELINE_YEARS)
ax_res.axvline(pre_n - 0.5, color='#D62728', linewidth=1.4, linestyle='--', alpha=0.75, zorder=3)

ax_res.set_xticks(x_pos)
ax_res.set_xticklabels(res_labels, rotation=35, ha='right', fontsize=FONT)
ax_res.set_ylabel('Velocity residual (%)', fontsize=FONT)
ax_res.set_xlabel('Winter season', fontsize=FONT)

ylims = ax_res.get_ylim()
ax_res.text(pre_n - 0.4, ylims[1] * 0.88, 'August 2021', fontsize=FONT_SM, color='#D62728',
            style='italic', va='top', ha='left')
ax_res.text(0.02, 0.97, '(b)', transform=ax_res.transAxes, fontsize=FONT_PL, fontweight='bold', va='top')


# (c) autumn velocity minimum
ax_aut.spines[['top', 'right']].set_visible(False)
ax_aut.grid(True, axis='y', alpha=0.20, linewidth=0.5, color='#999999')
ax_aut.tick_params(which='both', direction='out')

aut_years = np.array(YEARS)
aut_cols = []
for yr in YEARS:
    if yr < EVENT_YEAR:
        aut_cols.append(PRE_COLOR)
    else:
        aut_cols.append(POST_WINTER_COLORS[yr])

ax_aut.bar(aut_years, aut_mean, color=aut_cols, edgecolor='k', linewidth=0.8, width=0.7, zorder=2)
ax_aut.errorbar(aut_years, aut_mean, yerr=aut_se * 1.96, fmt='none', color='#222222', linewidth=1.4,
                capsize=5, capthick=1.4, zorder=4)
ax_aut.axhline(0, color='black', linewidth=1.0, zorder=3)
ax_aut.axvline(EVENT_YEAR - 0.5, color='#D62728', linewidth=1.4, linestyle='--', alpha=0.75, zorder=5)

# pre-event mean +/- 1 std
pre_aut = aut_mean[aut_years < EVENT_YEAR]
ax_aut.axhspan(pre_aut.mean() - pre_aut.std(ddof=1), pre_aut.mean() + pre_aut.std(ddof=1),
               alpha=0.10, color='black', zorder=1)
ax_aut.axhline(pre_aut.mean(), color='black', linewidth=1.2, linestyle='--', zorder=2)

ax_aut.set_xticks(aut_years)
ax_aut.set_xticklabels([str(yr) for yr in YEARS], rotation=45, ha='right')
ax_aut.set_xlabel('Year', fontsize=FONT)
ax_aut.set_ylabel('Autumn velocity\nmin. anomaly (%)', fontsize=FONT)

ylims_c = ax_aut.get_ylim()
ax_aut.text(EVENT_YEAR - 0.4, ylims_c[1] * 0.88, 'August 2021', fontsize=FONT_SM, color='#D62728',
            style='italic', va='top', ha='left')
ax_aut.text(0.02, 0.97, '(c)', transform=ax_aut.transAxes, fontsize=FONT_PL, fontweight='bold', va='top')


# (d) runoff and extreme melt days
ax_d.spines[['top', 'right']].set_visible(False)
ax_d.tick_params(which='both', direction='out')

stat_years = np.array(YEARS, dtype=float)
width = 0.38

b1 = ax_d.bar(stat_years - width / 2, domain_runoff_total, width=width,
              color=aut_cols, edgecolor='k', linewidth=0.8, zorder=3)
ax_d.set_ylabel('Runoff total (Gt)', fontsize=FONT)
ax_d.set_ylim(bottom=0)

ax_d2 = ax_d.twinx()
ax_d2.spines[['top']].set_visible(False)
b2 = ax_d2.bar(stat_years + width / 2, n_extreme, width=width, color=aut_cols, alpha=0.45, hatch='///',
               edgecolor='k', linewidth=0.8, zorder=3)
ax_d2.set_ylabel('Extreme melt\nevents (count)', fontsize=FONT)
ax_d2.set_ylim(bottom=0)
ax_d2.grid(False)

ax_d.axvline(EVENT_YEAR - 0.5, color='#D62728', linewidth=1.4, linestyle='--', alpha=0.75, zorder=5)
ax_d.text(EVENT_YEAR - 0.4, 1.03, 'August 2021', transform=ax_d.get_xaxis_transform(),
          fontsize=FONT_SM, color='#D62728', style='italic', va='bottom', ha='left', clip_on=False)

ax_d.set_xticks(stat_years)
ax_d.set_xticklabels([str(yr) for yr in YEARS], rotation=45, ha='right')
ax_d.set_xlabel('Year', fontsize=FONT)

ax_d.legend([b1, b2], ['Runoff total', 'Extreme melt events'],
            fontsize=FONT_SM - 3, framealpha=0.7, edgecolor='lightgray',
            loc='upper right', handlelength=1.4, bbox_to_anchor=(1, 1.15))
ax_d.text(0.02, 0.97, '(d)', transform=ax_d.transAxes, fontsize=FONT_PL, fontweight='bold', va='top',
          bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2))

out_path = os.path.join(OUTPUT_DIR, 'Fig07_melt_vel_relationship.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')


# second figure, velocity vs number of extreme melt days (not in paper)
x_pre_ext = []
y_pre_ext = []
scatter_ext = []   # (n extreme, y norm, year)

for lbl in norm:
    x_mu, x_sd, y_mu, y_sd = norm[lbl]
    for yr in YEARS:
        if not np.isfinite(winter_pct[lbl][yr]):
            continue
        xv = float(n_extreme_by_year[yr])
        yn = (winter_pct[lbl][yr] - y_mu) / y_sd
        scatter_ext.append((xv, yn, yr))
        if yr in BASELINE_YEARS:
            x_pre_ext.append(xv)
            y_pre_ext.append(yn)

x_pre_ext = np.array(x_pre_ext)
y_pre_ext = np.array(y_pre_ext)

reg_e = stats.linregress(x_pre_ext, y_pre_ext)
print('extreme days vs velocity R2', round(reg_e.rvalue**2, 3), 'p', round(reg_e.pvalue, 4))

n_e = len(x_pre_ext)
x_bar_e = x_pre_ext.mean()
Sxx_e = np.sum((x_pre_ext - x_bar_e) ** 2)
MSE_e = np.sum((y_pre_ext - (reg_e.slope * x_pre_ext + reg_e.intercept)) ** 2) / (n_e - 2)
s_e = np.sqrt(MSE_e)

all_x_ext = np.array([p[0] for p in scatter_ext])
x_line_e = np.linspace(all_x_ext.min() - 1, all_x_ext.max() + 1, 300)
y_line_e = reg_e.slope * x_line_e + reg_e.intercept
t_crit_e = stats.t.ppf(0.975, df=n_e - 2)
se_fit_e = s_e * np.sqrt(1.0 / n_e + (x_line_e - x_bar_e) ** 2 / Sxx_e)
ci_hi_e = y_line_e + t_crit_e * se_fit_e
ci_lo_e = y_line_e - t_crit_e * se_fit_e

fig2, ax_e = plt.subplots(figsize=(14, 11), facecolor='white')
ax_e.spines[['top', 'right']].set_visible(False)
ax_e.grid(True, axis='y', alpha=0.20, linewidth=0.5, color='#999999')
ax_e.tick_params(which='both', direction='out')
ax_e.grid(True, axis='x', alpha=0.20, linewidth=0.5, color='#999999')

ax_e.fill_between(x_line_e, ci_lo_e, ci_hi_e, alpha=0.18, color='#555555', zorder=1)
ax_e.plot(x_line_e, y_line_e, color='black', linewidth=1.8, zorder=4, label='OLS fit (pre-event)')

rng = np.random.default_rng(0)
jitter_pre = rng.uniform(-0.15, 0.15, size=len(x_pre_ext))
ax_e.scatter(x_pre_ext + jitter_pre, y_pre_ext, s=90, color='grey', alpha=0.6,
             linewidths=0.5, edgecolors='#444444', zorder=3)

post_legend_e = []
for yr in POST_WINTER_COLORS:
    xs = np.array([p[0] for p in scatter_ext if p[2] == yr])
    ys = np.array([p[1] for p in scatter_ext if p[2] == yr])
    jitter_post = rng.uniform(-0.15, 0.15, size=len(xs))
    ax_e.scatter(xs + jitter_post, ys, color=POST_WINTER_COLORS[yr], s=80, marker='D',
                 edgecolors='black', linewidths=0.6, alpha=0.80, zorder=5)
    ax_e.scatter(float(np.mean(xs)), float(np.mean(ys)), color=POST_WINTER_COLORS[yr], s=280, marker='D',
                 edgecolors='black', linewidths=1.2, zorder=7)
    post_legend_e.append(Line2D([0], [0], marker='D', color='w', markerfacecolor=POST_WINTER_COLORS[yr],
                                markeredgecolor='black', markersize=12, label=POST_WINTER_LABELS[yr]))

ax_e.axhline(0, color='#888888', linewidth=0.8, linestyle=':', zorder=1)

ax_e.text(0.97, 0.04, f'$R^{{2}}$ = {reg_e.rvalue**2:.2f},   $p$ = {reg_e.pvalue:.4f}',
          transform=ax_e.transAxes, fontsize=FONT_SM, ha='right', va='bottom', style='italic',
          bbox=dict(facecolor='white', alpha=0.80, edgecolor='none', pad=2))

ax_e.set_xlabel('Number of extreme melt events (summer, domain-wide)', fontsize=FONT)
ax_e.set_ylabel('Winter velocity\nanomaly (normalised)', fontsize=FONT)

pre_h_e = Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markeredgecolor='#444444',
                 markersize=10, label='Pre-event (per ROI, jittered)')
fit_h_e = Line2D([0], [0], color='black', linewidth=1.8, label='OLS fit (pre-event)')
ci_h_e = mpatches.Patch(facecolor='#555555', alpha=0.20, edgecolor='none', label='95% CI')

ax_e.legend(handles=[pre_h_e, fit_h_e, ci_h_e] + post_legend_e,
            fontsize=FONT_SM - 2, framealpha=0.95, edgecolor='lightgray',
            loc='lower left', bbox_to_anchor=(0.0, 1.15),
            ncols=4, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)

out_path_e = os.path.join(OUTPUT_DIR, 'extra-fig_alttest_norm_extreme_events.png')
fig2.savefig(out_path_e, dpi=300, bbox_inches='tight', facecolor='white')
