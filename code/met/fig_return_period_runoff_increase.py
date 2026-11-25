# Return periods of the 3-day runoff increase, MAR historical vs MAR-CESM2

import os
import sys
import socket
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

HIST_DIR = Path(config.MAR_DIR)
FUT_DIR = Path(config.MAR_CESM2_DIR)
MASK_PATH_FILE = Path(config.MASK_PATH_FILE)
OUT_DATA_DIR = Path(config.DATA_DIR) / 'met'
OUT_FIG_DIR = Path(config.FIG_DIR)

MAR_REGION = {'minx': -340000.0, 'miny': -2750000.0, 'maxx': 50000.0, 'maxy': -2000000.0}
PIXEL_AREA_M2 = (10e3) ** 2
MELT_MONTHS = [6, 7, 8, 9]
HIST_PERIOD = (1985, 2025)
FUT_PERIOD = (2060, 2100)

SLURM_CONFIG = {
    'account': 'geos_extra',
    'cores': 8,
    'memory': '32GB',
    'queue': 'General',
    'walltime': '06:00:00',
    'log_directory': '/home/s1834371/scratch/dask_log',
    'job_extra_directives': ['--job-name=return-period-mar', f'--nodelist={socket.gethostname().split(".")[0]}'],
}


def open_ru(directory, year):
    # MAR runoff
    ds = xr.open_dataset(directory / f'MARv3.14.3-10km-daily-ERA5-{year}.nc', chunks={'TIME': -1})
    ru = ds['RU'].sum(dim='SECTOR').rename({'TIME': 'time'})
    return ru.assign_coords(x=np.round(ru.x.values * 1000).astype(np.float64),
                            y=np.round(ru.y.values * 1000).astype(np.float64))


def daily_runoff(directory, period, mask):
    # daily runoff and 3-day change
    dfs = []
    for year in range(period[0], period[1] + 1):
        ru = open_ru(directory, year).where(mask == 1)
        ru = ru.sel(x=ru.x[(ru.x >= MAR_REGION['minx']) & (ru.x <= MAR_REGION['maxx'])],
                    y=ru.y[(ru.y >= MAR_REGION['miny']) & (ru.y <= MAR_REGION['maxy'])])
        ru = ru.sel(time=ru.time.dt.month.isin(MELT_MONTHS))
        total = (ru.sum(dim=['x', 'y']) * PIXEL_AREA_M2 * 1e-3 * 1000 / 1e12).load()

        vals = total.values.astype(float)
        dates = pd.DatetimeIndex(total.time.values).normalize()
        df = pd.DataFrame()
        df['year'] = [year] * len(vals)
        df['date'] = [str(t.date()) for t in dates]
        df['runoff_gt'] = vals
        df['delta_gt'] = vals - pd.Series(vals).shift(1).rolling(3).mean().values
        dfs.append(df)
        print(year)
    return pd.concat(dfs, ignore_index=True)


def bootstrap_gev(data, return_periods, n_boot=1000):
    # bootstrap 95% range
    rng = np.random.default_rng(42)
    levels = np.zeros((n_boot, len(return_periods)))
    for i in range(n_boot):
        c, loc, scale = gev.fit(rng.choice(data, size=len(data), replace=True))
        levels[i] = gev.ppf(1 - 1 / return_periods, c, loc, scale)
    return np.percentile(levels, 2.5, axis=0), np.percentile(levels, 97.5, axis=0)


OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
hist_csv = OUT_DATA_DIR / f'mar_hist_runoff_daily_{HIST_PERIOD[0]}_{HIST_PERIOD[1]}_rp.csv'
fut_csv = OUT_DATA_DIR / f'mar_proj_runoff_daily_{FUT_PERIOD[0]}_{FUT_PERIOD[1]}_rp.csv'

# daily runoff, only remade if the csvs aren't there
if hist_csv.exists() and fut_csv.exists():
    hist_df = pd.read_csv(hist_csv)
    fut_df = pd.read_csv(fut_csv)
else:
    cluster = SLURMCluster(**SLURM_CONFIG)
    cluster.scale(jobs=4)
    client = Client(cluster)

    # ice mask (==2) onto the MAR grid
    ref = open_ru(HIST_DIR, 2021).isel(time=0).load()
    mask_ds = xr.open_dataset(MASK_PATH_FILE).rio.write_crs('EPSG:3413').rio.clip_box(**MAR_REGION)
    mask = xr.where(mask_ds['Band1'] == 2, 1, 0).interp_like(ref, method='nearest').fillna(0)

    hist_df = daily_runoff(HIST_DIR, HIST_PERIOD, mask)
    hist_df.to_csv(hist_csv, index=False)
    fut_df = daily_runoff(FUT_DIR, FUT_PERIOD, mask)
    fut_df.to_csv(fut_csv, index=False)
    client.close()
    cluster.close()

# annual maxima of the positive increases
hist_ann = hist_df[hist_df['delta_gt'] > 0].groupby('year')['delta_gt'].max().dropna().sort_index()
fut_ann = fut_df[fut_df['delta_gt'] > 0].groupby('year')['delta_gt'].max().dropna().sort_index()
hist_max = hist_ann.values.astype(float)
hist_years = hist_ann.index.to_numpy(dtype=int)
fut_max = fut_ann.values.astype(float)
fut_years = fut_ann.index.to_numpy(dtype=int)
print('largest increase', round(hist_max.max(), 3), 'Gt/d in', hist_years[np.argmax(hist_max)])

# GEV fits
return_periods = np.logspace(0, 2, 100)
c_h, loc_h, scale_h = gev.fit(hist_max)
c_f, loc_f, scale_f = gev.fit(fut_max)
print('GEV hist', c_h, loc_h, scale_h)
print('GEV fut', c_f, loc_f, scale_f)
rl_hist = gev.ppf(1 - 1 / return_periods, c_h, loc_h, scale_h)
rl_fut = gev.ppf(1 - 1 / return_periods, c_f, loc_f, scale_f)
ci_h_l, ci_h_u = bootstrap_gev(hist_max, return_periods)
ci_f_l, ci_f_u = bootstrap_gev(fut_max, return_periods)
rl_hist[rl_hist <= 0] = np.nan
rl_fut[rl_fut <= 0] = np.nan
ci_h_l[ci_h_l <= 0] = np.nan
ci_h_u[ci_h_u <= 0] = np.nan
ci_f_l[ci_f_l <= 0] = np.nan
ci_f_u[ci_f_u <= 0] = np.nan

plt.figure(figsize=(8, 6), dpi=300)
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'axes.linewidth': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 7,
    'ytick.major.size': 7,
})
hist_color = '#0072B2'
fut_color = '#D55E00'

plt.plot(return_periods, rl_hist, color=hist_color, label=f'Hist. {HIST_PERIOD[0]}-{HIST_PERIOD[1]}', lw=2.5)
plt.fill_between(return_periods, ci_h_l, ci_h_u, color=hist_color, alpha=0.18)
plt.plot(return_periods, rl_fut, color=fut_color, label=f'Fut. {FUT_PERIOD[0]}-{FUT_PERIOD[1]} MAR-CESM2-SSP3-7.0', lw=2.5)
plt.fill_between(return_periods, ci_f_l, ci_f_u, color=fut_color, alpha=0.18)

# empirical return periods of the annual maxima
hist_sorted = np.sort(hist_max)[::-1]
fut_sorted = np.sort(fut_max)[::-1]
hist_rp = (len(hist_sorted) + 1) / np.arange(1, len(hist_sorted) + 1)
fut_rp = (len(fut_sorted) + 1) / np.arange(1, len(fut_sorted) + 1)
plt.scatter(hist_rp, hist_sorted, color=hist_color, s=40, zorder=5, edgecolor='k', linewidth=0.7, label='Hist. annual max')
plt.scatter(fut_rp, fut_sorted, color=fut_color, s=40, zorder=5, edgecolor='k', linewidth=0.7, label='Fut. annual max')

# circle 2021
val_2021 = hist_max[hist_years == 2021][0]
rp_2021 = hist_rp[np.where(hist_sorted == val_2021)[0][0]]
plt.scatter([rp_2021], [val_2021], facecolor='none', edgecolor='red', s=180, zorder=9, linewidth=2)
plt.scatter([rp_2021], [val_2021], color=hist_color, s=40, zorder=10, edgecolor='k', linewidth=0.7)

plt.axhline(hist_max.max(), color=hist_color, ls='--', lw=1.5, label='14 Aug 2021 melt event')
plt.xscale('log')
plt.xlabel('Return period (years)')
plt.ylabel('3-day runoff increase (Gt d$^{-1}$)')
plt.ylim(0, 10)
plt.legend(frameon=False, loc='upper left')
plt.gca().tick_params(axis='x', which='major', length=10)
plt.gca().tick_params(axis='x', which='minor', length=5)
plt.tight_layout(pad=0.5)
out_png = OUT_FIG_DIR / 'Fig11_return_period.png'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
