# M3 lake drainage, Sentinel-1 images and lake area

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import config

LAKE_DIR = os.path.join(config.EXTERNAL_DIR, 'ice_marginal_lake')
OUT_DIR = os.path.join(config.DATA_DIR, 'ice_marginal_lake')

# GrIML lake_id of the lake at 68.778 N, 50.411 W
LAKE_ID = 1073
YEARS = range(2016, 2024)
EVENT_DATE = pd.to_datetime('2021-08-14')

# Sentinel-1 monthly mosaics from Copernicus Browser
IMAGES = ['sept20', 'oct20', 'nov20', 'dec20',
          'sep21', 'oct21', 'nov21', 'dec2021']
IMAGE_DATES = ['Sep 2020', 'Oct 2020', 'Nov 2020', 'Dec 2020',
               'Sep 2021', 'Oct 2021', 'Nov 2021', 'Dec 2021']
LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
CROP_TOP = 26      # px, crops the date stamp
PX_PER_KM = 41     # from the 1 km scale bar

# lake area each year
dates = []
areas = []
for year in YEARS:
    inventory = gpd.read_file(os.path.join(LAKE_DIR, f'{year}0101-ESA-GRIML-IML-fv3.gpkg'))
    lake = inventory[inventory['lake_id'] == LAKE_ID]
    area = lake.geometry.union_all().area / 1e6   # m2 to km2
    dates.append(pd.to_datetime(f'{year}-08-01'))
    areas.append(area)
    print(year, round(area, 2), 'km2')

df = pd.DataFrame({'year': list(YEARS), 'area_sqkm': areas})
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(os.path.join(OUT_DIR, 'M3_lake_area.csv'), index=False)

# figure
FONT = 9
FSM = 8
fig = plt.figure(figsize=(7.5, 5.6))
gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1.1],
                       wspace=0.03, hspace=0.04)

# images
for i in range(8):
    img = plt.imread(os.path.join(LAKE_DIR, f'{IMAGES[i]}.png'))
    img = img[CROP_TOP:, :]
    ax_img = fig.add_subplot(gs[i // 4, i % 4])
    ax_img.imshow(img)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.text(0.03, 0.04, f'({LETTERS[i]})', transform=ax_img.transAxes,
                fontsize=FONT, fontweight='bold', color='white', va='bottom',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1.5))
    ax_img.text(0.97, 0.96, IMAGE_DATES[i], transform=ax_img.transAxes,
                fontsize=FSM, color='white', ha='right', va='top',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1.5))

    # 5 km scale bar on the first image
    if i == 0:
        ny = img.shape[0]
        nx = img.shape[1]
        x1 = nx - 40
        x0 = x1 - 5 * PX_PER_KM
        y = ny - 50
        ax_img.plot([x0, x1], [y, y], color='white', lw=2.5, solid_capstyle='butt')
        ax_img.text((x0 + x1) / 2, y - 20, '5 km', color='white', fontsize=FSM,
                    ha='center', va='bottom', fontweight='bold')

# lake area timeseries
ax = fig.add_subplot(gs[2, :])
pos = ax.get_position()
ax.set_position([pos.x0, pos.y0, pos.width, pos.height - 0.04])

ax.plot(dates, areas, color='#2C5F8A', lw=1.5, marker='o', ms=5, zorder=3)
ax.axvline(EVENT_DATE, color='#E84855', lw=1.2, ls='--', zorder=2, label='14 August 2021')

ax.set_xlim(pd.to_datetime('2016-01-01'), pd.to_datetime('2024-01-01'))
ax.set_ylim(0, 40)
# year ticks on 1 Jan
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylabel('Lake area (km²)', fontsize=FONT)
ax.tick_params(labelsize=FSM)
ax.grid(True, axis='y', alpha=0.18, lw=0.5)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=FSM, loc='upper right', frameon=False)
ax.text(0.01, 0.97, '(i)', transform=ax.transAxes,
        fontsize=FONT, fontweight='bold', va='top')

out = os.path.join(config.FIG_DIR, 'FigS10_M3_lake_drainage.png')
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
