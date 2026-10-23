# Paths and study area

import os

# repo directories
REPO_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(REPO_DIR, 'data', 'derived')    # outputs/ derived data
CACHE_DIR = os.path.join(DATA_DIR, 'cache')                # intermediate files
EXTERNAL_DIR = os.path.join(REPO_DIR, 'data', 'external')
FIG_DIR   = os.path.join(REPO_DIR, 'figures')

ROI_SHAPEFILE_PATH = os.path.join(DATA_DIR, 'roi_shapefiles', 'ROI_shapefiles.shp')
WATSON_PATH = os.path.join(EXTERNAL_DIR, 'watson', 'watson_river_discharge_2006_2025_daily.txt') #Bahbah et al (https://doi.org/10.22008/FK2/XEHYCM)

# external data
EXT_DATA_DIR = '/exports/csce/datastore/geos/users/s1834371'

VELOCITY_DATA_DIR = os.path.join(EXT_DATA_DIR, 'PROMICE_ICE_VELOCITIES')     # PROMICE Sentinel-1 ice velocity (Solgaard et al., https://doi.org/10.22008/FK2/K70OPK)
BEDMACHINE_PATH   = os.path.join(EXT_DATA_DIR, 'data', 'BedMachineGreenland-v6.nc') # Morlighem et al., https://doi.org/10.5067/7U2049INNRMP
MASK_PATH_FILE    = os.path.join(EXT_DATA_DIR, 'data', 'ice_mask_reprojected_full.nc') # from bedmachine data
GIMP_DEM_PATH     = os.path.join(EXT_DATA_DIR, 'Chapter_1', 'LATE-SEASON-MELT', 'data', 'QGIS_FIGURE', 'data', 'gimpdem_merged_SW.tif') # GIMP DEM (Howat et al., https://doi.org/10.5067/B8X58MQBFUPA)
MAR_DIR           = os.path.join(EXT_DATA_DIR, 'data', 'MAR_data')            # MARv3.14.3 ERA5, 10 km daily
MAR_CESM2_DIR     = os.path.join(EXT_DATA_DIR, 'data', 'MAR_CESM2-daily')     # MAR forced by CESM2 SSP3-7.0
THINNING_PATH     = os.path.join(EXT_DATA_DIR, 'data', 'khan_etal_2025_thinning_estimates',
                                 'Greenland_dhdt_icevol_1kmgrid_DB.nc') # Khan et al., 2025 (https://datadryad.org/dataset/doi:10.5061/dryad.s4mw6m9dh)
MANKOFF_DIR       = os.path.join(EXT_DATA_DIR, 'data', 'mankoff_runoff', 'freshwater', 'ice') # Mankoff et al., 2020 (https://doi.org/10.22008/FK2/XKQVL7)

# study area
# land-terminating ROIs (L1-L10)
ROI_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# marine-terminating ROIs (M1-M3)
MTG_ROI_LABELS = [11, 12, 13]

# study extent [xmin, xmax, ymin, ymax], ROIs + 40 km (+45 km east)
STUDY_EXTENT = [-272820, -117228, -2613666, -2316305]
