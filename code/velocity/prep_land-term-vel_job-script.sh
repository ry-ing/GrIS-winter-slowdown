#!/bin/bash
#SBATCH --job-name=prep-landterm
#SBATCH --account=geos_extra
#SBATCH --partition=General
#SBATCH --nodelist=forth04
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/s1834371/scratch/dask_log/prep-landterm-%j.out

source ~/.bashrc
conda activate c3_vel_proc
cd /scratch/s1834371/GrIS_winter_slowdown-repo_clean/code/velocity
python -u prepare_land-term_velocity_data.py
