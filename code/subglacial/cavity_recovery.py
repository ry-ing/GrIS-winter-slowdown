# Cavity recovery for L3 following Kamb (1987) and Harper et al. (2021)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'derived')

# constants
BAR = 1.0e5                  # Pa
ETA_R = 0.1 * BAR            # Pa yr, Kamb (1987) reference viscosity
SIG_R = 6.0 * BAR            # Pa, Kamb (1987) reference effective pressure
RHO_I = 917.0                # kg m-3
RHO_W = 1000.0               # kg m-3
G = 9.81                     # m s-2
L_FUS = 3.34e5               # J kg-1
SEC_YR = 365.25 * 24 * 3600
Q_GEO = 0.027                # W m-2, geothermal heat flux from Harper et al. (2021)
WINTER_FRAC = 212 / 365.25   # 212 day winter (1 Oct to 30 Apr)

# for ROI L3
TAU_B = RHO_I * G * 500.0 * np.sin(np.radians(2.3))   # Pa, 500 m ice, ~2.3 deg slope
# L3 baseline and winter velocities from prepare_vel_scatter_data.py
roi = pd.read_csv(os.path.join(DATA_DIR, 'scatter_plots', 'roi_recovery.csv'))
l3 = roi[roi['roi_label'] == 3].iloc[0]
U_BASE = l3['baseline_vel']  # m/yr, baseline winter velocity
U_SUPP = l3['W1_vel']        # m/yr, 2021/22 winter velocity
U_WIN = np.array([l3['W1_vel'], l3['W2_vel'], l3['W3_vel'], l3['W4_vel']])   # m/yr, W1-W4
print(f"L3 baseline {U_BASE:.1f} m/yr, W1-W4 {np.round(U_WIN, 1)} m/yr")
H_STEP = 1.0                 # m, step height
LS = 60.0                    # m, step spacing



# cavity storage in m w.e. (from Kamb 1987)
def storage(u_b, N):
    eta = ETA_R * (SIG_R / N)**2
    Lc = 4.0 * np.sqrt(eta * u_b * H_STEP / (np.pi * N))
    return 0.5 * H_STEP * Lc / LS


# basal melt rate in m/yr
def melt_rate(u_b, rho=RHO_W):
    q = (u_b / SEC_YR) * TAU_B + Q_GEO
    return (q / (rho * L_FUS)) * SEC_YR


# winters to refill the cavities after draining a fraction f
def simulate_recovery(f):
    S = (1.0 - f) * S_PRE
    for w in range(4):
        S = S + melt_rate(U_WIN[w]) * WINTER_FRAC
        if S >= storage(U_WIN[w], N_CAL):
            return w + 1
    return np.inf


# calibrate N to the Harper et al. (2021) cavity (Lc is 33 m, h = 1 m, u_b = 100 m/yr)
N_CAL = (16.0 * ETA_R * SIG_R**2 * 100.0 * 1.0 / (np.pi * 33.0**2))**(1 / 3)
P_I = RHO_I * G * 500.0 / BAR   # overburden at 500 m ice thickness in bar
Lc_pre = 4.0 * np.sqrt(ETA_R * (SIG_R / N_CAL)**2 * U_BASE * H_STEP / (np.pi * N_CAL))
S_PRE = storage(U_BASE, N_CAL)
print(f"N_cal = {N_CAL / BAR:.2f} bar ({(N_CAL / BAR) / P_I * 100:.1f}% of overburden at 500 m)")
print(f"tau_b = {TAU_B / 1e3:.1f} kPa")
print(f"pre-event cavity length {Lc_pre:.2f} m, storage {S_PRE * 100:.2f} cm w.e.")

# melt
m_winter = melt_rate(U_WIN[0]) * WINTER_FRAC
print(f"melt rate pre-event {melt_rate(U_BASE, RHO_I) * 100:.2f} cm/yr ice, 2021/22 {melt_rate(U_WIN[0], RHO_I) * 100:.2f} cm/yr ice")
print(f"one winter of melt = {m_winter * 100:.2f} cm w.e. ({m_winter / S_PRE * 100:.1f}% of the store)")

# storage change during the within-winter speed up (82 to 121 m/yr)
S_82 = storage(82.0, N_CAL)
S_121 = storage(121.0, N_CAL)
print(f"storage 2021/22 start {storage(U_SUPP, N_CAL) * 100:.2f} cm w.e., +{(S_121 - S_82) / S_82 * 100:.1f}% from 82 to 121 m/yr")

# drained fraction that gives a 3 winter recovery
f_all = np.arange(0.01, 0.99, 0.005)
rec_all = np.array([simulate_recovery(f) for f in f_all])
f_min = f_all[rec_all == 3].min()
f_max = f_all[rec_all == 3].max()
print(f"3 winter recovery: f = {f_min * 100:.0f}% to {f_max * 100:.0f}%")

# effective pressure and bed contact for f = 0.30
N_new = N_CAL * 0.7**(-2.0 / 3.0)
c_pre = 1.0 - Lc_pre / LS
c_new = 1.0 - Lc_pre * 0.7 / LS
print(f"f = 30%: N {N_CAL / BAR:.2f} -> {N_new / BAR:.2f} bar, water pressure {(1.0 - (N_new / BAR) / P_I) * 100:.1f}% of overburden, "
      f"contact {c_pre:.2f} -> {c_new:.2f}")

# basal melt reduction 
m_pre = melt_rate(U_BASE, RHO_I)
m_supp = melt_rate(U_SUPP, RHO_I)
deficit = TAU_B * (U_BASE - U_SUPP) / (RHO_W * L_FUS) * WINTER_FRAC * 100
print(f"melt {m_pre * 100:.2f} -> {m_supp * 100:.2f} cm/yr ice ({(m_pre - m_supp) / m_pre * 100:.1f}% less), "
      f"winter deficit {deficit:.2f} cm w.e.")


# figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2
})

# storage refilling each winter, and recovery time against f
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5), dpi=300)

winters = np.arange(0, 5)
f_examples = [0.15, 0.25, 0.30, 0.40]
colors = ['#2b5c8f', '#4682b4', '#d95f02', '#7570b3']
for f_val, col in zip(f_examples, colors):
    S = (1.0 - f_val) * S_PRE
    S_traj = [S * 100]
    for u in U_WIN:
        S = S + melt_rate(u) * WINTER_FRAC
        S_traj.append(S * 100)
    ax1.plot(winters, S_traj, marker='o', ms=3, color=col, label=f'$f = {f_val:.2f}$')

ax1.axhline(S_PRE * 100, color='black', linestyle='--', linewidth=0.8, label='$S_{\\mathrm{pre}}$')
ax1.set_xlabel('Winter season')
ax1.set_ylabel('Storage depth (cm w.e.)')
ax1.set_xticks(winters)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=False, loc='lower right')

f_grid = np.linspace(0.05, 0.45, 100)
recovery_times = [simulate_recovery(f) for f in f_grid]
ax2.plot(f_grid * 100, recovery_times, color='#2b5c8f')
ax2.axvspan(f_min * 100, f_max * 100, color='#d95f02', alpha=0.15)
ax2.set_xlabel('Drained fraction $f$ (%)')
ax2.set_ylabel('Recovery duration (winters)')
ax2.set_ylim(0.5, 4.5)
ax2.set_yticks([1, 2, 3, 4])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'FigS9_cavity_recovery.png'), bbox_inches='tight', dpi=300)
plt.close()


# cavity shape before and after draining 30%, and bed contact against f
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5), dpi=300)

for L, col, ls, alpha, label in [(Lc_pre, '#2b5c8f', '-', 0.1, f'Pre-event ($L_c = {Lc_pre:.1f}$ m)'),
                                 (Lc_pre * 0.7, '#d95f02', '--', 0.15, 'Post-event ($f = 30\\%$)')]:
    x = np.linspace(0, L, 200)
    xi = np.clip((2.0 * x - L) / L, -1.0, 1.0)
    g = H_STEP * (0.5 - np.arcsin(xi) / np.pi - 2.0 * (2.0 * x - L) * np.sqrt(np.maximum(x * (L - x), 0.0)) / (np.pi * L**2))
    ax1.plot(x, g, color=col, linestyle=ls, label=label)
    ax1.fill_between(x, 0, g, color=col, alpha=alpha)
ax1.set_xlabel('Distance along step $x$ (m)')
ax1.set_ylabel('Cavity gap height $g(x)$ (m)')
ax1.set_xlim(0, LS)
ax1.set_ylim(0, 1.05)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=False, loc='upper right')

f_plot = np.linspace(0.0, 0.45, 100)
contact = 1.0 - (Lc_pre * (1.0 - f_plot) / LS)
N_plot = N_CAL * (1.0 - f_plot)**(-2.0 / 3.0) / BAR
ax2_twin = ax2.twinx()
ax2.plot(f_plot * 100, contact, color='#2b5c8f', label='Contact fraction')
ax2_twin.plot(f_plot * 100, N_plot / contact, color='#d95f02', linestyle='--', label='Contact stress')
ax2.set_xlabel('Drained fraction $f$ (%)')
ax2.set_ylabel('Ice--bed contact fraction $c$', color='#2b5c8f')
ax2_twin.set_ylabel('Mean contact stress $N/c$ (bar)', color='#d95f02')
ax2.tick_params(axis='y', labelcolor='#2b5c8f')
ax2_twin.tick_params(axis='y', labelcolor='#d95f02')
ax2.spines['top'].set_visible(False)
ax2_twin.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'extra-fig_cavity_geometry.png'), bbox_inches='tight', dpi=300)
plt.close()
