# Channel closure timescale

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Parameters
n = 3
rho_i = 917.0
g = 9.81
s_per_day = 86400.0

# Glen's A [Pa^-3 s^-1]
A_vals = {'0 °C': 2.4e-24, '−5 °C': 9.3e-25, '−10 °C': 3.5e-25}

H = np.linspace(100, 1000, 1000)
N = rho_i * g * H  # Full overburden: P_w = 0

tau = {T: (n**n) / (2 * A * N**n) / s_per_day for T, A in A_vals.items()}

fig, ax = plt.subplots(figsize=(4, 3))

# 300-700 m thickness range
ax.axvspan(300, 700, color='grey', alpha=0.15, lw=0)
for T in A_vals:
    ax.semilogy(H, tau[T], lw=2, label=T)
# 3 days
ax.axhline(3, color='k', lw=0.8, ls=':')
ax.text(790, 3.4, '3 days', ha='right', va='bottom', fontsize=9)

ax.set_xlabel('Ice thickness (m)', fontsize=10)
ax.set_ylabel('Channel closure timescale (days)', fontsize=10)
ax.set_xlim(100, 800)
ax.set_ylim(0.1, 1e3)
ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures')
plt.savefig(os.path.join(fig_dir, 'FigS1_channel_closure.png'), dpi=180, bbox_inches='tight')
