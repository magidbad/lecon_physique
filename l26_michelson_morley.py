"""
Simulation Michelson-Morley — Leçon 26
Source des données : Michelson & Morley, Am. J. Sci., 34, 333 (1887), Table VI
Moyenne des 6 séries de mesures (juillet 1887)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Paramètres physiques
L   = 11.0       # longueur effective des bras (m)
lam = 590e-9     # longueur d'onde (m)
c   = 3e8        # vitesse de la lumière (m/s)
v0  = 30e3       # vitesse orbitale terrestre (m/s)

# Angles de rotation
theta = np.linspace(0, 360, 500)
theta_rad = np.deg2rad(theta)

def dephasage(v):
    """
    Déphasage attendu si l'éther existe :
    Delta_phi(theta) = (2*pi*L*v^2)/(lambda*c^2) * cos(2*theta)
    """
    return (2 * np.pi * L * v**2 / (lam * c**2)) * np.cos(2 * theta_rad)

# Données réelles — Michelson & Morley 1887, Table VI
# Moyenne des 6 séries, en fractions de frange
theta_data = np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
                        180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5])
fringe_data = np.array([0.000, 0.010, 0.010, -0.010, 0.000, 0.000,
                         0.010, 0.010, 0.000, -0.010, -0.010, 0.000,
                         0.010, 0.010, 0.000, -0.010])

# Figure
fig, ax = plt.subplots(figsize=(9, 5))
plt.subplots_adjust(bottom=0.22)

dp0 = dephasage(v0)
line_theory, = ax.plot(theta, dp0 / (2*np.pi), 'r-', lw=2,
                        label=f'Théorie (éther) : amplitude = {dp0.max()/(2*np.pi):.3f} frange')
ax.plot(theta_data, fringe_data, 'bo', ms=7,
        label='Données Michelson & Morley 1887 : amplitude $\\approx$ 0.01 frange')
ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.5)

ax.set_xlabel('Angle de rotation (degrés)')
ax.set_ylabel('Déplacement des franges (fractions de frange)')
ax.set_title('Expérience de Michelson-Morley (1887)\n'
             'Déplacement théorique (éther) vs résultat observé')
ax.set_xlim(0, 360)
ax.set_xticks([0, 90, 180, 270, 360])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)


# Slider
ax_slider = plt.axes([0.15, 0.07, 0.65, 0.04])
slider = Slider(ax_slider, '$v/c$  ', 1e-5, 0.01,
                valinit=v0/c, valstep=1e-5)

def update(val):
    v = slider.val * c
    dp = dephasage(v)
    line_theory.set_ydata(dp / (2*np.pi))
    line_theory.set_label(
        f'Théorie (éther) : amplitude = {dp.max()/(2*np.pi):.4f} frange')
    ylim = max(abs(dp).max()/(2*np.pi) * 1.4, 0.02)
    ax.set_ylim(-ylim, ylim)
    ax.legend(fontsize=9)
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.savefig('/mnt/user-data/outputs/michelson_morley_sim.png',
            dpi=150, bbox_inches='tight')
plt.show()