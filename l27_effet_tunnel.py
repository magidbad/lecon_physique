"""
Effet tunnel — Leçon 27
Deux panneaux :
  Gauche : densité de probabilité |psi|^2 pour une barrière rectangulaire
  Droite  : loi de Geiger-Nuttall (données réelles, régression linéaire)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import stats

# ══════════════════════════════════════════════════════════
# PANNEAU GAUCHE : densité de probabilité |psi|^2
# ══════════════════════════════════════════════════════════

def calcul_psi2(E_sur_V0, kappa_a):
    """
    Barrière rectangulaire de hauteur V0, largeur a.
    Retourne x et |psi(x)|^2 normalisés.
    """
    # Zones : x < 0 (zone 1), 0 < x < a (zone 2), x > a (zone 3)
    # k = sqrt(2mE)/hbar, kappa = sqrt(2m(V0-E))/hbar
    # On pose a=1 (unité de longueur), kappa*a = kappa_a

    E_V0 = E_sur_V0
    kappa = kappa_a          # kappa en unités de 1/a
    k     = kappa * np.sqrt(E_V0 / (1 - E_V0))  # depuis k/kappa = sqrt(E/(V0-E))

    # Coefficient de transmission (barrière épaisse)
    T = np.exp(-2 * kappa_a)

    # Amplitudes (normalisées à A1 = 1)
    A1 = 1.0
    A3 = np.sqrt(T)

    # Zone 1 : onde incidente + réfléchie
    x1 = np.linspace(-2, 0, 300)
    # |psi|^2 ≈ 1 + R + interférences ~ oscillations
    r = (1 - T)
    psi2_1 = 1 + r + 2*np.sqrt(r)*np.cos(2*k*x1 + np.pi)

    # Zone 2 : onde évanescente (+ anti-évanescente)
    x2 = np.linspace(0, 1, 200)
    # |psi|^2 ~ cosh^2(kappa*(x-a)) approximation barrière épaisse
    psi2_2 = np.cosh(kappa * (x2 - 1))**2 * T / np.cosh(kappa)**2

    # Zone 3 : onde transmise
    x3 = np.linspace(1, 3, 300)
    psi2_3 = np.full_like(x3, T)

    x_all   = np.concatenate([x1, x2, x3])
    psi2_all = np.concatenate([psi2_1, psi2_2, psi2_3])
    return x_all, psi2_all, T

# ══════════════════════════════════════════════════════════
# PANNEAU DROIT : loi de Geiger-Nuttall
# Données : NNDC NuDat 3, National Nuclear Data Center,
# Brookhaven National Laboratory
# https://www.nndc.bnl.gov/nudat3/
# Émetteurs alpha — éléments naturels
# ══════════════════════════════════════════════════════════

# (noyau, Z, E_alpha en MeV, t_1/2 en secondes)
donnees = [
    ("Po-212",  84, 8.784,  2.99e-7),
    ("Po-214",  84, 7.687,  1.64e-4),
    ("Po-216",  84, 6.778,  0.145),
    ("Po-218",  84, 6.002,  186.0),
    ("Rn-220",  86, 6.288,  55.6),
    ("Rn-222",  86, 5.590,  3.30e5),
    ("Ra-224",  88, 5.685,  3.16e5),
    ("Ra-226",  88, 4.871,  5.05e10),
    ("Th-228",  90, 5.423,  6.03e7),
    ("Th-232",  90, 4.082,  4.43e17),
    ("U-234",   92, 4.858,  7.74e12),
    ("U-238",   92, 4.270,  1.41e17),
]

noms   = [d[0] for d in donnees]
Z_list = np.array([d[1] for d in donnees])
E_list = np.array([d[2] for d in donnees])   # MeV
t_list = np.array([d[3] for d in donnees])   # s

# Geiger-Nuttall : ln(t_1/2) = A + B * Z / sqrt(E)
x_GN = Z_list / np.sqrt(E_list)
y_GN = np.log(t_list)

slope, intercept, r, p, se = stats.linregress(x_GN, y_GN)

# ══════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.25, wspace=0.35)

# ── Gauche ────────────────────────────────────────────────
x0, p0, T0 = calcul_psi2(0.5, 3.0)

line_psi, = ax1.plot(x0, p0, 'b-', lw=2)
ax1.axvspan(0, 1, alpha=0.15, color='red', label='Barrière $V_0$')
ax1.axhline(0, color='k', lw=0.5)
ax1.set_xlabel('Position $x/a$')
ax1.set_ylabel('$|\\psi|^2$ (u.a.)')
ax1.set_title('Densité de probabilité\npour une barrière rectangulaire')
ax1.set_xlim(-2, 3)
ax1.set_ylim(-0.1, 3)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
txt_T = ax1.text(0.98, 0.95, f'$T = {T0:.3f}$',
                  transform=ax1.transAxes, ha='right', va='top',
                  fontsize=11, color='darkblue',
                  bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Sliders
ax_E  = plt.axes([0.08, 0.12, 0.35, 0.03])
ax_ka = plt.axes([0.08, 0.07, 0.35, 0.03])
sl_E  = Slider(ax_E,  '$E/V_0$   ', 0.05, 0.95, valinit=0.5,  valstep=0.01)
sl_ka = Slider(ax_ka, '$\\kappa a$', 0.5,  8.0,  valinit=3.0,  valstep=0.1)

def update(_):
    x, p, T = calcul_psi2(sl_E.val, sl_ka.val)
    line_psi.set_data(x, p)
    txt_T.set_text(f'$T = {T:.2e}$')
    fig.canvas.draw_idle()

sl_E.on_changed(update)
sl_ka.on_changed(update)

# ── Droite ────────────────────────────────────────────────
x_fit = np.linspace(x_GN.min(), x_GN.max(), 100)
y_fit = slope * x_fit + intercept

ax2.plot(x_GN, y_GN, 'ro', ms=7, label='Données NNDC NuDat 3')
ax2.plot(x_fit, y_fit, 'b-', lw=2,
         label=f'Régression : $R^2 = {r**2:.4f}$')

for i, nom in enumerate(noms):
    ax2.annotate(nom, (x_GN[i], y_GN[i]),
                 fontsize=7, textcoords='offset points',
                 xytext=(4, 2), color='gray')

ax2.set_xlabel('$Z / \\sqrt{E_\\alpha}$ (MeV$^{-1/2}$)')
ax2.set_ylabel('$\\ln(t_{1/2})$ ($t_{1/2}$ en s)')
ax2.set_title('Loi de Geiger-Nuttall\n'
              '$\\ln t_{1/2} = A + B\\,Z/\\sqrt{E_\\alpha}$')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle('Effet tunnel — Leçon 27', fontsize=13)

plt.savefig('/mnt/user-data/outputs/effet_tunnel_sim.png',
            dpi=150, bbox_inches='tight')
plt.show()