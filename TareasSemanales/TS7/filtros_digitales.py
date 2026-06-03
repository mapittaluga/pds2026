import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.ticker

 
# ─────────────────────────────────────────────
#  DISEÑO DEL FILTRO
# ─────────────────────────────────────────────
fs      = 1000
N       = fs
fc      = [1, 35]
fst     = [0.1, 50]
ripple_db = 1.0
att_db  = 40
 
b, a = signal.iirdesign(
    wp=fc, ws=fst,
    gpass=ripple_db, gstop=att_db,
    analog=False, ftype='butter',
    output='ba', fs=fs
)

sos = signal.iirdesign(
    wp=fc, ws=fst,
    gpass=ripple_db, gstop=att_db,
    analog=False, ftype='butter',
    output='sos', fs=fs
)
 
 
# ─────────────────────────────────────────────
#  RESPUESTA EN FRECUENCIA
# ─────────────────────────────────────────────
# w, H = signal.freqz(b, a, worN=N, fs=fs)
w, H = signal.freqz_sos(sos, worN=N, fs=fs)
magnitude_dB = 20 * np.log10(np.abs(H) + 1e-300)   # evita log(0)
phase_rad    = np.unwrap(np.angle(H))
 
# ─────────────────────────────────────────────
#  POLOS Y CEROS
# ─────────────────────────────────────────────
zeros_arr = np.roots(b)
poles_arr = np.roots(a)
 
# Contar multiplicidad (tolerancia 1e-6)
def _group(vals, tol=1e-6):
    """Devuelve lista de (valor_representativo, multiplicidad)."""
    used = np.zeros(len(vals), dtype=bool)
    groups = []
    for i, v in enumerate(vals):
        if used[i]:
            continue
        close = np.abs(vals - v) < tol
        mult  = int(close.sum())
        used |= close
        groups.append((v, mult))
    return groups
 
zero_groups = _group(zeros_arr)
pole_groups = _group(poles_arr)
 
# ─────────────────────────────────────────────
#  FIGURA: 3 subplots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Análisis del Filtro IIR (Butterworth Banda Pasante)', fontsize=13, fontweight='bold')
 
nticks = 8
 
# ── 1. Magnitud ──────────────────────────────
ax_mag = axes[0]
ax_mag.plot(w, magnitude_dB, color='steelblue', linewidth=1.5)
ax_mag.set_title('Respuesta en Magnitud')
ax_mag.set_xlabel('Frecuencia [Hz]')
ax_mag.set_ylabel('Amplitud [dB]')
ax_mag.set_ylim([-120, 20])
ax_mag.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
ax_mag.grid(True, linestyle='--', alpha=0.6)
# Marcas de frecuencias de diseño
for f, lbl in zip(fst + fc, ['fst_low', 'fst_high', 'fc_low', 'fc_high']):
    ax_mag.axvline(f, color='red' if 'fc' in lbl else 'orange',
                   linestyle=':', linewidth=1, alpha=0.8)
 
# ── 2. Fase ───────────────────────────────────
ax_pha = axes[1]
ax_pha.plot(w, phase_rad, color='seagreen', linewidth=1.5)
ax_pha.set_title('Respuesta en Fase')
ax_pha.set_xlabel('Frecuencia [Hz]')
ax_pha.set_ylabel('Fase [rad]')
ax_pha.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
ax_pha.grid(True, linestyle='--', alpha=0.6)
for f, lbl in zip(fst + fc, ['fst_low', 'fst_high', 'fc_low', 'fc_high']):
    ax_pha.axvline(f, color='red' if 'fc' in lbl else 'orange',
                   linestyle=':', linewidth=1, alpha=0.8)
 
# ── 3. Diagrama Polos / Ceros ─────────────────
ax_pz = axes[2]
 
# Círculo unitario
theta = np.linspace(0, 2 * np.pi, 500)
ax_pz.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.8, alpha=0.5, label='Círculo unitario')
ax_pz.axhline(0, color='k', linewidth=0.5, alpha=0.4)
ax_pz.axvline(0, color='k', linewidth=0.5, alpha=0.4)
 
# Ceros
for z, mult in zero_groups:
    ax_pz.plot(z.real, z.imag, 'o',
               markersize=9, markerfacecolor='none',
               markeredgecolor='steelblue', markeredgewidth=1.8)
    if mult > 1:
        ax_pz.annotate(f' {mult}', xy=(z.real, z.imag),
                       fontsize=8, color='steelblue',
                       va='bottom', ha='left')
 
# Polos
for p, mult in pole_groups:
    ax_pz.plot(p.real, p.imag, 'x',
               markersize=9, markeredgecolor='firebrick', markeredgewidth=2)
    if mult > 1:
        ax_pz.annotate(f' {mult}', xy=(p.real, p.imag),
                       fontsize=8, color='firebrick',
                       va='bottom', ha='left')
 
# Leyenda manual
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='steelblue', markeredgewidth=1.8, markersize=9, label='Ceros'),
    Line2D([0], [0], marker='x', color='firebrick',
           markeredgewidth=2, markersize=9, linestyle='None', label='Polos'),
]
ax_pz.legend(handles=legend_elements, fontsize=8, loc='upper left')
 
ax_pz.set_title('Diagrama de Polos y Ceros')
ax_pz.set_xlabel('Parte Real')
ax_pz.set_ylabel('Parte Imaginaria')
ax_pz.set_aspect('equal')
ax_pz.grid(True, linestyle='--', alpha=0.4)
 
# Padding automático con margen
all_pts = np.concatenate([zeros_arr, poles_arr])
margin  = 0.2
lim     = max(np.max(np.abs(all_pts)) + margin, 1.3)
ax_pz.set_xlim(-lim, lim)
ax_pz.set_ylim(-lim, lim)
 
plt.tight_layout()

 
