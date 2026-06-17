import numpy as np
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
from pytc2.sistemas_lineales import plot_plantilla

# ─────────────────────────────────────────────
#  LECTURA DEL ECG
# ─────────────────────────────────────────────
fs_ecg = 1000  # Hz
fs = fs_ecg

mat_struct = loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
t_ecg = np.arange(N) / fs

hb_1 = mat_struct['heartbeat_pattern1'].flatten()
hb_2 = mat_struct['heartbeat_pattern2'].flatten()
t_hb1 = np.arange(len(hb_1)) / fs
t_hb2 = np.arange(len(hb_2)) / fs

# ─────────────────────────────────────────────
#  PLANTILLA DEL FILTRO
# ─────────────────────────────────────────────
fc = [1, 35]    # frecuencias de paso (Hz)
fst = [0.1, 45]  # frecuencias de stop (Hz)
ripple_db = 1
att_db = 40

# ─────────────────────────────────────────────
#  PARÁMETROS Y DISEÑO DEL FILTRO
# ─────────────────────────────────────────────

# odd numtaps, antisymmetric is False, type I filter is produced
# even numtaps, antisymmetric is False, type II filter is produced
# odd numtaps, antisymmetric is True, type III filter is produced
# even numtaps, antisymmetric is True, type IV filter is produced

freq = [0, 0.1, 1, 35, 45, 500]
freq_win = freq
freq_ls = [0, 0.1, 1, 35, 36, 500]
gain = [0, 0, 1, 1, 0, 0]
gain_win = gain
gain_ls = [0, 0, 1, 1, 0, 0]
Ntaps_win = 1801
Ntaps_ls = 1801
win = 'boxcar'

b_win = signal.firwin2(numtaps=Ntaps_win, freq=freq_win,
                       gain=gain_win, window=win, antisymmetric=True, fs=fs)
b_ls = signal.firls(numtaps=Ntaps_ls, bands=freq_ls,
                    desired=gain_ls, weight=None, fs=fs)
# ─────────────────────────────────────────────
#  RESPUESTA EN FRECUENCIA
# ─────────────────────────────────────────────
ww = np.concatenate([
    np.logspace(np.log10(0.01), np.log10(2),   250),
    np.linspace(2.1,            34,             N),
    np.logspace(np.log10(34.1), np.log10(52),  250),
    np.linspace(52.1,           fs / 2,         N),
])
_, H_win = signal.freqz(b_win, worN=ww, fs=fs)
mag_win_dB = 20 * np.log10(np.abs(H_win) + 1e-300)
phase_win = np.unwrap(np.angle(H_win)) * 180 / np.pi

_, H_ls = signal.freqz(b_ls, worN=ww, fs=fs)
mag_ls_dB = 20 * np.log10(np.abs(H_ls) + 1e-300)
phase_ls = np.unwrap(np.angle(H_ls)) * 180 / np.pi

# ─────────────────────────────────────────────
#  POLOS Y CEROS
# ─────────────────────────────────────────────
z_win = np.roots(b_win)          # ceros
p_win = np.zeros(len(b_win) - 1)  # todos los polos en z = 0

z_ls = np.roots(b_ls)          # ceros
p_ls = np.zeros(len(b_ls) - 1)  # todos los polos en z = 0

# ─────────────────────────────────────────────
#  FIGURA 1 — MÓDULO
# ─────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 4))

ax1.plot(ww, mag_win_dB, 'b', linewidth=1.5, label='Metodo de Ventana')
ax1.plot(ww, mag_ls_dB, 'r', linewidth=1.5, label='Cuadrados Minimos')

axes_hdl = plt.gca()

plot_plantilla(filter_type='bandpass', fpass=fc, fstop=fst,
               ripple=ripple_db, attenuation=att_db, fs=fs)
_ = axes_hdl.legend()

ax1.set_xlim([0, fs / 2])
ax1.set_ylim([-80, 5])
ax1.set_xlabel('Frecuencia (Hz)')
ax1.set_ylabel('Magnitud (dB)')
ax1.set_title('Respuesta en Frecuencia — Módulo')
ax1.legend(fontsize=8, loc='lower right', ncol=2)
ax1.grid(True, which='both', linestyle=':', alpha=0.6)
plt.tight_layout()

# ─────────────────────────────────────────────
#  FIGURA 2 — FASE
# ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 4))

ax2.plot(ww, phase_win, 'b', linewidth=1.5, label='Metodo de Ventana')
ax2.plot(ww, phase_ls,  'r', linewidth=1.5, label='Cuadrados Minimos')

ax2.set_xlim([0, fs / 2])
ax2.set_xlabel('Frecuencia (Hz)')
ax2.set_ylabel('Fase (grados)')
ax2.set_title('Respuesta en Frecuencia — Fase')
ax2.legend(fontsize=8)
ax2.grid(True, which='both', linestyle=':', alpha=0.6)
plt.tight_layout()

# ─────────────────────────────────────────────
#  FIGURA 3 — POLOS Y CEROS
# ─────────────────────────────────────────────
theta = np.linspace(0, 2 * np.pi, 512)
filtros = [
    ('Metodo de Ventana',      z_win,  p_win,  'b'),
    ('Cuadrados Minimos',      z_ls, p_ls, 'r'),
]

fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (nombre, z, p, color) in zip(axes, filtros):
    ax.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.8, alpha=0.4)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.4)

    ax.scatter(z.real, z.imag,
               marker='o', s=50, facecolors='none',
               edgecolors=color, linewidths=1.5, zorder=5, label='Ceros')
    ax.scatter(p.real, p.imag,
               marker='x', s=50,
               c=color, linewidths=1.5, zorder=5, label='Polos')

    margin = 0.2
    lim = 1 + margin
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal')
    ax.set_xlabel('Parte real')
    ax.set_ylabel('Parte imaginaria')
    ax.set_title(f'Polos y Ceros — {nombre}')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
