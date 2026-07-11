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
ripple_db_but = ripple_db/2
att_db_but = att_db/2
# Para el Win es la mitad porque lo paso dos veces

# 0,1 0,5 35 36 45
# 40 0,9 
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
freq_rz = [0, 0.1, 1, 35, 36, 500]
gain = [0, 0, 1, 1, 0, 0]
gain_win = gain
gain_ls = [0, 0, 1, 1, 0, 0]
gain_rz = [0, 1, 0]
weigth_rz = [1, 1, 10]
Ntaps_win = 1101
Ntaps_ls = 1801
Ntaps_rz = 1501
win = 'boxcar'

b_win = signal.firwin2(numtaps=Ntaps_win, freq=freq_win,
                       gain=gain_win, window=win, antisymmetric=True, fs=fs)
b_ls = signal.firls(numtaps=Ntaps_ls, bands=freq_ls,
                    desired=gain_ls, weight=None, fs=fs)

b_rz = signal.remez(numtaps=Ntaps_rz, bands=freq_rz, desired=gain_rz, weight=weigth_rz, 
      type='bandpass', maxiter=25, fs=fs)

sos_butter = signal.iirdesign(
    wp=fc, ws=fst, gpass=ripple_db_but, gstop=att_db_but,
    analog=False, ftype='butter', output='sos', fs=fs
)

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

_, H_rz = signal.freqz(b_rz, worN=ww, fs=fs)
mag_rz_dB = 20 * np.log10(np.abs(H_rz) + 1e-300)
phase_rz = np.unwrap(np.angle(H_rz)) * 180 / np.pi

# ─────────────────────────────────────────────
#  POLOS Y CEROS
# ─────────────────────────────────────────────
z_win = np.roots(b_win)          # ceros
p_win = np.zeros(len(b_win) - 1)  # todos los polos en z = 0

z_ls = np.roots(b_ls)          # ceros
p_ls = np.zeros(len(b_ls) - 1)  # todos los polos en z = 0

z_rz = np.roots(b_rz)          # ceros
p_rz = np.zeros(len(b_rz) - 1)  # todos los polos en z = 0

# ─────────────────────────────────────────────
#  FIGURA 1 — MÓDULO
# ─────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 4))

ax1.plot(ww, mag_win_dB, 'b', linewidth=1.5, label='Metodo de Ventana')
ax1.plot(ww, mag_ls_dB, 'r', linewidth=1.5, label='Cuadrados Minimos')
ax1.plot(ww, mag_rz_dB, 'g', linewidth=1.5, label='Parks-Mc Clellan-Remez')

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
ax2.plot(ww, phase_rz, 'g', linewidth=1.5, label='Parks-Mc Clellan-Remez')

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
    ('Parks-Mc Clellan-Remez',      z_rz, p_rz, 'g')
]

fig3, axes = plt.subplots(1, 3, figsize=(14, 5))

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

# ─────────────────────────────────────────────
#  FILTRADO DEL ECG IIR
# ─────────────────────────────────────────────
ecg_butter = signal.sosfiltfilt(sos_butter, ecg_one_lead)

ecg_win  = signal.lfilter(b_win, 1,   ecg_one_lead)
ecg_win  = signal.lfilter(b_win, 1,   ecg_win)
ecg_win = np.concatenate((ecg_win[Ntaps_win:], np.zeros(Ntaps_win)))
ecg_win = -ecg_win

ecg_ls  = signal.lfilter(b_ls, 1,   ecg_one_lead)
ecg_ls  =  np.concatenate((ecg_ls[Ntaps_ls//2:], np.zeros(Ntaps_ls//2)))

ecg_rz = signal.lfilter(b_rz, 1,   ecg_one_lead)
ecg_rz  =  np.concatenate((ecg_rz[Ntaps_rz//2:], np.zeros(Ntaps_rz//2)))

# ─────────────────────────────────────────────
#  REGIONES DE INTERÉS — sin ruido (señal cruda)
# ─────────────────────────────────────────────
regs_ruido = (
    [4000, 5500],
    [10000, 11000],
)
    
for ii in regs_ruido:
    zoom = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    plt.figure(figsize=(11, 3))
    plt.plot(zoom, ecg_one_lead[zoom], 'k',   linewidth=1,   label='ECG crudo',        alpha=0.5)
    plt.plot(zoom, ecg_butter[zoom],   'b',   linewidth=1.5, label='Butterworth')
    plt.plot(zoom, ecg_win[zoom],    'r', linewidth=1.5, label='Window')
    plt.plot(zoom, ecg_ls[zoom],    'g',   linewidth=1.5, label='Least Squares')
    plt.plot(zoom, ecg_rz[zoom],    'y',   linewidth=1.5, label='Parks-Mc Clellan-Remez')
    plt.title(f'ECG filtrado — muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.gca().legend(fontsize=8)
    plt.gca().set_yticks(())
    plt.tight_layout()

# ─────────────────────────────────────────────
#  REGIONES DE INTERÉS — señal filtrada
# ─────────────────────────────────────────────
regs_filtradas = (
    np.array([5,  5.2 ]) * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs,
)

for ii in regs_filtradas:
    zoom = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    plt.figure(figsize=(11, 3))
    plt.plot(zoom, ecg_one_lead[zoom], 'k',   linewidth=1,   label='ECG crudo',        alpha=0.5)
    plt.plot(zoom, ecg_butter[zoom],   'b',   linewidth=1.5, label='Butterworth')
    plt.plot(zoom, ecg_win[zoom],    'r', linewidth=1.5, label='Window')
    plt.plot(zoom, ecg_ls[zoom],    'g',   linewidth=1.5, label='Least Squares')
    plt.plot(zoom, ecg_rz[zoom],    'y',   linewidth=1.5, label='Parks-Mc Clellan-Remez')
    plt.title(f'ECG filtrado — muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.gca().legend(fontsize=8)
    plt.gca().set_yticks(())
    plt.tight_layout()

plt.show()
