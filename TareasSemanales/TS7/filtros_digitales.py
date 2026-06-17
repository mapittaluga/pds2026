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
#  PARÁMETROS DEL FILTRO
# ─────────────────────────────────────────────
fc        = [1, 35]    # frecuencias de paso (Hz)
fst       = [0.1, 45] # frecuencias de stop (Hz)
ripple_db = 0.5
att_db    = 20

# ─────────────────────────────────────────────
#  DISEÑO DE LOS FILTROS
# ─────────────────────────────────────────────
sos_butter = signal.iirdesign(
    wp=fc, ws=fst, gpass=ripple_db, gstop=att_db,
    analog=False, ftype='butter', output='sos', fs=fs
)

sos_cheby = signal.iirdesign(
    wp=fc, ws=fst, gpass=ripple_db, gstop=att_db,
    analog=False, ftype='cheby1', output='sos', fs=fs
)

sos_cauer = signal.iirdesign(
    wp=fc, ws=fst, gpass=ripple_db, gstop=att_db,
    analog=False, ftype='ellip', output='sos', fs=fs
)

# ─────────────────────────────────────────────
#  GRILLA DE FRECUENCIAS
# ─────────────────────────────────────────────
ww = np.concatenate([
    np.logspace(np.log10(0.01), np.log10(2),   250),
    np.linspace(2.1,            34,             250),
    np.logspace(np.log10(34.1), np.log10(52),  250),
    np.linspace(52.1,           fs / 2,         250),
])

# ─────────────────────────────────────────────
#  RESPUESTA EN FRECUENCIA
# ─────────────────────────────────────────────
_, H_butter = signal.freqz_sos(sos_butter, worN=ww, fs=fs)
mag_butter_dB = 20 * np.log10(np.abs(H_butter) + 1e-300)
phase_butter  = np.unwrap(np.angle(H_butter)) * 180 / np.pi

_, H_cheby = signal.freqz_sos(sos_cheby, worN=ww, fs=fs)
mag_cheby_dB = 20 * np.log10(np.abs(H_cheby) + 1e-300)
phase_cheby  = np.unwrap(np.angle(H_cheby)) * 180 / np.pi

_, H_cauer = signal.freqz_sos(sos_cauer, worN=ww, fs=fs)
mag_cauer_dB = 20 * np.log10(np.abs(H_cauer) + 1e-300)
phase_cauer  = np.unwrap(np.angle(H_cauer)) * 180 / np.pi

# ─────────────────────────────────────────────
#  POLOS Y CEROS
# ─────────────────────────────────────────────
z_but, p_but, _ = signal.sos2zpk(sos_butter)
z_cheb, p_cheb, _ = signal.sos2zpk(sos_cheby)
z_cau,  p_cau,  _ = signal.sos2zpk(sos_cauer)

# ─────────────────────────────────────────────
#  FIGURA 1 — MÓDULO 
# ─────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(11, 4))

ax1.plot(ww, mag_butter_dB, 'b',        linewidth=1.5, label='Butterworth')
ax1.plot(ww, mag_cheby_dB,  'r--',      linewidth=1.5, label='Chebyshev I')
ax1.plot(ww, mag_cauer_dB,  color='g',  linewidth=1.5, label='Cauer (Elíptico)')

axes_hdl = plt.gca()

plot_plantilla(filter_type = 'bandpass', fpass = fc, fstop =  fst, ripple = ripple_db, attenuation = att_db, fs = fs)
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

ax2.plot(ww, phase_butter, 'b',       linewidth=1.5, label='Butterworth')
ax2.plot(ww, phase_cheby,  'r--',     linewidth=1.5, label='Chebyshev I')
ax2.plot(ww, phase_cauer,  color='g', linewidth=1.5, label='Cauer (Elíptico)')


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
    ('Butterworth',      z_but,  p_but,  'b'),
    ('Chebyshev I',      z_cheb, p_cheb, 'r'),
    ('Cauer (Elíptico)', z_cau,  p_cau,  'g'),
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
ecg_cheby  = signal.sosfiltfilt(sos_cheby,  ecg_one_lead)
ecg_cauer  = signal.sosfiltfilt(sos_cauer,  ecg_one_lead)

# ─────────────────────────────────────────────
#  REGIONES DE INTERÉS — con ruido (señal cruda)
# ─────────────────────────────────────────────
regs_ruido = (
    [4000, 5500],
    [10000, 11000],
)

for ii in regs_ruido:
    zoom = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
    plt.figure(figsize=(11, 3))
    plt.plot(zoom, ecg_one_lead[zoom], 'k', label='ECG crudo', linewidth=1.5)
    plt.title(f'ECG crudo — muestras {ii[0]} a {ii[1]}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.gca().legend()
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
    plt.plot(zoom, ecg_cheby[zoom],    'r--', linewidth=1.5, label='Chebyshev I')
    plt.plot(zoom, ecg_cauer[zoom],    'g',   linewidth=1.5, label='Cauer (Elíptico)')
    plt.title(f'ECG filtrado — muestras {int(ii[0])} a {int(ii[1])}')
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    plt.gca().legend(fontsize=8)
    plt.gca().set_yticks(())
    plt.tight_layout()

plt.show()