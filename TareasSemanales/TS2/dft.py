# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import time

#%% Condiciones de muestreo

N    = 1000 
fs   = 1000 

#%% Funcion Generador de Señales
def signal( wave ='sin', vmax=1, dc=0, ff=1, ph=0, nn=N, fs=fs, duty=0.5 ):
    
    tt = np.arange(start = 0, step = 1/fs, stop = nn/fs)
    
    match wave:
        case 'sine':
            xx = vmax * np.sin( 2 * np.pi * ff * tt + ph) + dc
        case 'square':
            xx = vmax * sig.square( 2 * np.pi * ff * tt + ph, duty = duty) + dc
        case 'triangle':
            xx = vmax * sig.sawtooth( 2 * np.pi * ff * tt + ph, width=0.5) + dc
        case 'sawtooth':
            xx = vmax * sig.sawtooth( 2 * np.pi * ff * tt + ph) + dc
        case _:
            raise ValueError(f"Tipo '{wave}' no reconocido.")
    
    return tt, xx

#%% Funcion DFT

def dft(x):
    N  = len(x)
    n  = np.arange(N)
    k  = n.reshape((N, 1))
    e  = np.exp(-2j * np.pi * k * n / N)
    Xk = np.dot(e, x)
    return Xk

#%% Señal Muestreada

ff   = 100
vmax = 1
t, x = signal(wave='sine', vmax=vmax, dc=0, ff=ff, ph=0, nn=N, fs=fs)
xt = x.reshape((N, 1)) # Mi DFT usa (N,1)

#%%  Comparación de tiempos con FFT
t0     = time.perf_counter()
Xk_dft = dft(xt)
t_dft  = time.perf_counter() - t0

t0     = time.perf_counter()
Xk_fft = np.fft.fft(x)
t_fft  = time.perf_counter() - t0

print(f"DFT : {t_dft*1e3:.3f} ms")
print(f"FFT : {t_fft*1e3:.3f} ms")
print(f"Speedup: {t_dft/t_fft:.1f}x")

#%%  Componentes
Xk = Xk_dft
k  = np.arange(N) # Tiene simetria con respecto a N/2
f  = k * fs / N
               
modulo     = np.abs(Xk[k])  / N  # Normalizamos
fase       = np.angle(Xk[k])
parte_real = np.real(Xk)
parte_imag = np.imag(Xk)

#%% Fig. 1: Señal Temporal
fig1, ax1 = plt.subplots(figsize=(12, 4))
fig1.suptitle(f"Señal ff = {ff} Hz  N = {N}", fontsize=13)

ax1.plot(t, x, color='steelblue', linewidth=0.8)
ax1.set_xlabel("Tiempo [s]")
ax1.set_ylabel("Amplitud")
ax1.set_xlim(0, len(x)/fs)
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.subplots_adjust(top=0.88)

#%% Fig. 2: Modulo y Fase
fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(12, 7))
fig2.suptitle("Espectro de la señal", fontsize=13)

ax2.stem(f, modulo, linefmt='steelblue', markerfmt='C0.', basefmt='k')
ax2.set_xlabel("Frecuencia [Hz]")
ax2.set_ylabel("|X(k)| normalizado")
ax2.set_title("Módulo")
ax2.set_xlim(0, fs/2)
ax2.grid(True, alpha=0.3)

# Elimino los datos de fase que no importan
umbral    = modulo.max() * 1e-3 
fase_plot = np.where(modulo > umbral, fase, np.nan)

ax3.stem(f, np.degrees(fase_plot), linefmt='tomato', markerfmt='C3.', basefmt='k')
ax3.set_xlabel("Frecuencia [Hz]")
ax3.set_ylabel("Fase [°]")
ax3.set_title("Fase")
ax3.set_xlim(0, fs/2)
ax3.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.subplots_adjust(top=0.88)


#%% Fig. 2: Parte Real e Imaginaria
fig3, (ax4, ax5) = plt.subplots(2, 1, figsize=(12, 6))
fig3.suptitle("Parte real e imaginaria de la DFT", fontsize=13)

ax4.plot(f, parte_real, color='steelblue', linewidth=0.6)
ax4.set_xlabel("Frecuencia [Hz]")
ax4.set_ylabel("Re{X(k)}")
ax4.set_title("Parte real")
ax4.set_xlim(0, fs/2)
ax4.grid(True, alpha=0.3)

ax5.plot(f, parte_imag, color='tomato', linewidth=0.6)
ax5.set_xlabel("Frecuencia [Hz]")
ax5.set_ylabel("Im{X(k)}")
ax5.set_title("Parte imaginaria")
ax5.set_xlim(0, fs/2)
ax5.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.subplots_adjust(top=0.88)
plt.show()