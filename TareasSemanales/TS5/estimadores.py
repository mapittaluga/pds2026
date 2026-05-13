#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.stats import gaussian_kde

#%% Condiciones de muestreo

N      = 1000 
fs     = 2*np.pi  
deltaf = fs/N
M      = 200 #Realizaciones 

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

def nnoise(nn=1000, fs=1000, mean=0, std=1):
    
    tt  = np.arange(start = 0, step = 1/fs, stop = nn/fs)
    ww  = np.random.normal(loc=mean, scale=std, size= nn)
    return tt, ww

def signal_n(wave='sine', vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000, duty=0.5, snr=10.0):
   
    tt, xx = signal(wave=wave, vmax=vmax, dc=dc, ff=ff, ph=ph, nn=nn, fs=fs, duty=duty)

    # Potencia de la señal (ignoramos la potencia de DC)
    p_signal = np.mean((xx - dc) ** 2)
    

    # P_ruido = P_señal / 10^(SNR/10)
    p_noise  = p_signal / (10 ** (snr / 10))
    # P_ruido = var
    std_noise  = np.sqrt(p_noise)

    _, ww  = nnoise(nn=nn, fs=fs, mean=0, std=std_noise)
    xx_n   = xx + ww

    return tt, xx_n

#%% Señal muestreada
vmax = np.sqrt(2)
fo =  np.pi/2 
fr = np.random.uniform(-2,2,200)
ff = fo + fr * deltaf
ff = ff.reshape((M, 1))
SNR = 10

_, x = signal_n(wave='sine', vmax=vmax, dc=0, ff=ff, ph=0, nn=N, fs=fs, snr=SNR)
#%% Ventaneo

x_rt = x #Rectangular
x_ft = x * sig.windows.flattop(N)/ np.mean(sig.windows.flattop(N))
x_bm = x * sig.windows.blackmanharris(N)/ np.mean(sig.windows.blackmanharris(N))
x_hm = x * sig.windows.hamming(N)/ np.mean(sig.windows.hamming(N))


#%% FFT
X_rt = np.fft.fft(x_rt, axis = 1)/N
X_ft = np.fft.fft(x_ft, axis = 1)/N
X_bm = np.fft.fft(x_bm, axis = 1)/N
X_hm = np.fft.fft(x_hm, axis = 1)/N

f  = np.arange(start = 0, stop = fs, step = deltaf)

#%% Estimador de Amplitud
a0 = vmax

# a_rt = 2 * np.abs(X_rt[:, N//4])
# a_ft = 2 * np.abs(X_ft[:, N//4])
# a_bm = 2 * np.abs(X_bm[:, N//4])
# a_hm = 2 * np.abs(X_hm[:, N//4])

a_rt = 2 * np.sqrt(np.sum(np.abs(X_rt[:, N//4 - 2 : N//4 + 2 + 1])**2, axis=1))
a_ft = 2 * np.sqrt(np.sum(np.abs(X_ft[:, N//4 - 2 : N//4 + 2 + 1])**2, axis=1))
a_bm = 2 * np.sqrt(np.sum(np.abs(X_bm[:, N//4 - 2 : N//4 + 2 + 1])**2, axis=1))
a_hm = 2 * np.sqrt(np.sum(np.abs(X_hm[:, N//4 - 2 : N//4 + 2 + 1])**2, axis=1))

# Sesgos
Sa_rt = np.mean(a_rt) - a0
Sa_ft = np.mean(a_ft) - a0
Sa_bm = np.mean(a_bm) - a0
Sa_hm = np.mean(a_hm) - a0

# Varianza
Va_rt = np.var(a_rt)
Va_ft = np.var(a_ft)
Va_bm = np.var(a_bm)
Va_hm = np.var(a_hm)

print("\nResultados Estimadores de Amplitud:\n")
print(f"{'Ventana':<15}{'Sesgo':>15}{'Varianza':>15}")
print("-"*45)
print(f"{'Rectangular':<15}{Sa_rt:>15.6e}{Va_rt:>15.6e}")
print(f"{'Flattop':<15}{Sa_ft:>15.6e}{Va_ft:>15.6e}")
print(f"{'Blackman-Harris':<15}{Sa_bm:>15.6e}{Va_bm:>15.6e}")
print(f"{'Hamming':<15}{Sa_hm:>15.6e}{Va_hm:>15.6e}")

#%% Grafico de Densidad de Probabilidad
plt.figure(figsize=(10,5))

# Rango común
xmin = min(a_rt.min(), a_ft.min(), a_bm.min(), a_hm.min())
xmax = max(a_rt.max(), a_ft.max(), a_bm.max(), a_hm.max())
x_vals = np.linspace(xmin, xmax, 500)

# KDE (curvas suaves)
kde_rt = gaussian_kde(a_rt)
kde_ft = gaussian_kde(a_ft)
kde_bm = gaussian_kde(a_bm)
kde_hm = gaussian_kde(a_hm)

plt.plot(x_vals, kde_rt(x_vals), label='Rectangular')
plt.plot(x_vals, kde_ft(x_vals), label='Flattop')
plt.plot(x_vals, kde_bm(x_vals), label='Blackman-Harris')
plt.plot(x_vals, kde_hm(x_vals), label='Hamming')

# Líneas verticales
plt.axvline(a0, color='k', linestyle='--', linewidth=2, label='a0')
plt.axvline(np.mean(a_rt), linestyle=':', color='C0')
plt.axvline(np.mean(a_ft), linestyle=':', color='C1')
plt.axvline(np.mean(a_bm), linestyle=':', color='C2')
plt.axvline(np.mean(a_hm), linestyle=':', color='C3')

plt.xlabel('Amplitud estimada')
plt.ylabel('Densidad')
plt.title('Distribución de estimadores de Amplitud')
plt.legend()
plt.grid()

#%% Estimador de Frecuencia
X_rt_u = np.abs(X_rt[:, :N//2])
X_ft_u = np.abs(X_ft[:, :N//2])
X_bm_u = np.abs(X_bm[:, :N//2])
X_hm_u = np.abs(X_hm[:, :N//2])

fmax_rt = np.argmax(np.abs(X_rt_u), axis=1)*deltaf
fmax_ft = np.argmax(np.abs(X_ft_u), axis=1)*deltaf
fmax_bm = np.argmax(np.abs(X_bm_u), axis=1)*deltaf
fmax_hm = np.argmax(np.abs(X_hm_u), axis=1)*deltaf

# Sesgos
Sf_rt = np.mean(fmax_rt) - fo
Sf_ft = np.mean(fmax_ft) - fo
Sf_bm = np.mean(fmax_bm) - fo
Sf_hm = np.mean(fmax_hm) - fo

# Varianza
Vf_rt = np.var(fmax_rt)
Vf_ft = np.var(fmax_ft)
Vf_bm = np.var(fmax_bm)
Vf_hm = np.var(fmax_hm)

print("\nResultados Estimadores de Frecuencia:\n")
print(f"{'Ventana':<15}{'Sesgo':>15}{'Varianza':>15}")
print("-"*45)
print(f"{'Rectangular':<15}{Sf_rt:>15.6e}{Vf_rt:>15.6e}")
print(f"{'Flattop':<15}{Sf_ft:>15.6e}{Vf_ft:>15.6e}")
print(f"{'Blackman-Harris':<15}{Sf_bm:>15.6e}{Vf_bm:>15.6e}")
print(f"{'Hamming':<15}{Sf_hm:>15.6e}{Vf_hm:>15.6e}")

#%% Grafico de Distribución (Frecuencia) - Subplots

fig, axs = plt.subplots(2, 2, figsize=(12,8), sharex=True, sharey=True)

# --- Rectangular ---
axs[0,0].hist(fmax_rt, bins=20, density=True)
axs[0,0].axvline(fo, color='k', linestyle='--', linewidth=2)
axs[0,0].set_title('Rectangular')
axs[0,0].grid()

# --- Flattop ---
axs[0,1].hist(fmax_ft, bins=20, density=True)
axs[0,1].axvline(fo, color='k', linestyle='--', linewidth=2)
axs[0,1].set_title('Flattop')
axs[0,1].grid()

# --- Blackman-Harris ---
axs[1,0].hist(fmax_bm, bins=20, density=True)
axs[1,0].axvline(fo, color='k', linestyle='--', linewidth=2)
axs[1,0].set_title('Blackman-Harris')
axs[1,0].grid()

# --- Hamming ---
axs[1,1].hist(fmax_hm, bins=20, density=True)
axs[1,1].axvline(fo, color='k', linestyle='--', linewidth=2)
axs[1,1].set_title('Hamming')
axs[1,1].grid()

# Zoom (CLAVE)
for ax in axs.flat:
    ax.set_xlim(fo - 5*deltaf, fo + 5*deltaf)
    ax.set_xlabel('Frecuencia estimada')
    ax.set_ylabel('Densidad')

plt.suptitle('Distribución de estimadores de Frecuencia')
plt.tight_layout()
plt.show()