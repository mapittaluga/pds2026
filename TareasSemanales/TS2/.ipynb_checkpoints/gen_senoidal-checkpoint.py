#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal as sig

# ============================================================
# Generador de señales — TS1 PDS
# ============================================================

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

def noise(nn=1000, fs=1000, mean=0, std=1):
    
    tt  = np.arange(start = 0, step = 1/fs, stop = nn/fs)
    ww  = np.random.normal(loc=mean, scale=std, size= nn)
    return tt, ww


def signal_n(wave='sine', vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000, duty=0.5, snr=10):
   
    tt, xx = signal(wave=wave, vmax=vmax, dc=dc, ff=ff, ph=ph, nn=nn, fs=fs, duty=duty)

    # Potencia de la señal (ignoramos la potencia de DC)
    p_signal = np.mean((xx - dc) ** 2)

    # P_ruido = P_señal / 10^(SNR/10)
    p_noise  = p_signal / (10 ** (snr / 10))
    # P_ruido = var
    std_noise  = np.sqrt(p_noise)

    _, ww  = noise(nn=nn, fs=fs, mean=0, std=std_noise)
    xx_n   = xx + ww

    return tt, xx_n
#%% Visualizacion

types  = ['sine', 'square', 'triangle', 'sawtooth']
titles = ['Senoidal', 'Cuadrada', 'Triangular', 'Diente de sierra']

for typ, title in zip(types, titles):
    tt, xx = signal_n(vmax=1, dc=0, ff=3, ph=0, nn=N, fs=fs, wave=typ)

    plt.figure(figsize=(10, 3))
    plt.plot(tt, xx, linewidth=1)
    plt.title(title, fontsize=11)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (V)")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.tight_layout()
    plt.show()
#%%