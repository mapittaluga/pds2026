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

#%% Visualizacion

types  = ['sine', 'square', 'triangle', 'sawtooth']
titles = ['Senoidal', 'Cuadrada', 'Triangular', 'Diente de sierra']

for typ, title in zip(types, titles):
    tt, xx = signal(vmax=1, dc=0, ff=3, ph=0, nn=N, fs=fs, wave=typ)

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