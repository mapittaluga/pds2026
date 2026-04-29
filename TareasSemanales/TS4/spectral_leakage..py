#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig

#%% Condiciones de muestreo

N    = 1000 
fs   = 1000 
deltaf = fs/N

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

#%% Señal muestreada
vmax = np.sqrt(2)
k = [N/4, N/4 + 0.25, N/4 + 0.5]
ff1 = k[0]*deltaf
ff2 = k[1]*deltaf
ff3 = k[2]*deltaf

t, x1 = signal(wave='sine', vmax=vmax, dc=0, ff=ff1, ph=0, nn=N, fs=fs)
_, x2 = signal(wave='sine', vmax=vmax, dc=0, ff=ff2, ph=0, nn=N, fs=fs)
_, x3 = signal(wave='sine', vmax=vmax, dc=0, ff=ff3, ph=0, nn=N, fs=fs)

#%% Grafico Temporal
plt.figure(figsize=(10,5))
plt.plot(t, x1, label=f'f1={ff1:.2f} Hz')
plt.plot(t, x2, label=f'f2={ff2:.2f} Hz')
plt.plot(t, x3, label=f'f3={ff3:.2f} Hz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señales en el dominio temporal')
plt.legend()
plt.grid()
plt.show()

#%% FFT
X1 = np.fft.fft(x1)/N
X2 = np.fft.fft(x2)/N
X3 = np.fft.fft(x3)/N

f  = np.arange(start = 0, stop = fs, step = deltaf)
#%% Potencia (Parseval)
P1 = np.mean(np.abs(x1)**2)
P2 = np.mean(np.abs(x2)**2)
P3 = np.mean(np.abs(x3)**2)

print("Potencias desde tiempo:")
print(f"P1 = {P1:.4f}")
print(f"P2 = {P2:.4f}")
print(f"P3 = {P3:.4f}")

# Potencia desde FFT (verificación)
P1f = np.sum(np.abs(X1)**2)
P2f = np.sum(np.abs(X2)**2)
P3f = np.sum(np.abs(X3)**2)

print("\nPotencias desde FFT:")
print(f"P1f = {P1f:.4f}")
print(f"P2f = {P2f:.4f}")
print(f"P3f = {P3f:.4f}")

#%% Grafico PSD
plt.figure(figsize=(10,5))
plt.plot(f,10*np.log10(2*(np.abs(X1))**2), '.', label='X1')
plt.plot(f,10*np.log10(2*(np.abs(X2))**2), '.', label='X2')
plt.plot(f,10*np.log10(2*(np.abs(X3))**2), '.', label='X3')
plt.xlim([0, fs/2])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X(f)|')
plt.title('FFT (magnitud)')
plt.legend()

plt.grid()
plt.show()
#%% Zero Padding
Np = 9*N
Nt = N + Np

x1_zp = np.pad(x1, (0, Np),'constant', constant_values=0)
x2_zp = np.pad(x2, (0, Np),'constant', constant_values=0)
x3_zp = np.pad(x3, (0, Np),'constant', constant_values=0)

X1_zp = np.fft.fft(x1_zp)/Nt
X2_zp = np.fft.fft(x2_zp)/Nt
X3_zp = np.fft.fft(x3_zp)/Nt

fz  = np.arange(Nt) * fs / Nt

plt.figure(figsize=(10,5))
plt.plot(fz,10*np.log10(2*(np.abs(X1_zp)**2)), '.', label='X1 ZP')
plt.plot(fz,10*np.log10(2*(np.abs(X2_zp)**2)), '.', label='X2 ZP')
plt.plot(fz,10*np.log10(2*(np.abs(X3_zp)**2)), '.', label='X3 ZP')

plt.xlim([0, fs/2])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD con Zero Padding')
plt.legend()
plt.grid()
plt.show()

