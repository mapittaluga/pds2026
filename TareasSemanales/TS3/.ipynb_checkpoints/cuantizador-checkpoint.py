#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
from scipy import signal as sig

#%% Condiciones de muestreo

N      = 1000 
fs     = N 
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
#%% Funcion Cuantizador
def cuantizador(sR, B = 8, VFS = 1, adc = 'bipolar'):
    q = VFS / (2**B)
    sQ = q * np.round(sR / q)
    match adc:
        case 'unipolar':
            sQ = np.clip(sQ, 0, q *((2**B)-1))
        case 'bipolar':
            sQ = np.clip(sQ, -q *(2**(B-1)), q *(2**(B-1)-1))
    return sQ

#%% Señal muestreada
vmax = np.sqrt(2)
ff = 10*deltaf
VFS = 4
B = 8
SNR = 1000

t, sR = signal(wave='sine', vmax=vmax, ff=ff, nn=N, fs=fs)
p_signal = np.mean(sR**2)


p_noise  = p_signal / (10 ** (SNR / 10))
std_noise  = np.sqrt(p_noise)
_, sN = nnoise(nn = N, fs=fs, mean=0, std=std_noise)

p_noise = np.mean(sN**2)

sRN = sR + sN

sQ = cuantizador(sR = sRN, B = B, VFS = VFS, adc = 'bipolar')
#%% Grafico Temporal

plt.figure(figsize=(10,5))

# Señal original
plt.plot(t, sRN, label='Señal original', linewidth=2)

# Señal cuantizada 
plt.plot(t , sQ, drawstyle = 'steps-post')

plt.title(f"Cuantización ADC (B={B}, VFS={VFS} V)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.xlim(0, 10/ff)
plt.grid()
plt.legend()

plt.show()

#%% Ruido de cuantizacion
e = sQ - sRN
ree = np.correlate(e, e, mode='full') / N
p_quant = ree[N-1]

sNQ = sN + e
p_nq = np.mean(sNQ**2)

q = VFS / (2**B)

print("=" * 45)
print("       CARACTERIZACIÓN DEL RUIDO")
print("=" * 45)
print(f"  Media:                    {np.mean(e):.4e}")
print(f"  Varianza:                 {np.var(e):.4e}")
print(f"  Potencia medida:          {p_quant:.4e}")
print(f"  Potencia teórica (q²/12): {q**2/12:.4e}")
print(f"  SNR:                      {10*np.log10(p_signal/p_nq):.2f} dB")
print("=" * 45)
lags = np.arange(-(N-1), N) 

plt.figure(figsize=(10,5))

# Señal original
plt.plot(lags, ree, label='Ree', linewidth=1.5)

plt.title("Autocorrelacion del Error de Cuantizacion")
plt.xlabel("Lag [s]")
plt.ylabel("Voltaje [V]")
plt.grid()
plt.legend()
plt.show()

#%% PSD
Xk = np.fft.fft(sRN)
See = np.fft.fft(ree)
f     = np.arange(start = 0, stop = fs, step = deltaf)
f_see = np.arange(start = 0, stop = fs, step = deltaf/2)
f_see = f_see[:-1]
          
plt.figure(figsize=(10,6))

plt.plot(f, 10*np.log10(2*((np.abs(Xk)/N)**2) + 1e-12), label='PSD Señal')
plt.plot(f_see, 10*np.log10(2*np.abs(See) + 1e-12), label='PSD Ruido cuantización')

plt.title(f"PSD - B={B}")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("PSD [dB]")
plt.xlim(0,fs/2)
plt.grid()
plt.legend()

plt.show()
    

