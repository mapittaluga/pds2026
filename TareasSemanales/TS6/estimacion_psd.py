#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

#%%
def blackman_tukey(x, fs, M = None):    
    
    N = len(x)

    if M is None:
        M = N//5
        
    r_len = 2*M-1

    rxx = np.correlate(x, x, mode='full') / N

    # Recorto a lo afectado por la ventana
    mid = len(rxx) // 2
    rxx = rxx[mid-(M-1):mid+M]

    #Blackman
    w = sig.windows.blackman(r_len)/ np.mean(sig.windows.blackman(r_len))
    rxx_w = rxx * w

    # PSD
    Px = np.abs(np.fft.fft(rxx_w))
    
    f = np.arange(start = 0, stop = fs, step = fs/r_len)

    return f, Px;

def bandWidth(psd, deltaf, tipo = 'passband', cota = 0.95):
    cumP = np.cumsum(psd)*deltaf
    Pt = cumP[-1]
    cumP /= Pt
    match tipo:
        case 'passband':
            idx_fo  = np.argmin(np.abs(cumP - 0.5))
            idx_fcs = np.argmin(np.abs(cumP - (1 + cota)/2))
            idx_fci = np.argmin(np.abs(cumP - (1 - cota)/2))
        case 'lowpass':
            idx_fo  = 0
            idx_fcs = np.argmin(np.abs(cumP - cota))
            idx_fci = 0
            
    fo  = deltaf*idx_fo
    fcs = deltaf*idx_fcs
    fci = deltaf*idx_fci
    
    B = fcs - fci 
    return B, fo, fcs, fci;
#%%
##################
# Lectura de ECG #
##################
fs_ecg = 1000 # Hz
##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')
ecg_one_lead = ecg_one_lead - np.mean(ecg_one_lead)

K = [6, 6, 8, 10]
N = len(ecg_one_lead)

f1, Pecg_w1 = sig.welch(ecg_one_lead, fs_ecg, nperseg=N/K[0], return_onesided = True)
# f2, Pecg_w2 = sig.welch(ecg_one_lead, fs_ecg, nperseg=N/K[1], return_onesided = True)
# f3, Pecg_w3 = sig.welch(ecg_one_lead, fs_ecg, nperseg=N/K[2], return_onesided = True)
# f4, Pecg_w4 = sig.welch(ecg_one_lead, fs_ecg, nperseg=N/K[3], return_onesided = True)

deltaf = f1[1] - f1[0] 
cota = 0.99

B, fo, fcs, fci = bandWidth(Pecg_w1, deltaf, tipo = 'lowpass',  cota = cota)
print(f"Ancho de Banda Welch({cota*100}%): {B:.2f}") 

plt.figure(figsize=(10,5))
plt.plot(f1,10*np.log10(Pecg_w1), label=f'PSD (k = {K[0]})')
plt.axvline(fo, color='red', linestyle='--', label='fo')
plt.axvline(fcs, color='green', linestyle='--', label='fcs')
plt.axvline(fci, color='blue', linestyle='--', label='fci')
# plt.plot(f2,10*np.log10(Pecg_w2), label=f'PSD (k = {K[1]})')
# plt.plot(f3,10*np.log10(Pecg_w3), label=f'PSD (k = {K[2]})')
# plt.plot(f4,10*np.log10(Pecg_w4), label=f'PSD (k = {K[3]})')

plt.xlim([0, fs_ecg//2])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD con Welch ECG')
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.xlabel('n')
plt.ylabel('ECG')
plt.title('ECG sin ruido')
plt.plot(ecg_one_lead)

#%% Con Blackman-Tuckey

#M = [N//2, N//3, N//5, N//10]
M = N//5
f1b, Pecg_w1b = blackman_tukey(ecg_one_lead,  fs = fs_ecg, M = M)

deltaf = f1b[1] - f1b[0] 
cota = 0.99

B, fo, fcs, fci = bandWidth(Pecg_w1b[:M], deltaf, tipo = 'lowpass',  cota = cota)
print(f"Ancho de Banda BT({cota*100}%): {B:.2f}") 

plt.figure(figsize=(10,5))
plt.plot(f1b,10*np.log10(Pecg_w1b), label=f'PSD (M = {M})')
plt.axvline(fo, color='red', linestyle='--', label='fo')
plt.axvline(fcs, color='green', linestyle='--', label='fcs')
plt.axvline(fci, color='blue', linestyle='--', label='fci')


plt.xlim([0, fs_ecg//2])
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.title('PSD con BT ECG')
plt.legend()
plt.grid()
plt.show()


# #%%

# ####################################
# # Lectura de pletismografía (PPG)  #
# ####################################

# fs_ppg = 400 # Hz

# ##################
# ## PPG con ruido
# ##################

# # # Cargar el archivo CSV como un array de NumPy
# # ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe


# ##################
# ## PPG sin ruido
# ##################

# ppg = np.load('ppg_sin_ruido.npy')

# plt.figure()
# plt.plot(ppg)


# #%%

# ####################
# # Lectura de audio #
# ####################

# # Cargar el archivo CSV como un array de NumPy
# fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# # fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# # fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# plt.figure()
# plt.plot(wav_data)

# # si quieren oirlo, tienen que tener el siguiente módulo instalado
# # pip install sounddevice
# # import sounddevice as sd
# # sd.play(wav_data, fs_audio)
# #%%