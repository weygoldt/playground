import pathlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from IPython import embed

from thunderfish.fakefish import chirps, wavefish_eods

# simulate a recording
# chps1, amp1 = chirps(eodf=700.0, samplerate=44100.0, duration=1.0, chirp_freq=0,
#            chirp_size=100.0, chirp_width=0.01, chirp_kurtosis=1.0, chirp_contrast=0.05)

# chps2, amp2 = chirps(eodf=500.0, samplerate=44100.0, duration=1.0, chirp_freq=5.0,
#               chirp_size=100.0, chirp_width=0.01, chirp_kurtosis=1.0, chirp_contrast=0.05)

# eod1 = wavefish_eods(fish='Alepto', frequency=700, samplerate=44100.0,
#                   duration=1.0, phase0=0.0, noise_std=0.0)

# eod2 = wavefish_eods(fish='Alepto', frequency=chps2, samplerate=44100.0,
#                   duration=1.0, phase0=0.0, noise_std=0.0)

# # eod1 *= amp1
# eod2 *= amp2
# eod = (eod1 + eod2) / 2

# time = np.arange(0, 1, 1/44100)

datapath = pathlib.Path('../chirpdetector-cnn/real_data/raw.npy')
snippet = np.load(datapath)
mean_amps = np.mean(np.abs(snippet), axis=0)
eoi = np.argmax(mean_amps)
eod = snippet[:, eoi]

time = np.arange(0, len(eod)/20000, 1/20000)
toi = time[(time > 68) & (time < 70)]
eod = eod[(time > 68) & (time < 70)]

# compute the spectrogram with pytorch 
trans = T.Spectrogram(
    n_fft=2048,
    hop_length=1,  # One FFT per one video frame
    normalized=True,
    power=2,
)

invert = T.GriffinLim(
    n_fft=2048, 
    n_iter=32, 
    hop_length=1, 
    win_length=2048, 
    power=2
)

eod = torch.from_numpy(eod)
spec = trans(eod)

# mask all upper frequencies
spec_masked = spec.clone()
spec_masked[:76, :] = 0
spec_masked[79:, :] = 0

eod_inv = invert(spec_masked)
inv_spec = trans(eod_inv)

fig, axs = plt.subplots(5, 1)
axs[0].plot(toi, eod.numpy())
axs[1].imshow(librosa.power_to_db(spec.numpy()), aspect='auto', origin='lower')
axs[2].imshow(librosa.power_to_db(spec_masked.numpy()), aspect='auto', origin='lower')
axs[3].plot(toi, eod_inv.numpy())
axs[4].imshow(librosa.power_to_db(inv_spec.numpy()),  aspect='auto', origin='lower')
plt.show()
