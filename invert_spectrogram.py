import pathlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from thunderfish.fakefish import chirps, wavefish_eods

# simulate a recording
chps1, amp1 = chirps(eodf=600.0, samplerate=44100.0, duration=1.0, chirp_freq=5.0,
           chirp_size=100.0, chirp_width=0.01, chirp_kurtosis=1.0, chirp_contrast=0.05)

chps2, amp2 = chirps(eodf=500.0, samplerate=44100.0, duration=1.0, chirp_freq=5.0,
              chirp_size=100.0, chirp_width=0.01, chirp_kurtosis=1.0, chirp_contrast=0.05)

eod1 = wavefish_eods(fish='Alepto', frequency=chps1, samplerate=44100.0,
                  duration=1.0, phase0=0.0, noise_std=0.05)

eod2 = wavefish_eods(fish='Alepto', frequency=chps2, samplerate=44100.0,
                  duration=1.0, phase0=0.0, noise_std=0.05)

eod1 *= amp1
eod2 *= amp2
eod = (eod1 + eod2) / 2

time = np.arange(0, 1, 1/44100)

# compute the spectrogram with pytorch 
trans = T.Spectrogram(
    n_fft=1024,
    hop_length=1,  # One FFT per one video frame
    normalized=True,
    power=1,
)

eod = torch.from_numpy(eod)
spec = trans(eod)
plt.imshow(spec.log2().numpy(), aspect='auto', origin='lower')
plt.show()
