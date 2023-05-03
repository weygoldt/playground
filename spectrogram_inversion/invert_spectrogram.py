import pathlib

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chps2, amp2 = chirps(
    eodf=0,
    samplerate=44100.0,
    duration=1.0,
    chirp_freq=5.0,
    chirp_size=100.0,
    chirp_width=0.01,
    chirp_kurtosis=1.0,
    chirp_contrast=0.05,
)

bl1 = np.linspace(500, 600, 44100)
chps2 = bl1 + chps2
freqs1 = np.linspace(550, 500, 44100)

eod1 = wavefish_eods(
    fish="Alepto",
    frequency=freqs1,
    samplerate=44100.0,
    duration=1.0,
    phase0=0.0,
    noise_std=0.0,
)

eod2 = wavefish_eods(
    fish="Alepto",
    frequency=chps2,
    samplerate=44100.0,
    duration=1.0,
    phase0=0.0,
    noise_std=0.0,
)

# eod1 *= amp1
eod2 *= amp2
eod = (eod1 + eod2) / 2

time = np.arange(0, 1, 1 / 44100)

# datapath = pathlib.Path('../chirpdetector-cnn/real_data/raw.npy')
# snippet = np.load(datapath)
# mean_amps = np.mean(np.abs(snippet), axis=0)
# eoi = np.argmax(mean_amps)
# eod = snippet[:, eoi]

# time = np.arange(0, len(eod)/20000, 1/20000)
# toi = time[(time > 68) & (time < 70)]
# eod = eod[(time > 68) & (time < 70)]

# compute the spectrogram with pytorch
nfft = 4096
# hop_length = int(nfft / 100)
hop_length = 1
trans = T.Spectrogram(
    n_fft=nfft,
    hop_length=hop_length,
    normalized=True,
    power=2,
)

invert = T.GriffinLim(
    n_fft=nfft, n_iter=32, hop_length=hop_length, win_length=nfft, power=2
)

todb = T.AmplitudeToDB(stype="power", top_db=80)

eod = torch.from_numpy(eod).to(device)
spec = trans(eod)
spec_times = np.arange(0, spec.shape[1]) * hop_length / 44100
spec_freqs = np.arange(0, spec.shape[0]) * 44100 / nfft

# mask all upper frequencies

spec_masked = spec.clone()
for i, f in enumerate(bl1):
    spec_masked[:, i][spec_freqs > f + 10] = 0
    spec_masked[:, i][spec_freqs < f - 10] = 0

eod_inv = invert(spec_masked)
eod_inv_time = np.linspace(0, 1, len(eod_inv))
inv_spec = trans(eod_inv)

fig, axs = plt.subplots(5, 1, sharex=True)
axs[0].plot(time, eod.numpy())
axs[1].imshow(
    todb(spec),
    aspect="auto",
    origin="lower",
    extent=[spec_times[0], spec_times[-1], spec_freqs[0], spec_freqs[-1]],
)
axs[2].imshow(
    todb(spec_masked),
    aspect="auto",
    origin="lower",
    extent=[spec_times[0], spec_times[-1], spec_freqs[0], spec_freqs[-1]],
)
axs[3].plot(eod_inv_time, eod_inv.numpy())
axs[4].imshow(
    todb(inv_spec),
    aspect="auto",
    origin="lower",
    extent=[spec_times[0], spec_times[-1], spec_freqs[0], spec_freqs[-1]],
)
axs[4].set_ylim([400, 900])
axs[2].set_ylim([400, 900])
axs[1].set_ylim([400, 900])
plt.show()
