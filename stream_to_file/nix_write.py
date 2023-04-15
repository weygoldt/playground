#!/usr/bin/env python3

"""
This script uses the GPU and computes a spectrogram from a raw file and saves it 
to a HDF5 file.
"""

import math
import pathlib

import matplotlib.pyplot as plt
import nixio as nio
import numpy as np
import torch
from thunderfish.dataloader import DataLoader
from torchaudio.transforms import AmplitudeToDB, Spectrogram


def next_power_of_two(num):
    """
    Takes a float as input and returns the next power of two.

    Args:
        num (float): The input number.

    Returns:
        float: The next power of two.
    """
    # Check if the input is already a power of two
    if math.log2(num).is_integer():
        return num

    # Find the next power of two using log2 and ceil
    next_pow = math.ceil(math.log2(num))

    # Return the result
    return 2**next_pow


def freqres_to_nfft(freq_res, samplingrate):
    return next_power_of_two(samplingrate / freq_res)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = pathlib.Path(
    "../../chirpdetector/data/2022-06-02-10_00/traces-grid1.raw"
)

buffersize = 10
signal = DataLoader(
    filepath=str(path),
    buffersize=buffersize,
    backsize=0,
    verbose=0,
    channel=-1,
)
samplingrate = 20000
nfft = freqres_to_nfft(6, samplingrate)
overlap = 0.99
print(f"NFFT: {nfft}")

hop_length = int(nfft * (1 - overlap))
print(f"Hop length: {hop_length}")

chunksize = samplingrate * buffersize
nchunks = int(np.ceil(len(signal[0]) / chunksize))
nchunks = 5
nelectrodes = len(signal[1])

# Configure spec and conversion function and put to gpu
spectrogram_of = Spectrogram(
    n_fft=nfft, hop_length=hop_length, normalized=True, power=2
).to(device)
transform_to_db = AmplitudeToDB(stype="power", top_db=80).to(device)

# Open a NIX file
file = nio.File.open(
    "dataset.nix",
    nio.FileMode.Overwrite,
    compression=nio.Compression.DeflateNormal,
)

block = file.create_block("1", "recording number 1")

timetracker = 0  # Keep track of the time axis
for i in range(nchunks):
    print(f"Chunk {i}/{nchunks}")

    for e in range(nelectrodes):
        # Load data chunk from file to gpu
        wav_chunk = signal[i * chunksize : (i + 1) * chunksize, e]
        wav_chunk = torch.from_numpy(wav_chunk).to(device)

        # Compute spectrogram
        spec_chunk = spectrogram_of(wav_chunk)
        spec_chunk = transform_to_db(spec_chunk)

        # Sum spectrograms of all electrodes
        if e == 0:
            spec_chunk_sum = spec_chunk
        else:
            spec_chunk_sum += spec_chunk

    # Move spectrogram to cpu
    spec_chunk_sum = spec_chunk_sum.cpu().numpy()

    # Compute time and frequency axis
    time = np.arange(spec_chunk_sum.shape[1]) * hop_length / samplingrate
    time += timetracker
    freq = np.arange(spec_chunk_sum.shape[0]) * samplingrate / nfft

    # plt.pcolormesh(time, freq, spec_chunk_sum)
    # plt.show()

    if i == 0:
        # Create data arrays in the file
        spectrogram = block.create_data_array(
            "spec", "spectrogram matrix of raw data", data=spec_chunk_sum
        )
        spectrogram_time = block.create_data_array(
            "spec_time", "time axis of spectrogram", data=time
        )
        spectrogram_freq = block.create_data_array(
            "spec_freq", "frequency axis of spectrogram", data=freq
        )
    else:
        # Append data arrays to file
        spectrogram.append(spec_chunk_sum, axis=1)
        spectrogram_time.append(time, axis=0)
        spectrogram_freq.append(freq, axis=0)

    timetracker = time[-1]
