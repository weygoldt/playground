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
fft_overlap = 0.99
spec_overlap = 100  # num windows overlapping between two spectrograms
print(f"NFFT: {nfft}")
print(f"FFT overlap: {fft_overlap}")
print(f"Spec overlap: {spec_overlap}")

window_start_index = (3 * 60 * 60 + 6 * 60 + 20) * samplingrate
signal = signal[window_start_index:]

hop_length = int(nfft * (1 - fft_overlap))
print(f"Hop length: {hop_length}")

chunksize = samplingrate * buffersize
nchunks = int(np.ceil(len(signal[0]) / chunksize))
nchunks = 8
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

sampletracker = 0  # Keep track of the sample axis
for i in range(nchunks):
    print(f"Chunk {i+1}/{nchunks}")

    if i == 0:
        chunkstart = 0
        chunkstop = chunksize

    for e in range(nelectrodes):
        # Load data chunk from file to gpu
        wav_chunk = signal[chunkstart:chunkstop, e]
        wav_chunk = torch.from_numpy(wav_chunk).to(device)
        end_index = wav_chunk.shape[0]

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

    # Compute start for next window
    # Overlap for all subsequent chunks
    # For this convert spectrogram overlap into samples

    chunkstart = int(chunkstop - (spec_overlap * 2 * hop_length))
    chunkstop = chunkstart + chunksize

    print(chunkstart, chunkstop)

    # Compute current sample, time and frequency axis
    # for the spectrogram corresponding to the raw data

    samples = np.arange(spec_chunk_sum.shape[1]) * hop_length
    samples += sampletracker
    time = samples / samplingrate
    freq = np.arange(spec_chunk_sum.shape[0]) * samplingrate / nfft

    # apply overlap of spectrogram to remove edge effects
    if i == 0:
        spec_chunk_sum = spec_chunk_sum[:, :-spec_overlap]
        time = time[:-spec_overlap]
    elif i == nchunks - 1:
        spec_chunk_sum = spec_chunk_sum[:, spec_overlap:]
        time = time[spec_overlap:]
    else:
        spec_chunk_sum = spec_chunk_sum[:, spec_overlap:-spec_overlap]
        time = time[spec_overlap:-spec_overlap]

    plt.pcolormesh(time, freq, spec_chunk_sum)
    # plt.axvline(chunkstart / samplingrate, color="red")
    # plt.axvline(time[spec_overlap], color="gray")
    # plt.axvline(time[-spec_overlap], color="gray")
    plt.show()

    print(time[-1] - time[0])

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

    sampletracker = samples[-1]
