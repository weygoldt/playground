#!/usr/bin/env python3

"""
This script uses the GPU and computes a spectrogram from a raw file and saves it 
to a HDF5 file.
"""

import math
import pathlib

import h5py
import matplotlib.pyplot as plt
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

with h5py.File("spectrogram.h5", "w") as f:
    spec = f.create_dataset(
        "spec", shape=(0, 0), dtype=np.float32, maxshape=(None, None)
    )
    spec_time = f.create_dataset(
        "spec_time", shape=(0,), dtype=np.float32, maxshape=(None,)
    )
    spec_freq = f.create_dataset(
        "spec_freq", shape=(0,), dtype=np.float32, maxshape=(None,)
    )

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

        plt.pcolormesh(time, freq, spec_chunk_sum)
        plt.show()

        # Resize the time axis

        # spec_time.resize((new_shape[0],))
        # print(new_shape[0])
        # print(spec_time.shape)
        # spec_time[current_shape[0] : new_shape[0]] = time

        # Save the frequency axis
        # if current_shape[0] == 0:
        #     spec_freq.resize((new_shape[1],))
        #     spec_freq[:] = freq

        # Resize the spectrogram dataset to fit the new spectrogram
        current_shape = spec.shape
        new_shape = (
            current_shape[0] + spec_chunk_sum.shape[0],
            spec_chunk_sum.shape[1],
        )
        # Write the data to the dataset
        spec[current_shape[0] : new_shape[0], :] = spec_chunk_sum

        timetracker = time[-1]

f.close()
