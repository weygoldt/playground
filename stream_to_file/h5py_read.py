#!/usr/bin/env python3

"""
This script opens a spectrogram from a HDF5 file and plots it.
"""

import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

path = pathlib.Path("spectrogram.h5")

samplerate = 20000
buffersize = 2049

with h5py.File(path, "r") as f:
    spec = f["spec"][:]
    nchunks = int(np.ceil(spec.shape[0] / buffersize))

    print(spec.shape)
    print(nchunks)

    for i in range(nchunks):
        print(f"Chunk {i}/{nchunks}")

        # Load data chunk from file to gpu
        spec_chunk = spec[i * buffersize : (i + 1) * buffersize, :]
        print(f"Spectrogram chunk shape: {spec_chunk.shape}")

        # Plot spectrogram
        plt.imshow(spec_chunk, aspect="auto", origin="lower")
        plt.show()
