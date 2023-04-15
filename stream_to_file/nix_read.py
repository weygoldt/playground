import pathlib

import matplotlib.pyplot as plt
import nixio as nio
import numpy as np

path = pathlib.Path("dataset.nix")

samplerate = 20000
buffersize = 2049

with nio.File.open(str(path), nio.FileMode.ReadOnly) as f:
    spec = f.blocks[0].data_arrays["spectrogram"][:]
    time = f.blocks[0].data_arrays["time"][:]
    freq = f.blocks[0].data_arrays["freq"][:]

    print(spec.shape)
    print(time.shape)
    print(freq.shape)

    # Plot spectrogram
    # plt.pcolormesh(time, freq, spec, shading="auto", rasterized=True)
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.show()
