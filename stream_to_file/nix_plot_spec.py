import pathlib

import matplotlib.pyplot as plt
import nixio as nio
from IPython import embed


def main():
    path = pathlib.Path("../../chirpdetector-cnn/real_data/dataset.nix")
    with nio.File.open(str(path), nio.FileMode.ReadOnly) as f:
        # embed()g
        spec = f.blocks[0].data_arrays["spec"][:, :20000]
        time = f.blocks[0].data_arrays["spec_time"][:20000]
        freq = f.blocks[0].data_arrays["spec_freq"][:]

        print(spec.shape)
        print(time.shape)
        print(freq.shape)

        # Plot spectrogram
        plt.pcolormesh(time, freq, spec, shading="auto", rasterized=True)
        plt.show()


if __name__ == "__main__":
    main()
