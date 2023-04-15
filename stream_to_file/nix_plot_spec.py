import pathlib

import matplotlib.pyplot as plt
import nixio as nio


def main():
    path = pathlib.Path("dataset.nix")
    with nio.File.open(str(path), nio.FileMode.ReadOnly) as f:
        spec = f.blocks[0].data_arrays["spec"][:]
        time = f.blocks[0].data_arrays["time"][:]
        freq = f.blocks[0].data_arrays["freq"][:]

        print(spec.shape)
        print(time.shape)
        print(freq.shape)

        # Plot spectrogram
        plt.pcolormesh(time, freq, spec, shading="auto", rasterized=True)
        plt.show()


if __name__ == "__main__":
    main()
