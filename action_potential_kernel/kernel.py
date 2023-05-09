import matplotlib.pyplot as plt
import numpy as np
from IPython import embed


def gaussian(x, mu, sig, a=1):
    x = np.array(x)
    return a * np.exp(-(np.power((x - mu), 2)) / (2 * np.power((sig * 2.0), 2)))


def chirp(width, height, undershoot, fs):
    s = width / 10
    x = np.arange(-10 * s, 10 * s, 1 / fs)
    g = gaussian(x, 0, s, height)
    d = gaussian(x, s * 3, s, height * undershoot)
    return g - d


def main():
    c = chirp(
        width=0.02,
        height=120,
        undershoot=0.2,
        fs=20000,
    )
    t = np.arange(0, len(c) / 20000, 1 / 20000)
    t = t - t[len(t) // 2]
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")
    plt.plot(t, c)
    plt.show()
    pass


if __name__ == "__main__":
    main()
