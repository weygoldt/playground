#!/usr/bin/env python

"""
This module demonstrates the simulation of a chirp frequency trace with an 
optional undershoot at the end. 
"""


import matplotlib.pyplot as plt
import numpy as np


def gaussian(mu, width, kurt, fs):
    chirp_t = np.arange(-2.0 * width, 2.0 * width, 1.0 / fs)
    chirp_sig = 0.5 * width / (2.0 * np.log(10.0)) ** (0.5 / kurt)
    gauss = np.exp(-0.5 * (((chirp_t - mu) / chirp_sig) ** 2.0) ** kurt)
    return gauss


def main():
    # standard chirp parameters
    width = 0.02
    height = 120
    kurt = 1
    fs = 20000

    # the undershoot parameter is expressed as a fraction of the
    # main Gaussian's amplitude
    # --------------------
    undershoot = 0.3
    # --------------------

    # make the main Gaussian
    g1 = gaussian(
        mu=0,
        width=width,
        kurt=kurt,
        fs=fs,
    )

    # the undershoot Gaussian could start at half the width of
    # the main Gaussian
    g2 = gaussian(
        mu=width / 2,
        width=width,
        kurt=1,
        fs=fs,
    )

    chirp = g1 * height - g2 * height * undershoot

    plt.plot(chirp)
    plt.show()


if __name__ == "__main__":
    main()
