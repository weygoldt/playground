import math
import pathlib

import matplotlib.pyplot as plt
import nixio as nio
import numpy as np
import torch
from IPython import embed
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


def imshow(spec, time, freq):
    plt.pcolormesh(time, freq, spec)
    plt.ylim(0, 1000)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = pathlib.Path(
    "../../chirpdetector/data/2022-06-02-10_00/traces-grid1.raw"
)

buffersize = 10
samplingrate = 20000
nfft = freqres_to_nfft(6, samplingrate)
fft_overlap = 0.99
hop_lenght = int(nfft * (1 - fft_overlap))
chunk_size = samplingrate * buffersize
nchunks = 6
spec_overlap = 0.02

signal = DataLoader(
    filepath=str(path),
    buffersize=buffersize,
    backsize=0,
    verbose=0,
    channel=-1,
)

nelectrodes = signal.shape[1]
window_start_index = (3 * 60 * 60 + 6 * 60 + 20) * samplingrate
signal = signal[window_start_index : window_start_index + 240 * samplingrate]

spectrogram_of = Spectrogram(
    n_fft=nfft,
    hop_length=hop_lenght,
    power=2,
    normalized=True,
).to(device)

in_decibel = AmplitudeToDB(stype="power", top_db=80).to(device)

file = nio.File.open("dataset.nix", nio.FileMode.Overwrite)
block = file.create_block("spectrogram", "spectrogram")

timetracker = 0
for i in range(nchunks):
    print(f"Chunk {i + 1}/{nchunks}")

    if i == 0:
        start = 0
        stop = chunk_size
    # else:

    for e in range(nelectrodes):
        chunk = signal[start:stop, e]
        chunk = torch.from_numpy(chunk).to(device)
        spec_chunk = spectrogram_of(chunk)
        spec_chunk = in_decibel(spec_chunk)

        if e == 0:
            spec_sum = spec_chunk
        else:
            spec_sum += spec_chunk

    spec_sum = spec_sum / nelectrodes
    spec_sum = spec_sum.cpu().numpy()

    time = (
        np.arange(spec_sum.shape[1]) * hop_lenght / samplingrate + timetracker
    )
    timetracker = time[-1]
    freq = np.arange(spec_sum.shape[0]) * samplingrate / nfft

    start = int(stop - chunk_size * spec_overlap * 2)
    stop = int(start + chunk_size)
    print(start, stop)

    # only crop end for first spec
    if i == 0:
        imshow(spec_sum, time, freq)
        spec_sum = spec_sum[:, : -int(spec_overlap * spec_sum.shape[1])]
        time = time[: -int(spec_overlap * spec_sum.shape[1])]
        plt.axvline(time[-1], color="red", linestyle="--")
        plt.axvline(time[0], color="red", linestyle="--")
        plt.axvline(start / samplingrate, color="green", linestyle="--")
        # plt.show()

    # only crop start for last spec
    elif i == nchunks - 1:
        imshow(spec_sum, time, freq)
        spec_sum = spec_sum[:, int(spec_overlap * spec_sum.shape[1]) :]
        time = time[int(spec_overlap * spec_sum.shape[1]) :]
        plt.axvline(time[-1], color="red", linestyle="--")
        plt.axvline(time[0], color="red", linestyle="--")
        plt.axvline(start / samplingrate, color="green", linestyle="--")
        # plt.show()

    # crop both start and end for all other specs
    else:
        imshow(spec_sum, time, freq)
        spec_sum = spec_sum[
            :,
            int(spec_overlap * spec_sum.shape[1]) : -int(
                spec_overlap * spec_sum.shape[1]
            ),
        ]
        time = time[
            int(spec_overlap * spec_sum.shape[1]) : -int(
                spec_overlap * spec_sum.shape[1]
            )
        ]
        plt.axvline(time[0], color="red", linestyle="--")
        plt.axvline(time[-1], color="red", linestyle="--")
        plt.axvline(start / samplingrate, color="green", linestyle="--")
        # plt.show()

    # create nix data array in first iteration
    if i == 0:
        save_spectrogram = block.create_data_array(
            "spectrogram",
            "spectrogram",
            data=spec_sum,
            label="spectrogram",
            unit="dB",
        )
        save_time = block.create_data_array(
            "time",
            "time",
            data=time,
            label="time",
            unit="s",
        )
        save_freq = block.create_data_array(
            "freq",
            "freq",
            data=freq,
            label="freq",
            unit="Hz",
        )
    # append data in all other iterations
    else:
        save_spectrogram.append(spec_sum, axis=1)
        save_time.append(time, axis=0)
