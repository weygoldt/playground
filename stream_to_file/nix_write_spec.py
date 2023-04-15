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
    """
    Convert the frequency resolution of a spectrogram to
    the number of FFT bins.
    """
    return next_power_of_two(samplingrate / freq_res)


def overlap_to_hoplen(overlap, nfft):
    """
    Convert the overlap of a spectrogram to the hop length.
    """
    return int(np.floor(nfft * (1 - overlap)))


def imshow(spec, time, freq):
    """
    Plot a spectrogram.
    """
    plt.pcolormesh(time, freq, spec)
    plt.ylim(0, 2000)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = pathlib.Path(
        "../../chirpdetector/data/2022-06-02-10_00/traces-grid1.raw"
    )

    buffersize = 20
    signal = DataLoader(
        filepath=str(path),
        buffersize=buffersize,
        backsize=0,
        verbose=0,
        channel=-1,
    )
    samplingrate = signal.samplerate
    nelectrodes = signal.shape[1]

    nfft = freqres_to_nfft(6, samplingrate)
    hop_length = overlap_to_hoplen(0.99, nfft)
    chunk_size = samplingrate * buffersize
    padding = 1 * samplingrate  # padding of raw singnal to limit edge effects

    # Good window for this recording
    window_start_index = (3 * 60 * 60 + 6 * 60 + 20) * samplingrate
    signal = signal[
        window_start_index : window_start_index + 300 * samplingrate
    ]
    nchunks = math.ceil(signal.shape[0] / chunk_size)

    spectrogram_of = Spectrogram(
        n_fft=nfft,
        hop_length=hop_length,
        power=2,
        normalized=True,
    ).to(device)

    in_decibel = AmplitudeToDB(stype="power", top_db=80).to(device)

    file = nio.File.open("dataset.nix", nio.FileMode.Overwrite)
    block = file.create_block("spectrogram", "spectrogram")

    timetracker = 0
    for i in range(nchunks):
        print(f"Chunk {i + 1} of {nchunks}")

        for e in range(nelectrodes):
            # get chunk for current electrode
            # add overlap depending if first, middle or last chunk
            if i == 0:
                chunk = signal[
                    i * chunk_size : (i + 1) * chunk_size + padding, e
                ]
            elif i == nchunks - 1:
                chunk = signal[
                    i * chunk_size - padding : (i + 1) * chunk_size, e
                ]
            else:
                chunk = signal[
                    i * chunk_size - padding : (i + 1) * chunk_size + padding, e
                ]

            # compute how much padding to remove from the start and end of the spec
            # to get the correct time axis
            spec_padding = int(padding // hop_length)

            # convert to tensor and into gpu
            chunk = torch.from_numpy(chunk).to(device)

            # calculate spectrogram
            chunk_spec = spectrogram_of(chunk)

            # remove padding from spectrogram
            if i == 0:
                chunk_spec = chunk_spec[:, :-spec_padding]
            elif i == nchunks - 1:
                chunk_spec = chunk_spec[:, spec_padding:]
            else:
                chunk_spec = chunk_spec[:, spec_padding:-spec_padding]

            # convert to decibel
            chunk_spec = in_decibel(chunk_spec)

            # sum up the spectrograms
            if e == 0:
                spec = chunk_spec
            else:
                spec += chunk_spec

        # normalize by number of electrodes
        spec = spec / nelectrodes

        # convert to numpy and into ram
        spec = spec.cpu().numpy()

        # get time and frequency axis
        time = (
            np.arange(0, spec.shape[1]) * hop_length / samplingrate
            + timetracker
        )
        freq = np.arange(0, spec.shape[0]) * samplingrate / nfft

        # keep track of the time for the next iteration
        timetracker = time[-1]

        # create the data arrays on disk in the first iteration
        if i == 0:
            save_spec = block.create_data_array(
                "spec", "spectrogram matrix", data=spec, unit="dB"
            )
            save_time = block.create_data_array(
                "time", "time axis", data=time, unit="s"
            )
            save_freq = block.create_data_array(
                "freq", "frequency axis", data=freq, unit="Hz"
            )
        else:
            # frequency axis never changes so we can skip it in the following
            save_spec.append(spec, axis=1)
            save_time.append(time, axis=0)


if __name__ == "__main__":
    main()
