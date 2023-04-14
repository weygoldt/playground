import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed
from librosa import power_to_db
from thunderfish.dataloader import DataLoader
from torchaudio.transforms import AmplitudeToDB, Spectrogram

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = pathlib.Path(
    "../../chirpdetector/data/2022-06-02-10_00/traces-grid1.raw"
)

buffersize = 180
signal = DataLoader(
    filepath=str(path),
    buffersize=buffersize,
    backsize=0,
    verbose=0,
    channel=-1,
)
nfft = 4096
samplingrate = 20000
hop_length = nfft // 8

chunksize = samplingrate * buffersize
nchunks = int(np.ceil(len(signal[0]) / chunksize))
nchunks = 10
nelectrodes = len(signal[1])

spectrogram_of = Spectrogram(
    n_fft=nfft, hop_length=hop_length, normalized=True, power=2
).to(device)
transform_to_db = AmplitudeToDB(stype="power", top_db=80).to(device)

with h5py.File("spectrogram.h5", "w") as f:
    spec = f.create_dataset(
        "spec", shape=(1, 1), dtype=np.float32, maxshape=(None, None)
    )

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

        # plt.imshow(spec_chunk_sum, aspect="auto", origin="lower")
        # plt.show()

        # Resize the dataset to fit the new data
        current_shape = spec.shape
        new_shape = (
            current_shape[0] + spec_chunk.shape[0],
            spec_chunk.shape[1],
        )
        spec.resize(new_shape)

        # Write the data to the dataset
        spec[current_shape[0], new_shape[0], :] = spec_chunk
