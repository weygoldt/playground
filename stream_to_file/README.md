# Streaming things to files

I want to create, save and load spectrograms larger than my memory. Here, I will try to solve this issue.

I will compute spectrograms on my GPU using pytorch and stream them to a binary file using nix, a neuroscience data format built on top of HDF5. It provides a nice python interface and is easier and safer to use than the low-level h5py API.