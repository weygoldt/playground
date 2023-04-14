# Streaming things to files

I want to create, save and load spectrograms larger than my memory. Here, I will try to solve this issue.

I will compute spectrograms on my GPU using pytorch and stream them to a binary file using h5py. I will then load them back using h5py.