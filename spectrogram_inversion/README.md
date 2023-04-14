# Spectrogram inversion

Spectrograns are very useful 2D representations of periodic signals, such as electricity or sound. But why should you try to convert a spectrogram back to sound if you could just take the original recording? It is what we can do with the spectrogram **before** the inversion.

A spectrogram is essentially an image, so we can mask frequency bands, remove noise with a dynamic frequency component or extract certain frequency bands that change over time. This is essentially what a bandpass filter does, just that a bandpass filter has a fixed passband. 

So this is the motivation for this experiment. Spectrogram inversions are implemented in [pytorch](https://pytorch.org/) and [squeezepy](https://github.com/OverLordGoldDragon/ssqueezepy) (and probably some more packages as well). In this example I use pytorch since it is already optimized for GPU use.