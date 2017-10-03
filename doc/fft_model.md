# FFT inception model

FFT model learns to classify ecg signals using signal spectrum. At first step it convolves signal with a number of kernels. Then for each channel it applies fast fouriet transform, The result is considered as 2D image and is processed with a number of [Inception2d]() blocks to resulting output, which is a predicted class.

## How to use
We applied this model to arrhythmia prediction from single-lead ecg. Training pipeline we used for the fft model looks as follows:
```python
fft_train_pipeline = (ds.Pipeline()
                      .load(fmt="wfdb", components=["signal", "meta"])
                      .load(src=".../REFERENCE.csv", fmt="csv", components="target")
                      .drop_labels(["~"])
                      .replace_labels({"N": "NO", "O": "NO"})
                      .random_resample_signals("normal", loc=300, scale=10)
                      .drop_short_signals(4000)
                      .segment_signals(3000, 3000)
                      .binarize_labels()
                      .signal_transpose([0, 2, 1])
                      .ravel())
```