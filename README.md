# ecg

```ecg``` is a library that works with electrocardiogram signal. It allows easily load and process ECG records and learn models. 
The library is based on [Dataset](https://github.com/analysiscenter/dataset/blob/master/README.md) and supports its whole functionality. 
So you can define your own pipeline, write custom preprocess functions and models or use built-in ones and handle with datasets even if it does not fit into memory.

```ecg``` has two modules: [```batch```](doc/batch.md) and [```models```](doc/models.md). 

In ```batch``` we gather everything you may need to process ECG record:
* load and save signal in a number of formats
* resample, crop and flip signal
* filter signal
* allocate PQ, QT, QRS segments
* calculate heart rate
* apply complex transformations like fft or wavelets
* apply custom functions.

In ```models``` we provide a template model and several real models that should inspire you to start you own research.
Provided models are created to learn the most important problems in ECG:
* how to recognize specific features of ECG like R-peaks, P-wave, T-wave
* how to recignize dangerous deseases from ECG, for example - atrial fibrillation.

# Basic usage

Here is an example of pipeline that loads ECG signals, makes some preprocessing and learns model over 50 epochs.
```python
model_train_pipeline = (ds.Pipeline()
                        .load(fmt="wfdb", components=["signal", "meta"])
                        .load(src=".../data/REFERENCE.csv", fmt="csv", components="target")
                        .drop_labels(["~"])
                        .replace_labels({"N": "NO", "O": "NO"})
                        .random_resample_signals("normal", loc=300, scale=10)
                        .drop_short_signals(3000)
                        .split_signals(3000, 1000)
                        .binarize_labels()
                        .train_on_batch('my_ecg_model', metrics=f1_score, average='macro')
                        .run(batch_size=300, shuffle=True, n_epochs=50, prefetch=0))
```
As a result of this pipeline one obtains a trained model.

# How to start

See [tutorial](https://github.com/analysiscenter/ecg/blob/unify_models/doc/tutorial.md) to start working with ECG.

# Further reading

Detailed documentation on ```ecg``` is available [here](https://github.com/analysiscenter/ecg/blob/unify_models/doc/README.md).
