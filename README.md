# ecg

```ecg``` is a library that works with electrocardiogram signal. It allows easily load and process ecg signal and learn any model. 
The library is based on [Dataset](https://github.com/analysiscenter/dataset/blob/master/README.md) and supports its whole functionality. 
So you can define your own pipeline, write custom preprocess functions or use built-in ones and handle with datasets even if it does not fit into memory.

```ecg``` has two modules: [```batch```](doc/batch.md) and [```models```](doc/models.md). 

In ```batch``` we gather everything you may need to process ecg signal:
* load and save signal in a number of formats
* resample, crop and flip signal
* filter signal
* apply complex transformations like fft or wavelets
* apply custom functions.

In ```models``` we provide a template model and several real models that should inspire you to start you own research.
Provided models are created to learn the most important problems in ecg:
* how to recognize specific features of ecg like R-peaks, P-wave, T-wave
* how to recignize dangerous deseases from ecg, for example - atrial fibrillation.

# Basic usage

Here is an example of pipeline that loads ecg signals, makes some preprocessing and learns model over 50 epochs.
```python
model_train_pipeline = (ds.Pipeline()
                        .load(fmt="wfdb", components=["signal", "meta"])
                        .load(src=".../data/REFERENCE.csv", fmt="csv", components="target")
                        .drop_labels(["~"])
                        .replace_labels({"N": "NO", "O": "NO"})
                        .random_resample_signals("normal", loc=300, scale=10)
                        .drop_short_signals(3000)
                        .segment_signals(3000, 1000)
                        .binarize_labels()
                        .train_on_batch('my_ecg_model', metrics=f1_score, average='macro')
                        .run(batch_size=300, shuffle=True, n_epochs=50, prefetch=0))
```
As a result of this pipeline one obtains trained model.

# How to start

See [tutorial](https://github.com/analysiscenter/ecg/blob/unify_models/doc/tutorial.md) to start working with ecg.
