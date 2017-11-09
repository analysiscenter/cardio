# CardIO

CardIO is a library that works with electrocardiograms (ECG). With CardIO you can
* calculate heart beat rate and find other standard ECG characteristics
* recognize heart diseases from ECG
* efficiently work with large datasets that do not even fit into memory
* easily arrange new custom actions into pipelines
* do end-to-end ECG processing
* build, train and test custom models for deep research

… and do everything under a single API.

The library is based on [Dataset](https://github.com/analysiscenter/dataset/blob/master/README.md). We suggest to read Dataset's [documentation](https://analysiscenter.github.io/dataset/) to learn more, however, you may skip it for the first reading.

CardIO has three modules: [```batch```](doc/batch.md) [```models```](doc/models.md) and [```pipelines```](doc/pipelines.md).

Module ```pipelines``` contains high-level methods that
* train model to allocate PQ, QT, QRS segments
* calculate heart rate
* train model to find probabilities of heart diseases.

Under the hood these methods contain many actions that load signals, filter it and do complex caclulations. Using pipelines you do not think about this part of work and simply pass ECG datasets and get results.

Module ```batch``` contains low-level actions for ECG processing.
Actions are included in ```EcgBatch``` class that also defines how
to store ECGs. From these actions you can biuld new pipelines. You can also
write custom action and include it in ```EcgBatch```.

In ```models``` we provide several models that were elaborated to learn the most important problems in ECG:
* how to recognize specific features of ECG like R-peaks, P-wave, T-wave
* how to recognize heart diseases from ECG, for example - atrial fibrillation.

# Basic usage

Here is an example of pipeline that loads ECG signals, makes some preprocessing and learns model over 50 epochs.
```python
model_train_pipeline = (ds.Pipeline()
                        .init_model("dynamic", FFTModel, name="fft_model", config=model_config)
                        .init_variable("loss_history", init=list)
                        .load(fmt="wfdb", components=["signal", "meta"])
                        .load(src="/notebooks/data/ECG/training2017/REFERENCE.csv",
                              fmt="csv", components="target")
                        .drop_labels(["~"])
                        .replace_labels({"N": "NO", "O": "NO"})
                        .random_resample_signals("normal", loc=300, scale=10)
                        .drop_short_signals(4000)
                        .split_signals(3000, 3000)
                        .binarize_labels()
                        .apply(np.transpose , axes=[0, 2, 1])
                        .ravel()
                        .get_targets('true_targets')
                        .train_model('fft_model', make_data=make_data,
                                     save_to=V("loss_history"), mode="a")
                        .run(batch_size=300, shuffle=True,
                             drop_last=True, n_epochs=50, prefetch=0, lazy=True))
```
As a result of this pipeline one obtains a trained model.

# How to start

See [tutorial](doc/tutorial.md) to start working with ECG.

# Further reading

Detailed documentation on `ecg` is available [here](doc/README.md).
