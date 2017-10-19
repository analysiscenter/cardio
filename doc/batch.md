# Batch



## ecg_batch

Contains base class EcgBatch that defines how to store ECG data and lists actions
that can be applied to ECG in preprocessing stage. 

Attributes of EcgBatch:
* ```signal```, stores ECG signals in numpy array
* ```annotation```, array of dicts with different types of annotations, e.g. array of R-peaks
* ```meta```, array of dicts with metadata about ECG records, e.g. signal frequency
* ```target```, array of labels assigned to ECG records
* ```unique_labels```, array of all possible target labels in dataset.

Methods of EcgBatch allows:
* load ECG records from wfdb or blosc format
* segment, flip and resample signals
* filter signals 
* allocate PQ, QT, QRS segments
* dump results in blosc.

More detailed API see [here]().


## ecg_batch_tools

Contains general medods for signal processing that are exploited in EcgBatch actions.


## kernels

Contains kernel generation functions for signal convolution.

## model_ecg_batch

Here we define class ```ModelEcgBatch``` that extends ```EcgBatch``` and binds it with models. 
```ModelEcgBatch``` initializes models to make them available in pipeline and adds actions for
model training and prediction. 

We have a number of built-in models that are ready to use:
* [```fft_inception```](fft_model.md)
* [```triplet_learn```](triplet_model.md)
* [```dirichlet```](dirichlet_model.md)
* [```conv_model```](conv_model.md)


## utils

Miscellaneous ECG Batch utils.