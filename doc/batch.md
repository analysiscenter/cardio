# Batch

## ecg_batch
This is a base class that defines how to store ECG data and lists all the actions that can be applied to ECG regardless of models. Class ```ecg_base``` has the following attributes:
* ```signal``` stores ECG signal in numpy array
* ```annotation```, this can be any array that annotate specific points of ECG signal, e.g. R peaks
* ```meta``` contains any parameters of ECG signal, e.g. signal frequency
* ```target``` contains label assigned to ECG
* ```unique_labels``` is just a list of all possible target labels.

### Actions of ecg_batch

* ```load```

* ```drop_labels```

* ```keep_labels```

* ```replace_labels```

* ```binarize_labels```

* ```drop_short_signals```

* ```segment_signals```

* ```random_segment_signals```

* ```convolve_signals```

* ```band_pass_signals```

* ```apply```


## ecg_batch_tools

Contains helpful functions that typically are called from some action of the ecg_batch for each signal separately. 


## kernels

Contains kernel generation functions.

## model_ecg_batch

Initialize models to make them available in pipeline. We have a number of built-in models:

* [```fft_inception```](fft_model.md)

* [```triplet_learn```](triplet_model.md)

* [```dirichlet```](dirichlet_model.md)

* [```conv_model```](conv_model.md)


## utils

Miscellaneous ECG Batch utils