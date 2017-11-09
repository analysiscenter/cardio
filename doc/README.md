Repository ```cardio``` consists of three main modules ```batuleh```,  ```models```, ```pipelines``` and submodule ```dataset```.

Module [```batch```](batch.md) contains:
* ```ecg_batch```, definition of class EcgBatch and actions
* ```ecg_batch_tools```, general methods for signal processing
* ```kernels```, kernel generation fuctions for signal convolution
* ```utils```, miscellaneous EcgBatch utils.

In [```models```](models.md) we define built-in models for ECG classification and segmentation:
* ```dirichlet_model```
* ```fft_model```
* ```hmm_model```
* ```keras_custom_objects```
* ```metrics```

Module [```pipelines```](pipelines.md) contains pipelines we used
to train models and get predictions.