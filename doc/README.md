```Ecg``` repository consists of two main modules ```batch``` and ```models``` and submodule ```dataset```, which
is a core framework.

In [```Batch```](batch.md) we define a class EcgBatch that stores ECG
records and definitions of various actions that can be applied to them. Content of ```batch```:
* ```ecg_batch```, definition of class EcgBatch and actions
* ```ecg_batch_tools```, general methods for signal processing
* ```kernels```, kernel generation fuctions for signal convolution
* ```model_ecg_batch```, extention of EcgBatch class to bind it with models
* ```utils```, miscellaneous EcgBatch utils.

In [```Models```](models.md) we define a base class for ECG models and provide several
ready-to-use models. Content of ```models```:
* ```base_model```, a prototype for ECG models
* ```keras_base_model```, model actions (train, predict, save, etc) adapted for Keras models 
* ```keras_custom_objects```, loss functions and custom layers used in provided Keras models
* ...a number of built-in models for ECG classification.