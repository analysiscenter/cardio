[```Batch```](batch.md) defines main class ecg_batch that stores ecg data and 
lists all the actions that can be applied to ecg regardless of models. 

[```Models```](models.md) contains
* ```base_model```, a prototype for any further models
* ```keras_base_model```, stores and runs models with Keras backend
* ```keras_custom_objects```, loss functions and Keras layers used in our models
* a number of built-in models for ecg classification