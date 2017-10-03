[```Batch```](batch.md) defines main class ecg_batch that stores ecg data and 
lists all available actions for ecg preprocessing. 

[```Models```](models.md) contains
* ```base_model```, a prototype for any further models
* ```keras_base_model```, stores and runs model with Keras backend
* ```keras_custom_objects```, loss functions and Keras layers used in Keras models
* a number of built-in models for ecg classification.