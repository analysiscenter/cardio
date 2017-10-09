# Models

This is a place where ECG models live. You can write your own model or exploit provided models, however, any model should extend [base_model]() class. 

## Base model
All what you may need from model are the following methods:

* ```load```: load model

* ```save```: save model

* ```train_on_batch```: run a single gradient update on a single batch
        
* ```test_on_batch```: get model loss for a single batch

* ```predict_on_batch```: get model predictions for a single batch.

Once these methods are defined one can add them to pipeline.
For example, train pipline looks like

```python
model_train_pipeline = (ds.Pipeline()
                        .load(fmt="wfdb", components=["signal", "meta"])
                        .load(src=".../data/REFERENCE.csv", fmt="csv", components="target")
                        .do_some_preprocess()
                        .train_on_batch('fft_inception', metrics=f1_score, average='macro'))
```

Note that until now everything was independent on model backend.
Below you can find a guide how to build a model with [Keras]() and [Tensorflow]()

## KerasBaseModel
Defines model and implements methods of BaseModel for Keras backend. 
Class KerasBaseModel has two attributes:
* ```model```:this is where Keras [Model]() is stored
* ```hist```: keeps loss and metrics on every batch during trainig and testing of the model.

Available methods are:
* ```train_on_batch```: train model on batch and write loss and metrics to pipeline variable
* ```test_on_batch```: test model on batch and write loss and metrics to pipeline variable
* ```predict_on_batch```: predict batch and write prediction to batch or pipeline variable
* ```model_summary```: print model summary
* ```load```: load model
* ```save```: save model.

See details of methods [here]()

## How to build a model with Keras

To build a model with Keras you only need to define a sequence of layers, everything else is implemented in [KerasBaseModel]().
For example, let's build a simple FC model. 
```python
from keras.layers import Input, Dense
from keras.models import Model

class SimpleFCModel(KerasBaseModel):
    def __init__(self):
        super().__init__()
        self._input_shape = None
        
        def build(self, input_shape):
            '''
            Build and compile conv model
            '''
            self._input_shape = input_shape
            x = Input(self._input_shape)
            out = Dense(16)(x)
            self.model = Model(inputs=x, outputs=out)
            self.model.compile(loss="binary_crossentropy", optimizer="adam")
            return self
```
SimpleFCModel is a [dynamic]() model, i.e. it is build and compiles at first time it gets batch. So we do not need specify the input shape in advance. The dynamic model gets is automatically from batch. To enable dynamic mode, the following declaration is required in [model_ecg_batch]():

```python
@ds.model(mode="dynamic")
def fc_model(batch, config=None):
    '''
    Define simple FC model model
    '''
    signal_shape = batch.signal[0].shape
    return SimpleFCModel().build(signal_shape)
```

Now everything is ready to train:
```python
fc_train_pipeline = (ds.Pipeline()
                       .load(fmt="wfdb", components=["signal", "meta"])
                       .load(src=".../data/REFERENCE.csv", fmt="csv", components="target")
                       .do_some_preprocess()
                       .train_on_batch('fc_model')
                       .run(batch_size=300, shuffle=True, drop_last=True, n_epochs=50))
```
and predict our model:
```python
config = {'path': "/path_to_fc_model_dump"}
fc_predict_pipeline = (ds.Pipeline(config={'fc_model': config})
                         .init_model('fc_model')
                         .init_variable("prediction", [])
                         .load(fmt="wfdb", components=["signal", "meta"])
                         .load(src=".../data/REFERENCE.csv", fmt="csv", components="target")
                         .do_some_preprocess()
                         .predict_on_batch('fc_model')
                         .run(batch_size=100, shuffle=False, drop_last=False, n_epochs=1))
```