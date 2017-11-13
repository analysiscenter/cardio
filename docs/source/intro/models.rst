======
Models
======

This is a place where ECG models live. You can write your own model or exploit provided models. 

Built-in models
---------------

We have a number of built-in `models <../api/models>`_ for ECG classification and annotation:

* FFTModel
* DirichletModel
* HMModel

Below you can find a guide how to build your own model with Keras framework.

How to build a model with Keras
-------------------------------

Any custom Keras model starts with base model KerasModel. In most cases you simply create
a new class that inherit KerasModel and define a sequence of layers within the _build method.
Once it is done you can include train and predict actions into pipeline.

For example, let's build a simple fully-connected network. It will accept signal with shape (1000, ) and return shape (2, ).
First, we import KerasModel:

.. code-block :: python

  from ...dataset.dataset.models.keras import KerasModel

Second, define our model architecture. Note that _build should return input and output layers.

.. code-block :: python

  class SimpleModel(KerasModel):
      def _build(self, **kwargs):
          '''
          Build model
          '''
          x = Input(1000)
          out = Dense(2)(x)
          return x, out

Third, we specify model configuration (loss and optimizer) and initialize model in pipeline.
We suppose that batch has a component named 'signal' (this will be our input tensor) and a component
named 'target' (this will be our output tensor).

.. code-block :: python

  model_config = {
      "loss": "binary_crossentropy",
      "optimizer": "adam"
      }

  train_pipeline = (ds.Pipeline()
                    .init_model("static", SimpleModel, name="simple_model", config=model_config)
                    .init_variable("loss_history", init=list)
                    ...
                    some data preprocessing
                    ...
                    .train_model('simple_model', x=B('signal'), y=B('target'),
                                 save_to=V("loss_history"), mode="a"))

Fron now on ``train_pipeline`` contains compiled model and is ready for training.


See more detailed documentation on models `here <https://analysiscenter.github.io/dataset/intro/models.html>`_.
