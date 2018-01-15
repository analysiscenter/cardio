=========
Pipelines
=========

Module ``pipelines`` contains functions that build pipelines that we used to train models and make predictions. You can use them as-is to train similar models or adjust them in order to get better perfomance.

About pipelines
---------------

Pre-defined pipelines were designed to make it easier to use our models and make code simpler and more readable.

Here is an example:

Running this two lines of code

.. code-block:: python

  from cardio.pipelines import dirichlet_train_pipeline
  pipeline = dirichlet_train_pipeline(labels_path, batch_size=256, n_epochs=1000, gpu_options=gpu_options)

is similar to running this

.. code-block:: python
  
  from cardio import dataset as ds
  from cardio.dataset import F, V
  from cardio.models import DirichletModel

  model_config = {
        "session": {"config": tf.ConfigProto(gpu_options=gpu_options)},
        "input_shape": F(lambda batch: batch.signal[0].shape[1:]),
        "class_names": F(lambda batch: batch.label_binarizer.classes_),
        "loss": None}

  pipeline = (
    ds.Pipeline()
      .init_model("dynamic", DirichletModel, name="dirichlet", config=model_config)
      .init_variable("loss_history", init=list)
      .load(components=["signal", "meta"], fmt="wfdb")
      .load(components="target", fmt="csv", src=labels_path)
      .drop_labels(["~"])
      .rename_labels({"N": "NO", "O": "NO"})
      .flip_signals()
      .random_resample_signals("normal", loc=300, scale=10)
      .random_split_signals(2048, {"A": 9, "NO": 3})
      .binarize_labels()
      .train_model("dirichlet", make_data=concatenate_ecg_batch, fetches="loss", save_to=V("loss_history"), mode="a")
      .run(batch_size=256, shuffle=True, drop_last=True, n_epochs=1000, lazy=True))

In both cases you obtain ``pipeline``, ready for training DirichletModel. The first example is short but only allows to vary some hyperparameters of the model, while the second example is more flexible in data preprocessing.

How to use
----------
Working with pipelines consists of 3 simple steps. First, we import desired pipeline, e.g. dirichlet_train_pipeline:
::
  from cardio.pipelines import dirichlet_train_pipeline

Second, we specify its parameters, e.g. path to file with labels:
::
  pipeline = dirichlet_train_pipeline(labels_path='some_path')

Third, we pass dataset to the pipeline and run caclulation:
::
  (dataset >> pipeline).run(batch_size=100, n_epochs=10)

Result is typically a trained model or some values stored in pipeline variable (e.g. model predicitons).

Available pipelines
-------------------
At this moment the module contains following pipelines:

* :func:`~cardio.pipelines.dirichlet_train_pipeline`
* :func:`~cardio.pipelines.dirichlet_predict_pipeline`
* :func:`~cardio.pipelines.hmm_preprocessing_pipeline`
* :func:`~cardio.pipelines.hmm_train_pipeline`
* :func:`~cardio.pipelines.hmm_predict_pipeline`

API
---
See :doc:`Pipelines API <../api/pipelines>`