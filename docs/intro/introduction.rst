============
Introduction
============

This section describes briefly capabilities of the CardIO framework.


:doc:`Batch <./batch>`
--------
The main class the CardIO is EcgBatch. It contains the I/O and preprocessing :func:`actions <dataset.action>` that allow to load and prepare data for the modeling.

.. code-block :: python

  from cardio import EcgBatch
  from dataset import FilesIndex, Dataset
  ecg_index = FilesIndex(path='path/to/ecg/*', no_ext=True) # set up the index
  dtst = Dataset(index=ecg_index, batch_class=EcgBatch) # init the dataset with ECG files

:doc:`Models <./models>`
------
This module contain model suited to classify whether ECG signal is normal or pathological, to annotate segments of the signal (e.g., P-wave).

.. code-block:: python

  template_dirichlet_train = (
  ds.Pipeline()
    .init_model("dynamic", DirichletModel, name="dirichlet", config=model_config)
    .init_variable("loss_history", init=list)
    .load(components=["signal", "meta"], fmt="wfdb")
    .load(src='./path/to/taret/', fmt="csv", components="target")
    .drop_labels(["~"])
    .replace_labels({"N": "NO", "O": "NO"})
    .flip_signals()
    .random_resample_signals("normal", loc=300, scale=10)
    .random_split_signals(2048, {"A": 9, "NO": 3})
    .binarize_labels()
    .train_model("dirichlet", make_data=make_data,
                 fetches="loss", save_to=V("loss_history"), mode="a")
    .run(batch_size=100, shuffle=True, drop_last=True, n_epochs=100, lazy=True)
  )

:doc:`Pipelines <./pipelines>`
---------
Pipelines were designed to ease usage of exhisting models make final code simpler. 

.. code-block:: python

  from cardio.pipelines import hmm_predict_pipeline
  res = (data >> hmm_train_pipeline(model_path)).run()

Under the hood this function contains a list of actions:

.. code-block:: python

  template_hmm_predict = (
  ds.Pipeline()
    .init_model("dynamic", HMModel, "HMM", config=config_train)
    .load(fmt='wfdb', components=["signal", "annotation", "meta"], ann_ext='pu1')
    .wavelet_transform_signal(cwt_scales=[4,8,16], cwt_wavelet="mexh")
    .train_model("HMM", make_data=make_data)
    .run(batch_size=20, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
  )
  res = (data >> template_hmm_predict).run()  

Tutorials
---------

There are three tutorials:

* `CardIO <https://github.com/analysiscenter/cardio/blob/master/tutorials/I.CardIO.ipynb>`_
In this tutorail we briefly introduce some instances of `Dataset <https://github.com/analysiscenter/dataset>`_ and show capabilities of the CardIO's EcgBatch class.

* `Pipelines <https://github.com/analysiscenter/cardio/blob/master/tutorials/II.Pipelines.ipynb>`_
In this tutorial we show how to create pipelines, use them for preprocessing and add your custom action to the EcgBatch with ease.

* `Models <https://github.com/analysiscenter/cardio/blob/master/tutorials/III.Models.ipynb>`_
This tutorial shows how to embed models in pipelines to perform training and prediction.