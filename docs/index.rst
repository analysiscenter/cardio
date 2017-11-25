===================================
Welcome to CardIO's documentation!
===================================
`CardIO` is designed to build end-to-end machine learning models for deep research of electrocardiograms.

Main features:

* load and save signal in various formats (wfdb, blosc, etc)
* resample, crop, flip and filter signals
* detect PQ, QT, QRS segments
* calculate heart rate and other ECG characteristics
* apply complex transformations like fft and wavelets, as well as custom functions
* recognize heart diseases (e.g. atrial fibrillation)
* efficiently work with large datasets that do not even fit into memory
* perform end-to-end ECG processing
* build, train and test neural networks and other machine learning models.


.. note:: CardIO is based on `Dataset <https://github.com/analysiscenter/dataset>`_. You might benefit from reading `its documentation <https://analysiscenter.github.io/dataset>`_. However, it is not required, especially at the beginning.

CardIO has three modules: :doc:`core <./api/core>`, :doc:`models <./api/models>` and :doc:`pipelines <api/pipelines>`.


``core`` module contains ``EcgBatch`` and ``EcgDataset`` classes.
``EcgBatch`` defines how ECGs are stored and includes actions for ECG processing. These actions might be used to build multi-staged workflows that can also involve machine learning models. ``EcgDataset`` is a class that stores indices of ECGs and generates batches of type ``EcgBatch``.

``models`` module provides several ready to use models for important problems in ECG analysis:

* how to detect specific features of ECG like R-peaks, P-wave, T-wave, etc;
* how to recognize heart diseases from ECG, for example, atrial fibrillation.

``pipelines`` module contains predefined workflows to

* train a model to detect PQ, QT, QRS segments
* calculate heart rate
* train a model to find probabilities of heart diseases, in particular, atrial fibrillation.

Under the hood these workflows contain actions that load signals, filter them and do complex calculations.

Contents
========
.. toctree::
   :maxdepth: 2
   :titlesonly:

   modules/modules
   tutorials
   api/api


Basic usage
===========

Here is an example of pipeline that loads ECG signals, makes preprocessing and trains a model for 50 epochs:

.. code-block:: python

  train_pipeline = (
    dataset.train
        .pipeline
        .init_model("dynamic", DirichletModel, name="dirichlet",
                    config=model_config)
        .init_variable("loss_history", init=list)
        .load(components=["signal", "meta"], fmt="wfdb")
        .load(components="target", fmt="csv", src=LABELS_PATH)
        .drop_labels(["~"])
        .replace_labels({"N": "NO", "O": "NO"})
        .flip_signals()
        .random_resample_signals("normal", loc=300, scale=10)
        .random_split_signals(2048, {"A": 9, "NO": 3})
        .binarize_labels()
        .train_model("dirichlet", make_data=make_data,
                     fetches="loss", save_to=V("loss_history"), mode="a")
        .run(batch_size=100, shuffle=True, drop_last=True, n_epochs=50)
  )


Installation
============

.. note:: `CardIO` module is in the beta stage. Your suggestions and improvements are very welcome.

.. note:: `CardIO` supports python 3.5 or higher.


Installation as a python package
--------------------------------

With `pipenv <https://docs.pipenv.org/>`_::

    pipenv install git+https://github.com/analysiscenter/cardio.git#egg=cardio

With `pip <https://pip.pypa.io/en/stable/>`_::

    pip3 install git+https://github.com/analysiscenter/cardio.git


After that just import `cardio`::

    import cardio


Installation as a project repository:
--------------------------------------

When cloning repo from GitHub use flag ``--recursive`` to make sure that ``Dataset`` submodule is also cloned::

    git clone --recursive https://github.com/analysiscenter/cardio.git


Citing CardIO
==============
Please cite CardIO in your publications if it helps your research.::

    Khudorozhkov R., Illarionov E., Kuvaev A., Podvyaznikov D. CardIO library for data science research of heart signals. 2017.