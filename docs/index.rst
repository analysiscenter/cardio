===================================
Welcome to CardIO's documentation!
===================================
`CardIO` is a library designed to build end-to-end machine learning models for deep research of electrocardiograms.

Main features:

* load and save signal in a number of formats
* resample, crop and flip signal
* filter signal
* allocate PQ, QT, QRS segments
* calculate heart rate and find other standard ECG characteristics
* apply complex transformations like fft and wavelets, or any other custom functions.
* recognize heart diseases from ECG
* efficiently work with large datasets that do not even fit into memory
* easily arrange new custom actions into pipelines
* do end-to-end ECG processing
* build, train and test custom models for deep research

… and do everything under a single API.

The library is based on `Dataset <https://github.com/analysiscenter/dataset>`_. We suggest to read Dataset's documentation to learn more, however, you may skip it for the first reading.

CardIO has three modules: :doc:`batch <./api/cardio.batch>`, :doc:`models <./api/cardio.models>` and :doc:`pipelines <api/cardio.pipelines>`.

Module batch contains low-level actions for ECG processing. Actions are included in EcgBatch class that also defines how to store ECGs. From these actions you can build new pipelines. You can also write custom action and include it in EcgBatch.

In models we provide several models that were elaborated to learn the most important problems in ECG:

* how to recognize specific features of ECG like R-peaks, P-wave, T-wave
* how to recognize heart diseases from ECG, for example - atrial fibrillation.

Module pipelines contains high-level methods that build pipelines for model training and prediction, preprocessing, etc.


Contents
========
.. toctree::
   :maxdepth: 2
   :titlesonly:

   intro/intro
   api/cardio


Basic usage
===========

Here is an example of pipeline that loads ECG signals, makes some preprocessing and learns model over 50 epochs:

.. code-block :: python

  train_ppl = (
    dtst.train
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
        .run(batch_size=100, shuffle=True, drop_last=True,
             n_epochs=50)
)

As a result of this pipeline one obtains a trained model.

Installation
============

With `pipenv <https://docs.pipenv.org/>`_::

    pipenv install git+https://github.com/analysiscenter/cardio.git#egg=cardio

With `pip <https://pip.pypa.io/en/stable/>`_::

    pip3 install git+https://github.com/analysiscenter/cardio.git


After that just import `cardio`::

    import cardio


.. note:: `CardIO` module is in the beta stage. Your suggestions and improvements are very welcome.

.. note:: `CardIO` supports python 3.5 or higher.

.. note:: When cloning repo from GitHub use flag ``--recursive`` to make sure that you clone ``Dataset`` submodule as well.

Citing CardIO
==============
Please cite CardIO in your publications if it helps your research.
