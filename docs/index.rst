===================================
Welcome to CardIO's documentation!
===================================
`CardIO` is a library designed to build end-to-end machine learning models for deep research of electrocardiograms and provides ready-to-use methods for heart diseases detection

Main features:

* calculate heart beat rate and find other standard ECG characteristics
* recognize heart diseases from ECG
* efficiently work with large datasets that do not even fit into memory
* easily arrange new custom actions into pipelines
* do end-to-end ECG processing
* build, train and test custom models for deep research

… and do everything under a single API.

The library is based on `Dataset <https://github.com/analysiscenter/dataset>`_. We suggest to read Dataset's documentation to learn more, however, you may skip it for the first reading.

CardIO has three modules: :doc:`batch <./api/cardio.ecg_batch>`, :doc:`models <./api/cardio.models>` and :doc:`pipelines <api/cardio.pipelines>`.

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

  model_train_template = (
      ds.Pipeline()
      .init_model("dynamic", FFTModel, name="fft_model", config=model_config)
      .init_variable("loss_history", init=list)
      .load(fmt="wfdb", components=["signal", "meta"])
      .load(src="/notebooks/data/ECG/training2017/REFERENCE.csv",
            fmt="csv", components="target")
      .drop_labels(["~"])
      .replace_labels({"N": "NO", "O": "NO"})
      .random_resample_signals("normal", loc=300, scale=10)
      .drop_short_signals(4000)
      .split_signals(3000, 3000)
      .binarize_labels()
      .apply(np.transpose , axes=[0, 2, 1])
      .ravel()
      .get_targets('true_targets')
      .train_model('fft_model', make_data=make_data,
                   save_to=V("loss_history"), mode="a")
      .run(batch_size=300, shuffle=True,
           drop_last=True, n_epochs=50, prefetch=0, lazy=True))


After linking this pipeline to the dataset with the signals you will obtain a pipeline with trained model and loss values kept in pipiline variable `loss_history`:

.. code-block :: python

  model_train_pipeline = (dataset >> model_train_template).run()

      
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


Citing CardIO
==============
Please cite CardIO in your publications if it helps your research.
