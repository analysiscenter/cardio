=========
Pipelines
=========

This module contains pipelines that we used for model training and
predicting. Ypu can use them as-is to train similar models or 
adjust them in order to find better solution.

How to use
----------

There are 3 simple steps in using pipelines. First, we import desired pipeline, e.g. dirichlet_train
::
  from cardio.pipelines import dirichlet_train

Second, we specify its parameters, e.g. path to data:
::
  pipeline = dirichlet_train('path')

Third, we pass dataset to pipeline and run caclulation:
::
  (dataset >> pipeline).run(batch_size=100, n_epochs=10)

Result is typically a trained model or some values stored in pipeline variable (e.g. model predicitons).

Available pipelines
-------------------

At this moment the module contains following pipelines:

* dirichlet_train
* dirichlet_prediction
* hmm_train
* hmm_predicition
* show_ecg_segments
