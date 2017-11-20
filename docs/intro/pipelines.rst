=========
Pipelines
=========

This module contains pipelines that we used to train models and make predictions. You can use them as-is to train similar models or 
adjust them in order to get better perfomance.

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
* dirichlet_train_pipeline
* dirichlet_predict_pipeline
* hmm_preprocessing_pipeline
* hmm_train_pipeline
* hmm_predict_pipeline

API
---
See :doc:`Pipelines API <../api/cardio.pipelines>`