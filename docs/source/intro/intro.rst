============
Introduction
============

This section describes briefly capabilities of the CardIO framework.

Contents
========

.. toctree::
   :maxdepth: 2

   batch
   models
   pipelines
   tutorials


EcgBatch
--------
The main class the CardIO is EcgBatch. It contains the I/O and preprocessing :func:`actions <dataset.action>` that allow to load and prepare data for the modeling.

.. code-block :: python

  from cardio import EcgBatch
  from dataset import FilesIndex, Dataset
  ecg_index = FilesIndex(path='path/to/ecg/*', no_ext=True) # set up the index
  dicomset = Dataset(index=ecg_index, batch_class=EcgBatch) # init the dataset of dicom files

Models
------
This module contain model suited to classify whether ECG signal is normal or pathological, to annotate segments of the signal (e.g., P-wave).

.. code-block:: python

  dirichlet_train_ppl = (
  ds.Pipeline()
    .init_model("dynamic", DirichletModel, name="dirichlet", config=model_config)
    .init_variable("loss_history", init=list)
    .load(components=["signal", "meta"], fmt="wfdb")
    .load(components="target", fmt="csv", src=LABELS_PATH)
    .drop_labels(["~"])
    .replace_labels({"N": "NO", "O": "NO"})
    .flip_signals()
    .random_resample_signals("normal", loc=300, scale=10)
    .random_split_signals(2048, {"A": 9, "NO": 3})
    .binarize_labels()
    .train_model("dirichlet", make_data=concatenate_ecg_batch,
                 fetches="loss", save_to=V("loss_history"), mode="a")
    .run(batch_size=BATCH_SIZE, shuffle=True, drop_last=True, n_epochs=N_EPOCH, lazy=True)
  )

Pipelines
---------
Pipelines were designed to ease usage of exhisting models make final code simpler. 

.. code-block:: python

  from cardio.pipelines import hmm_predict_pipeline
  res = (data >> hmm_predict_pipeline(model_path)).run()

Under the hood this function contains a list of actions:

.. code-block:: python

  template_hmm_predict = (
  ds.Pipeline()
    .init_model("static", HMModel, "HMM", config=config_predict)
 	.init_variable("batch", init_on_each_run=list)
 	.load(fmt="wfdb", components=["signal", "annotation", "meta"], ann_ext="pu1")
 	.wavelet_transform_signal(cwt_scales=[4,8,16], cwt_wavelet="mexh")
 	.predict_model("HMM", make_data=prepare_batch, save_to=ds.B("_temp"), mode='w')
	.write_to_annotation("hmm_annotation", "_temp")
 	.calc_ecg_parameters()
    .update_variable("batch", ds.F(get_batch), mode='e')
    .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
  )
  res = (data >> template_hmm_predict).run()  

Tutorials
---------
In this section you can find links to the tutorails on how to use CardIO and its' modules.
