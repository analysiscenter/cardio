======
Models
======


Dirichlet model
===============

The model predicts Dirichlet distribution parameters from which class probabilities are sampled. 

How to use
----------

.. code-block :: python

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

.. autoclass:: cardio.models.DirichletModel
    :members:
    :inherited-members:
    :noindex:


Hidden Markov model
===================

Hidden Markov Model is used to annotate ECG signal. This allows to calculate number of
important parameters, important for diagnosing.
This model allows to detect P and T waves; Q, R, S peaks; PQ and ST segments. The model 
has a total of 19 states, the mapping of them to the segments of ECG signal can  be found in ``ecg_batch_tools`` module.

How to use
----------

.. code-block :: python

  HMM_train_ppl = (
    ds.Pipeline()
      .init_model("dynamic", HMModel, "HMM", config=config_train)
      .load(fmt='wfdb', components=["signal", "annotation", "meta"], ann_ext='pu1')
      .wavelet_transform_signal(cwt_scales=[4,8,16], cwt_wavelet="mexh")
      .train_model("HMM", make_data=prepare_batch)
      .run(batch_size=20, shuffle=False, drop_last=False, n_epochs=1, lazy=True)
  )

.. autoclass:: cardio.models.HMModel
    :members:
    :noindex:


FFT model
=========

FFT model learns to classify ECG signals using signal spectrum. At first step it convolves signal with a number of 1D kernels.
Then for each channel it applies fast fourier transform. 
The result is considered as 2D image and is processed with a number of Inception2 blocks
to resulting output, which is a predicted class. See below the model architecture:

.. image:: fft_model.PNG

How to use
----------
We applied this model to arrhythmia prediction from single-lead ECG. Train pipeline we used for the fft model looks as follows:

.. code-block :: python

  train_pipeline = (
    ds.Pipeline()
      .init_model("dynamic", FFTModel, name="fft_model", config=model_config)
      .init_variable("loss_history", init=list)
      .init_variable("true_targets", init=list)
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
           drop_last=True, n_epochs=1, prefetch=0, lazy=True)
  )

.. autoclass:: cardio.models.FFTModel
    :members:
    :noindex:
