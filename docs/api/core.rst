=====
Core
=====


EcgBatch
========

.. autoclass:: cardio.EcgBatch
	:show-inheritance:

Methods
-------

Input/output methods
^^^^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.EcgBatch.load
	.. automethod:: cardio.EcgBatch.dump
	.. automethod:: cardio.EcgBatch.show_ecg

Batch processing
^^^^^^^^^^^^^^^^
	.. automethod:: cardio.EcgBatch.deepcopy
	.. automethod:: cardio.EcgBatch.merge

Versatile components processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.EcgBatch.apply_transform
	.. automethod:: cardio.EcgBatch.apply_to_each_channel
	
Labels processing
^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.EcgBatch.drop_labels
	.. automethod:: cardio.EcgBatch.keep_labels
	.. automethod:: cardio.EcgBatch.rename_labels
	.. automethod:: cardio.EcgBatch.binarize_labels

Channels processing
^^^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.EcgBatch.drop_channels
	.. automethod:: cardio.EcgBatch.keep_channels
	.. automethod:: cardio.EcgBatch.rename_channels

Signal processing
^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.EcgBatch.convolve_signals
	.. automethod:: cardio.EcgBatch.band_pass_signals
	.. automethod:: cardio.EcgBatch.drop_short_signals
	.. automethod:: cardio.EcgBatch.flip_signals
	.. automethod:: cardio.EcgBatch.slice_signals
	.. automethod:: cardio.EcgBatch.split_signals
	.. automethod:: cardio.EcgBatch.random_split_signals
	.. automethod:: cardio.EcgBatch.unstack_signals
	.. automethod:: cardio.EcgBatch.resample_signals
	.. automethod:: cardio.EcgBatch.random_resample_signals

Complex ECG processing
^^^^^^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.EcgBatch.fft
	.. automethod:: cardio.EcgBatch.ifft
	.. automethod:: cardio.EcgBatch.rfft
	.. automethod:: cardio.EcgBatch.irfft
	.. automethod:: cardio.EcgBatch.spectrogram
	.. automethod:: cardio.EcgBatch.wavelet_transform_signal
	.. automethod:: cardio.EcgBatch.calc_ecg_parameters


EcgDataset
==========

.. autoclass:: cardio.EcgDataset
	:show-inheritance: