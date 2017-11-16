Batch
=====

.. autoclass:: cardio.batch.EcgBatch
	:show-inheritance:

Methods
-------

Input/output methods
^^^^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.batch.EcgBatch.load
	.. automethod:: cardio.batch.EcgBatch.dump
	.. automethod:: cardio.batch.EcgBatch.show_ecg

Batch modifications
^^^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.batch.EcgBatch.update
	.. automethod:: cardio.batch.EcgBatch.merge
	.. automethod:: cardio.batch.EcgBatch.apply_transform

Label processing
^^^^^^^^^^^^^^^^
	.. automethod:: cardio.batch.EcgBatch.drop_labels
	.. automethod:: cardio.batch.EcgBatch.keep_labels
	.. automethod:: cardio.batch.EcgBatch.replace_labels
	.. automethod:: cardio.batch.EcgBatch.binarize_labels

Signal processing
^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.batch.EcgBatch.convolve_signals
	.. automethod:: cardio.batch.EcgBatch.band_pass_signals
	.. automethod:: cardio.batch.EcgBatch.drop_short_signals
	.. automethod:: cardio.batch.EcgBatch.flip_signals
	.. automethod:: cardio.batch.EcgBatch.ravel
	.. automethod:: cardio.batch.EcgBatch.slice_signal
	.. automethod:: cardio.batch.EcgBatch.split_signals
	.. automethod:: cardio.batch.EcgBatch.random_split_signals
	.. automethod:: cardio.batch.EcgBatch.resample_signals
	.. automethod:: cardio.batch.EcgBatch.random_resample_signals

Complex ECG processing
^^^^^^^^^^^^^^^^^^^^^^
	.. automethod:: cardio.batch.EcgBatch.wavelet_transform_signal
	.. automethod:: cardio.batch.EcgBatch.calc_ecg_parameters
