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
	.. automethod:: cardio.EcgBatch.standardize
	
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
	.. automethod:: cardio.EcgBatch.reorder_channels
	.. automethod:: cardio.EcgBatch.convert_units

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

Fourier-based transformations
"""""""""""""""""""""""""""""
	.. automethod:: cardio.EcgBatch.fft
	.. automethod:: cardio.EcgBatch.ifft
	.. automethod:: cardio.EcgBatch.rfft
	.. automethod:: cardio.EcgBatch.irfft
	.. automethod:: cardio.EcgBatch.spectrogram

Wavelet-based transformations
"""""""""""""""""""""""""""""
	.. automethod:: cardio.EcgBatch.dwt
	.. automethod:: cardio.EcgBatch.idwt
	.. automethod:: cardio.EcgBatch.wavedec
	.. automethod:: cardio.EcgBatch.waverec
	.. automethod:: cardio.EcgBatch.pdwt
	.. automethod:: cardio.EcgBatch.cwt

Other methods
"""""""""""""
	.. automethod:: cardio.EcgBatch.calc_ecg_parameters

EcgDataset
==========

.. autoclass:: cardio.EcgDataset
	:show-inheritance: