# Release 0.3.0

## Major Features and Improvements
* `load` method now supports Schiller XML format
* Added channels processing methods:
	* `EcgBatch.reorder_channels`
	* `EcgBatch.convert_units`


# Release 0.2.0

## Major Features and Improvements
* `load` method now supports new signal formats:
	* DICOM
	* EDF
	* WAV
* `meta` component structure has changed - now it always contains a number of predefined keys.
* Added channels processing methods:
	* `EcgBatch.keep_channels`
	* `EcgBatch.drop_channels`
	* `EcgBatch.rename_channels`
* Added `apply_to_each_channel` method.
* Added `standardize` method.
* Added complex ECG transformations:
	* Fourier-based transformations:
		* `EcgBatch.fft`
		* `EcgBatch.ifft`
		* `EcgBatch.rfft`
		* `EcgBatch.irfft`
		* `EcgBatch.spectrogram`
	* Wavelet-based transformations:
		* `EcgBatch.dwt`
		* `EcgBatch.idwt`
		* `EcgBatch.wavedec`
		* `EcgBatch.waverec`
		* `EcgBatch.pdwt`
		* `EcgBatch.cwt`

## Breaking Changes to the API
* Changed signature of the following methods:
	* `EcgBatch.apply_transform`
	* `EcgBatch.show_ecg`
	* `EcgBatch.calc_ecg_parameters`
* Changed signature of the following pipelines:
	* `dirichlet_train_pipeline`
	* `dirichlet_predict_pipeline`
	* `hmm_preprocessing_pipeline`
	* `hmm_train_pipeline`
	* `hmm_predict_pipeline`
* `wavelet_transform` method has been deleted.
* `update` method has been deleted.
* `replace_labels` method has been renamed to `rename_labels`.
* `slice_signal` method has been renamed to `slice_signals`.
* `ravel` method has been renamed to `unstack_signals`.


# Release 0.1.0

Initial release of CardIO.
