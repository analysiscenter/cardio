# Release 0.2.0

## Major Features and Improvements
* ``load`` method now supports new signal formats:
	* DICOM
	* EDF
	* wav
* ``meta`` component structure has changed - now it always contains a number of predefined keys.
* Added channels processing methods:
	* `EcgBatch.keep_channels`
	* `EcgBatch.drop_channels`
	* `EcgBatch.rename_channels`
* Added `apply_to_each_channel` method.
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
* `apply_transform` method's signature has changed.
* `update` method has been deleted.
* `replace_labels` method has been renamed to `rename_labels`.
* `slice_signal` method has been renamed to `slice_signals`.
* `ravel` method has been renamed to `unstack_signals`.


# Release 0.1.0

Initial release of CardIO.
