=====
Batch
=====

This module stores batch class for ECG and various actions on ECG batch.

ecg_batch
---------

Contains class EcgBatch that defines how to store ECG data and lists actions
that can be applied to ECG in preprocessing stage. 

Attributes of EcgBatch:

* ``signal``, stores ECG signals in numpy array
* ``annotation``, array of dicts with different types of annotations, e.g. array of R-peaks
* ``meta``, array of dicts with metadata about ECG records, e.g. signal frequency
* ``target``, array of labels assigned to ECG records
* ``unique_labels``, array of all possible target labels in dataset.

Actions of EcgBatch allows e.g.:

* load ECG records from wfdb or blosc format
* segment, flip and resample signals
* filter signals 
* allocate PQ, QT, QRS segments
* dump results.

More detailed API see `here <https://analysiscenter.github.io/ecg/index.html>`_.


ecg_batch_tools
---------------

Contains general methods for signal processing that are exploited in EcgBatch actions.


kernels
-------

Contains kernel generation functions for signal convolution.


utils
-----

Miscellaneous ECG Batch utils.