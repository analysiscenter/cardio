=====
Batch
=====

This module stores batch class for ECG and various actions on ECG batch.

EcgBatch
---------

EcgBatch is the main class that defines how to store ECG data and contains actions
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

If you use pipeline you don't have a direct access to the EcgBatch objects, but you can create an object with ``next_batch`` method of :func:`Index <dataset.Index>`:
.. code-block:: python

  from cardio import EcgBatch
  import cardio.dataset as ds

  index = ds.FilesIndex(path="path/to/data/", no_ext=True, sort=True)
  dtst = ds.Dataset(index, batch_class=EcgBatch)

  template_ppln = (
      ds.Pipeline()
        .load(fmt="wfdb", components=["signal", "meta"])
        .wavelet_transform_signal(cwt_scales=[4,8,16], cwt_wavelet="mexh")
  )

  ppln = (dtst >> template_ppln)
  batch = ppln.next_batch(10)  # create instance of EcgBatch class with size 10

Other capabilities
------------------

Batch module also contains several submodules that support functionality of EcgBacth and not included in API.
You can find brief description of those submodules below. To learn more see the source code.

ecg_batch_tools
~~~~~~~~~~~~~~~

Contains general methods for signal processing that are exploited in EcgBatch actions.
Most of those methods support multiprocessing and are written with ``numba``. 
If you want to explore those methods or use them outside EcgBatch simply write
.. code-block:: python

  from cardio.batch import ecg_batch_tools as bt


kernels
~~~~~~~

This submodule contains kernel generation functions for signal convolution.


utils
~~~~~

Miscellaneous ECG Batch utils.

API
===
See :doc:`Batch API <../api/cardio.batch>`