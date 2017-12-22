====
Core
====

CardIO's core classes are EcgBatch and EcgDataset. They are responsible for
storing of ECGs, batch generation and applying actions to batches.

EcgBatch
---------

EcgBatch is a class that defines how to store ECG data and contains actions
that can be applied to ECG in preprocessing stage. 

Attributes of EcgBatch:

* ``signal``, stores ECG signals in numpy array
* ``annotation``, array of dicts with different types of annotations, e.g. array of R-peaks
* ``meta``, array of dicts with metadata about ECG records, e.g. signal frequency
* ``target``, array of labels assigned to ECG records
* ``unique_labels``, array of all possible target labels in dataset.

Actions of EcgBatch allows to:

* load ECG records from wfdb, DICOM, EDF, wav or blosc format
* segment, flip and resample signals
* filter signals 
* allocate PQ, QT, QRS segments
* dump results.

Actions can be arranged in a single workflow like the following one which loads
data and flips ECGs whose R-peaks are directed downwards:

.. code-block:: python

  import cardio.dataset as ds

  example_workflow = (
      ds.Pipeline()
        .load(fmt="wfdb", components=["signal", "meta"])
        .flip_signals()
  )


EcgDataset
----------

EcgDataset helps to conveniently create a list of ECG indices and generate batches
(small subsets of data) of default type EcgBatch. 

CardIO generates batches trought a `Dataset <https://github.com/analysiscenter/dataset>`_ library. To initialize this process we need to create a sequence of data item ids, e.g. using names of files in specific folder:

.. code-block:: python

  import cardio.dataset as ds
  index = ds.FilesIndex(path="../cardio/tests/data/*.hea", no_ext=True, sort=True)

Then we specify type of batches we want to generate, e.g. EcgBatch:

.. code-block:: python  

  from cardio import EcgBatch
  eds = ds.Dataset(index, batch_class=EcgBatch)

EcgDataset helps to get the same result in a shorter way:

.. code-block:: python  

  from cardio import EcgDataset
  eds = EcgDataset(path="../cardio/tests/data/*.hea", no_ext=True, sort=True)

Now we can call ``eds.next_batch(batch_size=3)`` to generate batches manually and apply any action of EcgBatch to them. But it is more convenient to define the 
workflow (sequence of actions we want to apply) first and then run automatic batch generation and processing once for the whole dataset:

.. code-block:: python

  import cardio.dataset as ds

  example_workflow = (
      ds.Pipeline()
        .load(fmt="wfdb", components=["signal", "meta"])
        .flip_signals()
  )

  (eds >> example_workflow).run(batch_size=2, n_epochs=1)


API
---
See :doc:`Core API <../api/core>`