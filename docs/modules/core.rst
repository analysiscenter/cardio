====
Core
====

CardIO's core classes are ``EcgBatch`` and ``EcgDataset``. They are responsible for
storing of ECGs, batch generation and applying actions to batches.

EcgBatch
---------

``EcgBatch`` is a class that defines how to store ECG data and contains actions
that can be applied to ECG in preprocessing stage. 

Attributes of ``EcgBatch``:

* ``signal``, stores ECG signals in numpy array
* ``annotation``, array of dicts with different types of annotations, e.g. array of R-peaks
* ``meta``, array of dicts with metadata about ECG records, e.g. signal frequency
* ``target``, array of labels assigned to ECG records
* ``unique_labels``, array of all possible target labels in dataset

Actions of ``EcgBatch`` allow to:

* load ECG records from wfdb, DICOM, EDF, wav or blosc format
* segment, flip and resample signals
* filter signals 
* detect PQ, QT, QRS segments
* dump results

To learn more about actions refer to the `tutorial <https://github.com/analysiscenter/cardio/blob/master/tutorials/I.CardIO.ipynb>`_.

EcgDataset
----------

``EcgDataset`` helps to conveniently create a list of ECG indices and generate batches
(small subsets of data) of default type ``EcgBatch``. 

CardIO generates batches trought a `Batchflow <https://github.com/analysiscenter/batchflow>`_ library. To initialize this process we need to create a sequence of data item ids, e.g. using names of files in specific folder:

.. code-block:: python

  import cardio.batchflow as bf
  index = bf.FilesIndex(path="../cardio/tests/data/*.hea", no_ext=True, sort=True)

Then we specify type of batches we want to generate, e.g. ``EcgBatch``:

.. code-block:: python  

  from cardio import EcgBatch
  eds = bf.Dataset(index, batch_class=EcgBatch)

``EcgDataset`` helps to get the same result in a shorter way:

.. code-block:: python  

  from cardio import EcgDataset
  eds = EcgDataset(path="../cardio/tests/data/*.hea", no_ext=True, sort=True)

Now we can call ``EcgDataset.next_batch`` with specified ``batch_size`` argument to generate batches and process them using actions of ``EcgBatch``. 


API
---
See :doc:`Core API <../api/core>`