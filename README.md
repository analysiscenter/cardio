# ECG

`ECG` is a module based on `Dataset`, designed by Data Analysis Center to handle ECG data.
The main component is an `EcgBatch` class, which incorporates all the methods that can be applied to
the ECG data. Some methods support inbatch parallelism.

## Basic usage
```
ecg_batch = EcgBatch(ind)
	       .load()
	       .generate_subseqs(500,500)
```
The result of this code is an object of class `EcgBatch`, containing processed data.

##List of methods:
```
ecg_batch.load()
ecg_batch.dump()
ecg_batch.generate_subseqs()
```

## Installation
To be implemented