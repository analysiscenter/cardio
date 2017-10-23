# Convolutional model

Convolutional model learns to classify ECG signal. 
It consists of a series of convolution blocks and results in predicted class. 


## How to use
We train this model on ECG signals, preprocessed with a [triplet model](triplet_model.md). 
From an intermediate layer of the triplet model we obtain signal embedding that becomes an input for the convolution model. 
Train pipeline we used for the convolutional model looks as follows:
```python
conv_train_pipeline = (ds.Pipeline(config={'triplet_embedding': config_tr})
                         .init_model('triplet_embedding')
                         .load(fmt="wfdb", components=["signal", "meta"])
                         .load(src="REFERENCE.csv", fmt="csv", components="target")
                         .drop_labels(["~"])
                         .replace_labels({"N": "NO", "O": "NO"})
                         .split_signals(3000, 3000)
                         .binarize_labels()
                         .ravel()
                         .tile(3)
                         .apply(np.transpose, axes=[2, 1, 0])
                         .predict_on_batch('triplet_embedding', inplace=True)
                         .slice_signal(slice(1))
                         .apply(np.squeeze, axis=0)
                         .train_on_batch('conv_model', metrics=f1_score, average='macro'))
```