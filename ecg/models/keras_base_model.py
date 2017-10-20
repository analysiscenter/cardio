"""Model and model tools for ECG with Keras backend"""

import numpy as np
from .base_model import BaseModel

class KerasBaseModel(BaseModel):
    '''
    Contains model and history.
    '''
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.hist = {'train_loss': [], 'train_metric': [],
                     'val_loss': [], 'val_metric': []}

    def train_on_batch(self, batch, metrics=None, **kwagrs):
        '''
        Train model

        Parameters
        ----------
        metrics : string
            Metrics from list of Keras metrics to be evaluated by the model during training.
        **kwargs : keyword arguments
            Any kwargs that should be passed to metrics
        '''
        train_x = np.array(list(batch.signal))

        res = self.model.train_on_batch(train_x, batch.target)
        self.hist['train_loss'].append(res)

        if metrics is not None:
            pred = self.model.predict_on_batch(train_x)
            y_pred = (pred > 0.5).astype(int)
            self.hist['train_metric'].append(metrics(batch.target, y_pred, **kwagrs))
        else:
            self.hist['train_metric'].append(0.)
        return batch

    def test_on_batch(self, batch, metrics=None, **kwagrs):
        '''
        Test model

        Parameters
        ----------
        metrics : string or None
            Metrics from list of Keras metrics to be evaluated by the model during testing. Default None.
        **kwargs : keyword arguments
            Any kwargs that should be passed to metrics
        '''
        test_x = np.array(list(batch.signal))
        res = self.model.test_on_batch(test_x, batch.target)
        self.hist['val_loss'].append(res)
        if metrics is not None:
            pred = self.model.predict_on_batch(test_x)
            y_pred = (pred > 0.5).astype(int)
            self.hist['val_metric'].append(metrics(batch.target, y_pred, **kwagrs))
        else:
            self.hist['val_metric'].append(0.)
        return batch

    def predict_on_batch(self, batch, inplace=False):#pylint: disable=arguments-differ
        '''
        Predict data

        Parameters
        ----------
        inplace : bool
            If True predictions replace signal, otherwise are saved to pipeline component.
        '''
        test_x = np.array(list(batch.signal))
        if inplace:
            batch.signal = list(self.model.predict_on_batch(test_x))
            batch.signal.append([])
            batch.signal = np.array(batch.signal)[:-1]
        else:
            predict = batch.pipeline.get_variable("prediction")
            predict.append(self.model.predict_on_batch(test_x))
        return batch

    def model_summary(self):
        '''
        Print model layers
        '''
        print(self.model.summary())
        return self

    def save(self, fname):#pylint: disable=arguments-differ
        '''
        Save keras model

        Parameters
        ----------
        fname : string
            Filename to which model is saved
        '''
        self.model.save(fname)
        return self

    def load(self, fname):#pylint: disable=arguments-differ
        '''
        Load keras model

        Parameters
        ----------
        fname : string
            Filename from which model is loaded
        '''
        self.model.load(fname)
        return self
