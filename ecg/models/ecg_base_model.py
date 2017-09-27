"""Model and model tools for ECG"""

import numpy as np
from .base_model import BaseModel

class EcgBaseModel(BaseModel):
    '''
    Contains model, history and additional pretrained_model.
    '''
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.hist = {'train_loss': [], 'train_metric': [],
                     'val_loss': [], 'val_metric': []}

    def train_on_batch(self, batch, metrics=None, **kwagrs):
        '''
        Train model
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
        Validate model
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
        '''
        self.model.save(fname)
        return self

    def load(self, fname):#pylint: disable=arguments-differ
        '''
        Load keras model
        '''
        self.model.load(fname)
        return self
