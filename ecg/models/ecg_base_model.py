"""Model and model tools for ECG"""

import numpy as np
from .base_model import BaseModel
from keras.models import model_from_yaml

class EcgBaseModel(BaseModel):
    '''
    Contains model, history and additional pretrained_model.
    '''
    def __init__(self):
        super().__init__()
        self.model = None
        self.hist = {'train_loss': [], 'train_metric': [],
                     'val_loss': [], 'val_metric': []}

    def train_on_batch(self, batch, metrics=None, **kwagrs):
        '''
        Train model
        '''
        train_x = np.array(list(batch.signal))

        res = self.model.train_on_batch(train_x, batch.target)
        self.hist['train_loss'].append(res)

        loss = batch.pipeline.get_variable("loss")
        loss.append(res)

        if metrics is not None:
            pred = self.model.predict_on_batch(train_x)
            y_pred = (pred > 0.5).astype(int)
            self.hist['train_metric'].append(metrics(batch.target, y_pred, **kwagrs))
        else:
            self.hist['train_metric'].append(0.)
        return self

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
        return self

    def predict_on_batch(self, batch):
        '''
        Predict data
        '''
        test_x = np.array(list(batch.signal))
        predict = batch.pipeline.get_variable("prediction")
        predict.append(self.model.predict_on_batch(test_x))
        return self

    def model_summary(self):
        '''
        Print model layers
        '''
        print(self.model.summary())
        return self

    @ds.action
    def save(self, fname):
        '''
        Save model layers and weights
        '''
        self.model.save_weights(fname)
        yaml_string = self.model.to_yaml()
        fout = open(fname + ".layers", "w")
        fout.write(yaml_string)
        fout.close()
        return self

    @ds.action
    def load(self, model_name, weights_only=True):
        '''
        Load model layers and weights
        '''
        if not weights_only:
            fin = open(fname + ".layers", "r")
            yaml_string = fin.read()
            fin.close()
            self.model = model_from_yaml(yaml_string)
        self.model.load_weights(fname)
        return self