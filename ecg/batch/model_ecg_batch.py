"""Contains ECG Batch class with models actions."""

from .. import dataset as ds
from .ecg_batch import EcgBatch
from ..models.beta_model import BetaModel


class ModelEcgBatch(EcgBatch):
    def __init__(self, index, preloaded=None, unique_labels=None):
        super().__init__(index, preloaded, unique_labels)

    @ds.model(mode="dynamic")
    def beta(batch):
        signal_shape = batch.signal[0].shape[1:]
        if len(signal_shape) != 2:
            raise ValueError("Beta model expects 2-D signals")
        target_shape = batch.target.shape[1:]
        if len(target_shape) != 1:
            raise ValueError("Beta model expects 1-D targets")
        return BetaModel().build(signal_shape, target_shape)

    @ds.action(use_lock="beta_train")
    def train_on_batch(self, model_name, *args, **kwargs):
        model = self.get_model_by_name(model_name)
        return model.train_on_batch(self, *args, **kwargs)

    @ds.action
    def test_on_batch(self, model_name, *args, **kwargs):
        model = self.get_model_by_name(model_name)
        return model.test_on_batch(self, *args, **kwargs)

    @ds.action
    def predict_on_batch(self, model_name, *args, **kwargs):
        model = self.get_model_by_name(model_name)
        return model.predict_on_batch(self, *args, **kwargs)
