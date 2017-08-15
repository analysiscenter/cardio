"""Contains ECG Batch class with models actions."""

from .. import dataset as ds
from .ecg_batch import EcgBatch
from ..models.beta_model import BetaModel


class ModelEcgBatch(EcgBatch):
    def __init__(self, index, preloaded=None, unique_labels=None):
        super().__init__(index, preloaded, unique_labels)

    @ds.model()
    def beta():
        return BetaModel().build()

    @ds.action(singleton=True)
    def train_on_batch(self, model_name, *args, **kwargs):
        model = self.get_model_by_name(model_name)
        return model.train_on_batch(self, *args, **kwargs)

    @ds.action(singleton=True)
    def test_on_batch(self, model_name, *args, **kwargs):
        model = self.get_model_by_name(model_name)
        return model.test_on_batch(self, *args, **kwargs)

    @ds.action(singleton=True)
    def predict_on_batch(self, model_name, *args, **kwargs):
        model = self.get_model_by_name(model_name)
        return model.predict_on_batch(self, *args, **kwargs)
