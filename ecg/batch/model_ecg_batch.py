"""Contains ECG Batch class with models actions."""

from .. import dataset as ds
from .ecg_batch import EcgBatch
from ..models.dirichlet_model import DirichletModel


class ModelEcgBatch(EcgBatch):
    def __init__(self, index, preloaded=None, unique_labels=None):
        super().__init__(index, preloaded, unique_labels)

    @ds.model(mode="dynamic")
    def dirichlet(batch, config=None):  # pylint: disable=no-self-argument
        signal_shape = batch.signal[0].shape[1:]
        if len(signal_shape) != 2:
            raise ValueError("Dirichlet model expects 2-D signals")
        target_shape = batch.target.shape[1:]
        if len(target_shape) != 1:
            raise ValueError("Dirichlet model expects 1-D targets")
        classes = batch.label_binarizer.classes_
        return DirichletModel().build(signal_shape, target_shape, classes)

    @ds.model(mode="dynamic")
    def dirichlet_pretrained(batch, config=None):  # pylint: disable=no-self-argument
        if config is None:
            raise ValueError("Model config must be specified")
        paths = ("graph_path", "checkpoint_path", "classes_path")
        diff = set(paths) - config.keys()
        if diff:
            raise KeyError("Model config does not contain {}".format(", ".join(sorted(diff))))
        args = [config[path] for path in paths]
        return DirichletModel().load(*args)

    @ds.action(use_lock="train_lock")
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
