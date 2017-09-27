"""Contains ECG Batch class with models' actions."""

from keras.models import Model
from .ecg_batch import EcgBatch
from ..import dataset as ds
from ..models import DirichletModel
from ..models import TripletModel
from ..models import ConvModel
from ..models import FFTModel
from ..models import EcgBaseModel


class ModelEcgBatch(EcgBatch):
    """ECG Batch class with models' actions."""

    def __init__(self, index, preloaded=None, unique_labels=None):
        super().__init__(index, preloaded, unique_labels)

    @ds.model(mode="dynamic")
    def triplet_learn(batch, config=None):#pylint: disable=no-self-argument
        '''
        Define triplet model
        '''
        _ = config
        signal_shape = batch.signal[0].shape
        return TripletModel().build(signal_shape)

    @ds.model(mode="static")
    def triplet_pretrained(pipeline, config=None):  # pylint: disable=no-self-argument
        """Load pretrained Triplet model.

        Parameters
        ----------
        pipeline :
        config : dict
            Model config.

        Returns
        -------
        model : TripletModel
            Loaded model.
        """
        _ = pipeline
        if config is None:
            raise ValueError("Model config must be specified")
        if 'path' not in config.keys():
            raise KeyError("Model config does not contain path")
        return TripletModel().load(fname=config['path'])

    @ds.model(mode="static")
    def triplet_embedding(pipeline, config=None):  # pylint: disable=no-self-argument
        """Load pretrained Triplet model.

        Parameters
        ----------
        pipeline :
        config : dict
            Model config.

        Returns
        -------
        model : TripletModel
            Loaded model.
        """
        _ = pipeline
        if config is None:
            raise ValueError("Model config must be specified")
        if 'path' not in config.keys():
            raise KeyError("Model config does not contain path")
        triplet_model = TripletModel().load(fname=config['path'])
        emb_model = Model(triplet_model.model.layers[0].input, triplet_model.model.layers[config['out_layer']].output)
        return EcgBaseModel(model = emb_model)

    @ds.model(mode="dynamic")
    def conv_model(batch, config=None):#pylint: disable=no-self-argument
        '''
        Define conv model
        '''
        _ = config
        signal_shape = batch.signal[0].shape
        return ConvModel().build(signal_shape)

    @ds.model(mode="static")
    def conv_pretrained(pipeline, config=None):  # pylint: disable=no-self-argument
        """Load pretrained conv model.

        Parameters
        ----------
        pipeline :
        config : dict
            Model config.

        Returns
        -------
        model : TripletModel
            Loaded model.
        """
        _ = pipeline
        if config is None:
            raise ValueError("Model config must be specified")
        if 'path' not in config.keys():
            raise KeyError("Model config does not contain path")
        return ConvModel().load(fname=config['path'])

    @ds.model(mode="static")
    def fft_inception(batch, config=None):#pylint: disable=no-self-argument
        '''
        Define fft model
        '''
        _ = config
        return FFTModel().build()

    @ds.model(mode="static")
    def fft_pretrained(pipeline, config=None):  # pylint: disable=no-self-argument
        """Load pretrained FFT model.

        Parameters
        ----------
        pipeline :
        config : dict
            Model config.

        Returns
        -------
        model : FFTModel
            Loaded model.
        """
        _ = pipeline
        if config is None:
            raise ValueError("Model config must be specified")
        if 'path' not in config.keys():
            raise KeyError("Model config does not contain path")
        return FFTModel().load(config['path'])

    @ds.model(mode="dynamic")
    def dirichlet(batch, config=None):  # pylint: disable=no-self-argument
        """Build dynamic Dirichlet model.

        Parameters
        ----------
        batch : ModelEcgBatch
            First batch to request a model.
        config : dict
            Model config.

        Returns
        -------
        model : DirichletModel
            Built model.
        """
        _ = config
        signal_shape = batch.signal[0].shape[1:]
        if len(signal_shape) != 2:
            raise ValueError("Dirichlet model expects 2-D signals")
        target_shape = batch.target.shape[1:]
        if len(target_shape) != 1:
            raise ValueError("Dirichlet model expects 1-D targets")
        classes = batch.label_binarizer.classes_
        return DirichletModel().build(signal_shape, target_shape, classes)

    @ds.model(mode="static")
    def dirichlet_pretrained(pipeline, config=None):  # pylint: disable=no-self-argument
        """Load pretrained Dirichlet model.

        Parameters
        ----------
        pipeline :
        config : dict
            Model config.

        Returns
        -------
        model : DirichletModel
            Loaded model.
        """
        _ = pipeline
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
        """Run a single gradient update for a model with given model_name.

        Parameters
        ----------
        model_name : str
            Model name.
        *args, **kwargs : misc
            Any additional model.train_on_batch argments.

        Returns
        -------
        result : misc
            model.train_on_batch output.
        """
        model = self.get_model_by_name(model_name)
        return model.train_on_batch(self, *args, **kwargs)

    @ds.action
    def test_on_batch(self, model_name, *args, **kwargs):
        """Get model loss for a single batch.

        Parameters
        ----------
        model_name : str
            Model name.
        *args, **kwargs : misc
            Any additional model.test_on_batch argments.

        Returns
        -------
        result : misc
            model.test_on_batch output.
        """
        model = self.get_model_by_name(model_name)
        return model.test_on_batch(self, *args, **kwargs)

    @ds.action
    def predict_on_batch(self, model_name, *args, **kwargs):
        """Get model predictions for a single batch.

        Parameters
        ----------
        model_name : str
            Model name.
        *args, **kwargs : misc
            Any additional model.predict_on_batch argments.

        Returns
        -------
        result : misc
            model.predict_on_batch output.
        """
        model = self.get_model_by_name(model_name)
        return model.predict_on_batch(self, *args, **kwargs)

    @ds.action
    def save_model(self, model_name, *args, **kwargs):
        """Save model weights.

        Parameters
        ----------
        model_name : str
            Model name.
        *args, **kwargs : misc
            Any additional model.save.

        Returns
        -------
        result : misc
            model.save output.
        """
        model = self.get_model_by_name(model_name)
        return model.save(self, *args, **kwargs)

    @ds.action
    def load_model(self, model_name, *args, **kwargs):
        """Load model weights.

        Parameters
        ----------
        model_name : str
            Model name.
        *args, **kwargs : misc
            Any additional model.load argments.

        Returns
        -------
        result : misc
            model.load output.
        """
        model = self.get_model_by_name(model_name)
        return model.load(self, *args, **kwargs)
