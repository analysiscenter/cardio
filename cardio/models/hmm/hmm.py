""" HMModel """

import numpy as np
import dill

from ...batchflow.models.base import BaseModel


def prepare_hmm_input(batch, model, features, channel_ix):
    """Concatenate selected channel ``channel_ix`` of the batch attribute
    ``features``.

    Parameters
    ----------
    batch : EcgBatch
        Batch to concatenate.
    model : BaseModel
        A model to get the resulting arguments.
    features : str
        Specifies batch attribute that contains features for ``HMModel``.
    channel_ix : int
        Index of channel, which data should be used in training and
        predicting.

    Returns
    -------
    kwargs : dict
        Named argments for model's train or predict method. Has the
        following structure:
        "X" : 2-D ndarray
            Features concatenated along -1 axis and transposed.
        "lengths" : list
            List of lengths of individual feature arrays along -1 axis.
    """
    _ = model
    hmm_features = getattr(batch, features)
    x = np.concatenate([features[channel_ix].T for features in hmm_features])
    lengths = [features.shape[-1] for features in hmm_features]
    return {"X": x, "lengths": lengths}


class HMModel(BaseModel):
    """
    Hidden Markov Model.

    This implementation is based on ``hmmlearn`` API. It is supposed
    that estimators of ``HMModel`` are model classes of ``hmmlearn``.
    """

    def __init__(self, *args, **kwargs):
        self.estimator = None
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        """
        Set up estimator as an attribute and make initial settings.

        Uses estimator from model config variable as estimator.
        If config contains key ``init_param``, sets up initial
        values ``means_``, ``covars_``, ``transmat_`` and ``startprob_``
        of the estimator as defined in ``init_params``.
        """
        _ = args, kwargs
        self.estimator = self.get("estimator", self.config)
        init_params = self.get("init_params", self.config)
        if init_params is not None:
            if "m" not in self.estimator.init_params:
                self.estimator.means_ = init_params["means_"]
            if "c" not in self.estimator.init_params:
                self.estimator.covars_ = init_params["covars_"]
            if "t" not in self.estimator.init_params:
                self.estimator.transmat_ = init_params["transmat_"]
            if "s" not in self.estimator.init_params:
                self.estimator.startprob_ = init_params["startprob_"]

    def save(self, path, *args, **kwargs):  # pylint: disable=arguments-differ
        """Save ``HMModel`` with ``dill``.

        Parameters
        ----------
        path : str
            Path to the file to save model to.
        """
        if self.estimator is not None:
            with open(path, "wb") as file:
                dill.dump(self.estimator, file)
        else:
            raise ValueError("HMM estimator does not exist. Check your cofig for 'estimator'.")

    def load(self, path, *args, **kwargs):  # pylint: disable=arguments-differ
        """Load ``HMModel`` from file with ``dill``.

        Parameters
        ----------
        path : str
            Path to the model.
        """
        with open(path, "rb") as file:
            self.estimator = dill.load(file)

    def train(self, X, lengths=None, *args, **kwargs):
        """ Train the model using data provided.

        Parameters
        ----------
        X : array-like
            A matrix of observations.
            Should be of shape (n_samples, n_features).
        lengths : array-like of integers optional
            If present, should be of shape (n_sequences, ).
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Notes
        -----
        For more details and other parameters look at the documentation for the estimator used.
        """
        self.estimator.fit(X, lengths)
        return list(self.estimator.monitor_.history)

    def predict(self, X, lengths=None, *args, **kwargs):
        """ Make prediction with the data provided.

        Parameters
        ----------
        X : array-like
            A matrix of observations.
            Should be of shape (n_samples, n_features).
        lengths : array-like of integers optional
            If present, should be of shape (n_sequences, ).
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        output: array
            Labels for each sample of X.

        Notes
        -----
        For more details and other parameters look at the documentation for the estimator used.
        """
        preds = self.estimator.predict(X, lengths)
        if lengths:
            output = np.array(np.split(preds, np.cumsum(lengths)[:-1]) + [None])[:-1]
        else:
            output = preds
        return output
