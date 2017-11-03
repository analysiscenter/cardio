""" HMModel """

import numpy as np
import dill

from ecg.dataset.dataset.models.base import BaseModel


class HMModel(BaseModel):
    """
    Class for Hidden Markov Models based on hmmlearn API.
    """

    def __init__(self, *args, **kwargs):
        self.estimator = None
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        """
        Biuld model.

        Uses estimator from model config variable as estimator.
        If config contains key "init_params", sets up initial
        values "mean_", "covars_", "transmat_" and "startprob_"
        of the estimator as defined in "init_params".
        """
        _ = args, kwargs
        self.estimator = self.get_from_config("estimator")
        init_params = self.get_from_config("init_params", None)
        if init_params is not None:
            self.estimator.means_ = init_params["means_"]
            self.estimator.covars_ = init_params["covars_"]
            self.estimator.transmat_ = init_params["transmat_"]
            self.estimator.startprob_ = init_params["startprob_"]

    def save(self, path, *args, **kwargs): # pylint: disable=arguments-differ
        """Save HMModel with dill.

        Parameters
        ----------
        path : str
            Path to the location where to save the model.
        """
        if self.estimator is not None:
            with open(path, "wb") as file:
                dill.dump(self.estimator, file)
        else:
            raise ValueError("HMM estimator does not exist. Check your cofig for 'estimator'.")

    def load(self, path, *args, **kwargs): # pylint: disable=arguments-differ
        """Load HMModel.

        Parameters
        ----------
        path : str
            Path to the model.
        """
        with open(path, "rb") as file:
            self.estimator = dill.load(file)

    def train(self, X, *args, **kwargs):
        """ Train the model with the data provided

        Parameters
        ----------
        X : array
            An array of observations to learn from. Also, it is expected
            that kwargs contain key "lenghts", which define lengths of 
            sequences.

        For more details and other parameters look at the documentation for the estimator used.
        """
        lengths = kwargs.get("lengths", None)
        self.estimator.fit(X, lengths)
        return list(self.estimator.monitor_.history)

    def predict(self, X, *args, **kwargs):
        """ Predict with the data provided

        Parameters
        ----------
        X : array
            An array of observations to make predictions from.

        For more details and other parameters look at the documentation for the estimator used.

        Returns
        -------
        output: array
            Predicted value per observation.
        """
        lengths = kwargs.get("lengths", None)
        preds = self.estimator.predict(X, lenghts)
        if lengths:
            output = np.array(np.split(preds, np.cumsum(lengths)[:-1])+[None])[:-1]
        else:
            output = preds
        return output
