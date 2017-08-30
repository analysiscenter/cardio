""" HMM annotation """

from hmmlearn.hmm import GaussianHMM
from sklearn.externals import joblib

class HMMAnnotation(GaussianHMM):
    """ Model to generate ECG signal annotations from wavelet features.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features;
        * "diag" --- each state uses a diagonal covariance matrix;
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix;
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars_ : array
        Covariance parameters for each state.

        The shape depends on ``covariance_type``::

            (n_components, )                        if 'spherical',
            (n_features, n_features)                if 'tied',
            (n_components, n_features)              if 'diag',
            (n_components, n_features, n_features)  if 'full'
    """


    def __init__(self, n_components=1, covariance_type='diag', min_covar=0.001, startprob_prior=1.0, # pylint: disable=too-many-arguments
                 transmat_prior=1.0, means_prior=0, means_weight=0, covars_prior=0.01, covars_weight=1,
                 algorithm='viterbi', random_state=None, n_iter=10, tol=0.01, verbose=False, params='stmc',
                 init_params='stmc'):
        super().__init__(n_components, covariance_type, min_covar,
                         startprob_prior, transmat_prior, means_prior,
                         means_weight, covars_prior, covars_weight, algorithm,
                         random_state, n_iter, tol, verbose, params,
                         init_params)

    def save(self, path):
        """Save HMMAnnotation.

        Parameters
        ----------
        path : str
            Path to the location where to save the model.

        Returns
        -------
        model : HMMAnnotation
            HMMAnnotation instance unchanged.
        """
        joblib.dump(self, path)

        return self

    def load(self, path):
        """Load HMMAnnotation model.

        Parameters
        ----------
        path : str
            Path to the model.

        Returns
        -------
        model : HMMAnnotation
            Loaded HMMAnnotation instance.
        """
        model = joblib.load(path)
        return model

    def predict_on_batch(self, batch):
        """Get model predictions for a single batch.

        Parameters
        ----------
        batch : ModelEcgBatch
            Batch of signals to predict.
        predictions_var_name : str
            Pipeline variable for predictions storing.

        Returns
        -------
        batch : ModelEcgBatch
            Input batch unchanged.
        """
        for ind in batch.indices:
            batch[ind].annotation["hmm_annotation"] = \
            self.predict(self[ind].annotation["hmm_features"]).reshape((1, -1)).flatten()
