"""Contains pipelines."""

from functools import partial
import numpy as np

import tensorflow as tf
from hmmlearn import hmm

import cardio.dataset as ds
from cardio.dataset import F, V
from cardio.models.dirichlet_model import DirichletModel, concatenate_ecg_batch
from cardio.models.hmm import HMModel


def dirichlet_train_pipeline(labels_path, batch_size=256, n_epochs=1000, gpu_options=None):
    """Train pipeline for Dirichlet model.

    This pipeline trains Dirichlet model to find propability of artrial fibrillation.
    It works with dataset that generates bathes of class EcgBatch.

    Parameters
    ----------
    labels_path : str
        Path to csv file with true labels.
    batch_size : int
        Number of samples per gradient update.
        Default value is 256.
    n_epochs : int
        Number of times to iterate over the training data arrays.
        Default value is 1000.
    gpu_options : GPUOptions
        Magic attribute generated for tf.ConfigProto "gpu_options" proto field.
        Default value is None.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    model_config = {
        "session": {"config": tf.ConfigProto(gpu_options=gpu_options)},
        "input_shape": F(lambda batch: batch.signal[0].shape[1:]),
        "class_names": F(lambda batch: batch.label_binarizer.classes_)
    }

    return (ds.Pipeline()
            .init_model("dynamic", DirichletModel, name="dirichlet", config=model_config)
            .init_variable("loss_history", init=list)
            .load(components=["signal", "meta"], fmt="wfdb")
            .load(components="target", fmt="csv", src=labels_path)
            .drop_labels(["~"])
            .replace_labels({"N": "NO", "O": "NO"})
            .flip_signals()
            .random_resample_signals("normal", loc=300, scale=10)
            .random_split_signals(2048, {"A": 9, "NO": 3})
            .binarize_labels()
            .train_model("dirichlet", make_data=concatenate_ecg_batch,
                         fetches="loss", save_to=V("loss_history"), mode="a")
            .run(batch_size=batch_size, shuffle=True, drop_last=True, n_epochs=n_epochs, lazy=True))

def dirichlet_predict_pipeline(model_path, batch_size=100, gpu_options=None):
    """Pipeline for prediction with Dirichlet model.

    This pipeline finds propability of artrial fibrillation according to Dirichlet model.
    It works with dataset that generates bathes of class EcgBatch.

    Parameters
    ----------
    model_path : str
        path to pretrained Dirichlet model
    batch_size : int
        Number of samples in batch.
        Default value is 100.
    gpu_options : GPUOptions
        Magic attribute generated for tf.ConfigProto "gpu_options" proto field.
        Default value is None.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    model_config = {
        "session": {"config": tf.ConfigProto(gpu_options=gpu_options)},
        "build": False,
        "load": {"path": model_path},
    }

    return (ds.Pipeline()
            .init_model("static", DirichletModel, name="dirichlet", config=model_config)
            .init_variable("predictions_list", init_on_each_run=list)
            .load(fmt="wfdb", components=["signal", "meta"])
            .flip_signals()
            .split_signals(2048, 2048)
            .predict_model("dirichlet", make_data=partial(concatenate_ecg_batch, return_targets=False),
                           fetches="predictions", save_to=V("predictions_list"), mode="e")
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True))

def hmm_preprocessing_pipeline(batch_size=20):
    """Pipeline for prediction with hmm model.

    This pipeline prepares data for hmm_train_pipeline.
    It works with dataset that generates bathes of class EcgBatch.

    Parameters
    ----------
    batch_size : int
        Number of samples in batch.
        Default value is 100.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    def get_wavelets(batch):
        """Get wavelets from annotation
        """
        return [ann["wavelets"] for ann in batch.annotation]

    def get_annsamples(batch):
        """Get annsamples from annotation
        """
        return [ann["annsamp"] for ann in batch.annotation]

    def get_anntypes(batch):
        """Get anntypes from annotation
        """
        return [ann["anntype"] for ann in batch.annotation]

    return (ds.Pipeline()
            .init_variable("annsamps", init_on_each_run=list)
            .init_variable("anntypes", init_on_each_run=list)
            .init_variable("wavelets", init_on_each_run=list)
            .load(fmt='wfdb', components=["signal", "annotation", "meta"], ann_ext='pu1')
            .wavelet_transform_signal(cwt_scales=[4, 8, 16], cwt_wavelet="mexh")
            .update_variable("annsamps", ds.F(get_annsamples), mode='e')
            .update_variable("anntypes", ds.F(get_anntypes), mode='e')
            .update_variable("wavelets", ds.F(get_wavelets), mode='e')
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True))

def hmm_train_pipeline(hmm_preprocessed, batch_size=20):
    """Train pipeline for Hidden Markov Model.

    This pipeline trains hmm model to isolate QRS, PQ and QT segments.
    It works with dataset that generates bathes of class EcgBatch.

    Parameters
    ----------
    hmm_preprocessed : Pipeline
        Pipeline with precomputed hmm features through hmm_preprocessing_pipeline
    batch_size : int
        Number of samples in batch.
        Default value is 20.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    def prepare_batch(batch, model):
        """Prepare data for training
        """
        _ = model
        x = np.concatenate([ann["wavelets"] for ann in batch.annotation])
        lengths = [ann["wavelets"].shape[0] for ann in batch.annotation]
        return {"X": x, "lengths": lengths}

    def prepare_means_covars(wavelets, clustering, states=[3, 5, 11, 14, 17, 19], num_states=19, num_features=3):#pylint: disable=dangerous-default-value
        """This function is specific to the task and the model configuration, thus contains hardcode.
        """
        means = np.zeros((num_states, num_features))
        covariances = np.zeros((num_states, num_features, num_features))

        # Prepearing means and variances
        last_state = 0
        unique_clusters = len(np.unique(clustering)) - 1 # Excuding value -1, which represents undefined state
        for state, cluster in zip(states, np.arange(unique_clusters)):
            value = wavelets[clustering == cluster, :]
            means[last_state:state, :] = np.mean(value, axis=0)
            covariances[last_state:state, :, :] = value.T.dot(value) / np.sum(clustering == cluster)
            last_state = state

        return means, covariances

    def prepare_transmat_startprob():
        """ This function is specific to the task and the model configuration, thus contains hardcode.
        """
        # Transition matrix - each row should add up tp 1
        transition_matrix = np.diag(19 * [14/15.0]) + np.diagflat(18 * [1/15.0], 1) + np.diagflat([1/15.0], -18)

        # We suppose that absence of P-peaks is possible
        transition_matrix[13, 14] = 0.9*1/15.0
        transition_matrix[13, 17] = 0.1*1/15.0

        # Initial distribution - should add up to 1
        start_probabilities = np.array(19 * [1/np.float(19)])

        return transition_matrix, start_probabilities

    def unravel_annotation(annsamp, anntype, length):
        """Unravel annotation
        """
        begin = -1
        end = -1
        s = 'none'
        states = {'N':0, 'st':1, 't':2, 'iso':3, 'p':4, 'pq':5}
        annot = -1 * np.ones(length)

        for j, samp in enumerate(annsamp):
            if anntype[j] == '(':
                begin = samp
                if (end > 0) & (s != 'none'):
                    if s == 'N':
                        annot[end:begin] = states['st']
                    elif s == 't':
                        annot[end:begin] = states['iso']
                    elif s == 'p':
                        annot[end:begin] = states['pq']
            elif anntype[j] == ')':
                end = samp
                if (begin > 0) & (s != 'none'):
                    annot[begin:end] = states[s]
            else:
                s = anntype[j]

        return annot

    lengths = [wavelet.shape[0] for wavelet in hmm_preprocessed.get_variable("wavelets")]
    wavelets = np.concatenate(hmm_preprocessed.get_variable("wavelets"))
    anntype = hmm_preprocessed.get_variable("anntypes")
    annsamp = hmm_preprocessed.get_variable("annsamps")

    unravelled = np.concatenate([unravel_annotation(samp, types, length) for
                                 samp, types, length in zip(annsamp, anntype, lengths)])
    means, covariances = prepare_means_covars(wavelets, unravelled, states=[3, 5, 11, 14, 17, 19], num_features=3)
    transition_matrix, start_probabilities = prepare_transmat_startprob()

    config_train = {
        'build': True,
        'estimator': hmm.GaussianHMM(n_components=19, n_iter=25, covariance_type="full", random_state=42,
                                     init_params='', verbose=False),
        'init_params': {'means_': means, 'covars_': covariances, 'transmat_': transition_matrix,
                        'startprob_': start_probabilities}
    }

    return (ds.Pipeline()
            .init_model("dynamic", HMModel, "HMM", config=config_train)
            .load(fmt='wfdb', components=["signal", "annotation", "meta"], ann_ext='pu1')
            .wavelet_transform_signal(cwt_scales=[4, 8, 16], cwt_wavelet="mexh")
            .train_model("HMM", make_data=prepare_batch)
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True))

def hmm_predict_pipeline(model_path, batch_size=20):
    """Pipeline for prediction with hmm model.

    This pipeline isolates QRS, PQ and QT segments.
    It works with dataset that generates bathes of class EcgBatch.

    Parameters
    ----------
    model_path : str
        Path to pretrained hmm model.
    batch_size : int
        Number of samples in batch.
        Default value is 20.

    Returns
    -------
    pipeline : Pipeline
        Output pipeline.
    """

    def prepare_batch(batch, model):
        """Prepare data for train
        """
        _ = model
        x = np.concatenate([ann["wavelets"] for ann in batch.annotation])
        lengths = [ann["wavelets"].shape[0] for ann in batch.annotation]
        return {"X": x, "lengths": lengths}

    def get_batch(batch):
        """Get batch list
        """
        return [batch]

    config_predict = {
        'build': False,
        'load': {'path': model_path}
    }

    return (ds.Pipeline()
            .init_model("static", HMModel, "HMM", config=config_predict)
            .init_variable("batch", init_on_each_run=list)
            .load(fmt="wfdb", components=["signal", "meta"])
            .wavelet_transform_signal(cwt_scales=[4, 8, 16], cwt_wavelet="mexh")
            .predict_model("HMM", make_data=prepare_batch, save_to=ds.B("_temp"), mode='w')
            .write_to_annotation("hmm_annotation", "_temp")
            .calc_ecg_parameters()
            .update_variable("batch", ds.F(get_batch), mode='e')
            .run(batch_size=batch_size, shuffle=False, drop_last=False, n_epochs=1, lazy=True))
