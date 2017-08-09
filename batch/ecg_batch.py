"""Contains ECG Batch class."""

import os
import copy
import traceback
import itertools
import warnings

import numpy as np
import pandas as pd
import scipy

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, log_loss
from sklearn.externals import joblib

from keras.layers import Input, Conv1D, Lambda, \
                         MaxPooling1D, MaxPooling2D, \
                         Dense, GlobalMaxPooling2D
from keras.layers.core import Dropout
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam
import keras.backend as K

from hmmlearn import hmm

import dataset as ds
from . import kernels
from . import ecg_batch_tools as bt
from .keras_extra_layers import RFFT, Crop, Inception2D


class EcgBatch(ds.Batch):  # pylint: disable=too-many-public-methods
    """
    Batch of ECG data
    """

    def __init__(self, index, preloaded=None, unique_labels=None):
        super().__init__(index, preloaded)
        self._data = (None, None, None, None)
        self.signal = np.array([])
        self.annotation = np.array([])
        self.meta = np.array([])
        self.target = np.array([])
        self._unique_labels = None
        self._label_binarizer = None
        self.unique_labels = unique_labels

    def _reraise_exceptions(self, results):
        if ds.any_action_failed(results):
            all_errors = self.get_errors(results)
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")

    @staticmethod
    def _check_2d(signal):
        if signal.ndim != 2:
            raise ValueError("Each signal in batch must be 2-D ndarray")

    @property
    def components(self):
        return "signal", "annotation", "meta", "target"

    @property
    def unique_labels(self):
        return self._unique_labels

    @unique_labels.setter
    def unique_labels(self, val):
        self._unique_labels = val
        if self.unique_labels is None:
            self._label_binarizer = None
        else:
            self._label_binarizer = LabelBinarizer().fit(self.unique_labels)

    @property
    def label_binarizer(self):
        return self._label_binarizer

    def _update(self, signal=None, annotation=None, meta=None, target=None):
        """
        Update content of ecg_batch
        """
        if signal is not None:
            self.signal = np.asarray(signal)
        if annotation is not None:
            self.annotation = np.asarray(annotation)
        if meta is not None:
            self.meta = np.asarray(meta)
        if target is not None:
            self.target = np.asarray(target)
        return self

    @ds.action
    @ds.inbatch_parallel(init="indices", post="_post_load_signals", target="threads")
    def load_signals(self, index, src=None, fmt=None):
        """
        Loads ecg data

        Arguments
        index: list or array of ecg indices.
        src: dict of type index: path to ecg.
        fmt: format of ecg files. Supported formats: 'wfdb', 'npz'.
        """
        if src is not None:
            path = src[index]
        elif isinstance(self.index, ds.FilesIndex):
            path = self.index.get_fullpath(index)  # pylint: disable=no-member
        else:
            raise ValueError("Source path is not specified")
        fmt = fmt or os.path.splitext(path)[-1][1:]
        if fmt == "hea":
            return bt.load_wfdb(path)
        else:
            raise TypeError("Unsupported type of source {}".format(fmt))

    def _post_load_signals(self, results, *args, **kwargs):
        _ = args, kwargs
        self._reraise_exceptions(results)
        signal, annotation, meta, target = zip(*results)
        signal = np.array(signal + (None,))[:-1]
        return self._update(signal, annotation, meta, target)

    @ds.action
    def load_labels(self, src):
        """
        Load labels from file with signal labels. File should have a csv format
        and contain 2 columns: index of ecg and label.

        Arguments
        path: path to the file with labels
        """
        if not isinstance(src, (str, pd.Series)):
            raise TypeError("Unsupported type of source")
        if self.pipeline is None:
            raise RuntimeError("Batch must be created in pipeline")
        ds_indices = self.pipeline.dataset.indices
        if isinstance(src, str):
            src = pd.read_csv(src, header=None, names=["index", "label"], index_col=0)["label"]
        self.unique_labels = np.sort(src[ds_indices].unique())
        self._update(target=src[self.indices].values)
        return self

    def _filter_batch(self, keep_mask):
        indices = self.indices[keep_mask]
        if len(indices) == 0:
            raise ds.SkipBatchException("All batch data was dropped")
        res_batch = EcgBatch(ds.DatasetIndex(indices), unique_labels=self.unique_labels)
        res_batch._update(self.signal[keep_mask], self.annotation[keep_mask],
                          self.meta[keep_mask], self.target[keep_mask])  # pylint: disable=protected-access
        return res_batch

    @ds.action
    def drop_labels(self, drop_list):
        '''
        Drop signals labeled as label from the batch.

        Arguments
        label: label to be dropped from batch
        '''
        drop_arr = np.asarray(drop_list)
        self.unique_labels = np.setdiff1d(self.unique_labels, drop_arr)
        keep_mask = ~np.in1d(self.target, drop_arr)
        return self._filter_batch(keep_mask)

    @ds.action
    def keep_labels(self, keep_list):
        '''
        Drop signals labeled as label from the batch.

        Arguments
        label: label to be dropped from batch
        '''
        keep_arr = np.asarray(keep_list)
        self.unique_labels = np.intersect1d(self.unique_labels, keep_arr)
        keep_mask = np.in1d(self.target, keep_arr)
        return self._filter_batch(keep_mask)

    @ds.action
    def replace_labels(self, replace_dict):
        self.unique_labels = np.array(sorted({replace_dict.get(t, t) for t in self.unique_labels}))
        return self._update(target=[replace_dict.get(t, t) for t in self.target])

    @ds.action
    def binarize_labels(self):
        return self._update(target=self.label_binarizer.transform(self.target))

    @ds.action
    def drop_short_signals(self, min_length, axis=-1):
        keep_mask = np.array([sig.shape[axis] >= min_length for sig in self.signal])
        return self._filter_batch(keep_mask)

    @staticmethod
    def _pad_signal(signal, length, pad_value):
        pad_len = length - signal.shape[1]
        sig = np.pad(signal, ((0, 0), (pad_len, 0)), "constant", constant_values=pad_value)
        return sig

    def _get_segmentation_parameter(self, var, i, var_name):
        if isinstance(var, int):
            return var
        elif isinstance(var, dict):
            var = var.get(self.target[i])
            if var is None:
                raise KeyError("Undefined {} for target {}".format(var_name, self.target[i]))
            else:
                return var
        else:
            raise ValueError("Unsupported {} type".format(var_name))

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def segment_signals(self, index, length, step, pad_value=0):
        i = self.get_pos(None, "signal", index)
        self._check_2d(self.signal[i])
        step = self._get_segmentation_parameter(step, i, "step size")
        if self.signal[i].shape[1] < length:
            tmp_sig = self._pad_signal(self.signal[i], length, pad_value)
            self.signal[i] = tmp_sig[np.newaxis, ...]
        else:
            self.signal[i] = bt.segment_signals(self.signal[i], length, step)
        self.meta[i]["siglen"] = length

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def random_segment_signals(self, index, length, n_segments, pad_value=0):
        i = self.get_pos(None, "signal", index)
        self._check_2d(self.signal[i])
        n_segments = self._get_segmentation_parameter(n_segments, i, "number of segments")
        if self.signal[i].shape[1] < length:
            tmp_sig = self._pad_signal(self.signal[i], length, pad_value)
            self.signal[i] = np.tile(tmp_sig, (n_segments, 1, 1))
        else:
            self.signal[i] = bt.random_segment_signals(self.signal[i], length, n_segments)
        self.meta[i]["siglen"] = length

    def _safe_fs_resample(self, index, fs):
        i = self.get_pos(None, "signal", index)
        self._check_2d(self.signal[i])
        new_len = max(1, int(fs * self.signal[i].shape[1] / self.meta[i]["fs"]))
        self.meta[i]["fs"] = fs
        self.meta[i]["siglen"] = new_len
        self.signal[i] = bt.resample_signals(self.signal[i], new_len)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def resample_signals(self, index, fs):
        if fs <= 0:
            raise ValueError("Sampling rate must be a positive float")
        self._safe_fs_resample(index, fs)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def random_resample_signals(self, index, distr, **kwargs):
        if hasattr(np.random, distr):
            distr_fn = getattr(np.random, distr)
            fs = distr_fn(**kwargs)
        elif callable(distr):
            fs = distr_fn(**kwargs)
        else:
            raise ValueError("Unknown type of distribution parameter")
        if fs <= 0:
            fs = self[index].meta["fs"]
        self._safe_fs_resample(index, fs)

    @ds.action
    def convolve_signals(self, kernel, padding_mode="edge", axis=-1, **kwargs):
        """Convolve signals with given kernel.

        Parameters
        ----------
        kernel : array_like
            Convolution kernel.
        padding_mode : str or function
            np.pad padding mode.
        axis : int
            Axis along which signals are sliced.
        **kwargs :
            Any additional named argments to np.pad.

        Returns
        -------
        batch : EcgBatch
            Convolved batch. Changes self.signal inplace.
        """
        for i in range(len(self.signal)):
            self.signal[i] = bt.convolve_signals(self.signal[i], kernel, padding_mode, axis, **kwargs)
        return self

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def band_pass_signals(self, index, low=None, high=None, axis=-1):
        """Reject frequencies outside given range.

        Parameters
        ----------
        low : positive float
            High-pass filter cutoff frequency (Hz).
        high : positive float
            Low-pass filter cutoff frequency (Hz).
        axis : int
            Axis along which signals are sliced.

        Returns
        -------
        batch : EcgBatch
            Filtered batch. Changes self.signal inplace.
        """
        i = self.get_pos(None, "signal", index)
        self.signal[i] = bt.band_pass_signals(self.signal[i], self.meta[i]["fs"], low, high, axis)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def flip_signals(self, index):
        """Flip signals whose R-peaks are directed downwards.

        Each element of self.signal must be a 2-D ndarray. Signals are flipped along axis 1.

        Returns
        -------
        batch : EcgBatch
            Batch with flipped signals.
        """
        i = self.get_pos(None, "signal", index)
        self._check_2d(self.signal[i])
        sig = bt.band_pass_signals(self.signal[i], self.meta[i]["fs"], low=5, high=50)
        sig = bt.convolve_signals(sig, kernels.gaussian(11, 3))
        self.signal[i] *= np.where(scipy.stats.skew(sig, axis=-1) < 0, -1, 1).reshape(-1, 1)

    # The following action methods are not guaranteed to work properly

    def init_parallel(self, *args, **kwargs):
        '''
        Return array of ecg with index
        '''
        _ = args, kwargs
        return [[*self[i], i] for i in self.indices]

    def post_parallel(self, all_results, *args, **kwargs):
        #pylint: disable=too-many-locals
        #pylint: disable=too-many-branches
        '''
        Build ecg_batch from a list of items either [signal, annot, meta] or None.
        All Nones are ignored.
        Signal can be either a single signal or a list of signals.
        If signal is a list of signals, annot and meta can be a single annot and meta
        or a list of annots and metas of the same lentgh as the list of signals. In the
        first case annot and meta are broadcasted to each signal in the list of signals.

        Arguments
        all results: list of items either [signal, annot, meta] or None
        '''
        _ = args, kwargs
        if any([isinstance(res, Exception) for res in all_results]):
            print([res for res in all_results if isinstance(res, Exception)])
            return self

        valid_results = [res for res in all_results if res is not None]
        if len(valid_results) == 0:
            print('Error: all resulta are None')
            return self

        list_of_arrs = [x[0] for x in valid_results]
        list_of_lens = np.array([len(x[0]) for x in valid_results])
        list_of_annot = np.array([x[1] for x in valid_results]).ravel()
        list_of_meta = np.array([x[2] for x in valid_results]).ravel()
        list_of_origs = np.array([x[3] for x in valid_results]).ravel()

        if max(list_of_lens) <= 1:
            ind = ds.DatasetIndex(index=list_of_origs)
        else:
            ind = ds.DatasetIndex(index=np.arange(sum(list_of_lens), dtype=int))
        out_batch = EcgBatch(ind)

        if list_of_arrs[0].ndim > 3:
            raise ValueError('Signal is expected to have ndim = 1, 2 or 3, found ndim = {0}'
                             .format(list_of_arrs[0].ndim))
        if list_of_arrs[0].ndim in [1, 3]:
            #list_of_arrs[0] has shape (nb_signals, nb_channels, siglen)
            #ndim = 3 for signals with similar siglens and 1 for signals with differenr siglens
            list_of_arrs = list(itertools.chain([x for y in list_of_arrs
                                                 for x in y]))
        list_of_arrs.append([])
        batch_data = np.array(list_of_arrs)[:-1]

        if len(ind.indices) == len(list_of_origs):
            origins = list_of_origs
        else:
            origins = np.repeat(list_of_origs, list_of_lens)

        if len(ind.indices) == len(list_of_meta):
            metas = list_of_meta
        else:
            metas = []
            for i, rep in enumerate(list_of_lens):
                for _ in range(rep):
                    metas.append(copy.deepcopy(list_of_meta[i]))
            metas = np.array(metas)
        for i in range(len(batch_data)):
            metas[i].update({'origin': origins[i]})
        batch_meta = dict(zip(ind.indices, metas))

        if len(ind.indices) == len(list_of_annot):
            annots = list_of_annot
        else:
            annots = []
            for i, rep in enumerate(list_of_lens):
                for _ in range(rep):
                    annots.append(copy.deepcopy(list_of_annot[i]))
            annots = np.array(annots)
        if len(annots) > 0:
            keys = list(annots[0].keys())
        else:
            keys = []
        batch_annot = {}
        for k in keys:
            list_of_arrs = [x[k] for x in annots]
            list_of_arrs.append(np.array([]))
            batch_annot[k] = np.array(list_of_arrs)[:-1]

        return out_batch._update(signal=batch_data,
                                 annotation=batch_annot,
                                 meta=batch_meta)  # pylint: disable=protected-access

    @ds.model()
    def hmm_learn():
        """
        Hidden Markov Model to find n_components in signal
        """
        n_components = 3
        n_iter = 10
        warnings.filterwarnings("ignore")
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full",
                                n_iter=n_iter)
        return model

    @ds.action
    def set_new_model(self, model_name, new_model):
        '''
        Replace base model by new model.

        Arguments
        model_name: name of the model where to set new model.
        new_model: new model to replace previous.
        '''
        model = self.get_model_by_name(model_name)#pylint: disable=unused-variable
        model = new_model
        return self

    @ds.model()
    def fft_inception():#pylint: disable=too-many-locals
        '''
        FFT inception model. Includes initial convolution layers, then FFT transform, then
        a series of inception blocks.
        '''
        x = Input((None, 1))

        conv_1 = Conv1D(4, 4, activation='relu')(x)
        mp_1 = MaxPooling1D()(conv_1)

        conv_2 = Conv1D(8, 4, activation='relu')(mp_1)
        mp_2 = MaxPooling1D()(conv_2)
        conv_3 = Conv1D(16, 4, activation='relu')(mp_2)
        mp_3 = MaxPooling1D()(conv_3)
        conv_4 = Conv1D(32, 4, activation='relu')(mp_3)

        fft_1 = RFFT()(conv_4)
        crop_1 = Crop(begin=0, size=128)(fft_1)
        to2d = Lambda(K.expand_dims)(crop_1)

        incept_1 = Inception2D(4, 4, 3, 5, activation='relu')(to2d)
        mp2d_1 = MaxPooling2D(pool_size=(4, 2))(incept_1)

        incept_2 = Inception2D(4, 8, 3, 5, activation='relu')(mp2d_1)
        mp2d_2 = MaxPooling2D(pool_size=(4, 2))(incept_2)

        incept_3 = Inception2D(4, 12, 3, 3, activation='relu')(mp2d_2)

        pool = GlobalMaxPooling2D()(incept_3)

        fc_1 = Dense(8, kernel_initializer='uniform', activation='relu')(pool)
        drop = Dropout(0.2)(fc_1)

        fc_2 = Dense(2, kernel_initializer='uniform',
                     activation='softmax')(drop)

        opt = Adam()
        model = Model(inputs=x, outputs=fc_2)
        model.compile(optimizer=opt, loss="categorical_crossentropy")

        hist = {'train_loss': [], 'train_metric': [],
                'val_loss': [], 'val_metric': []}
        diag_classes = ['A', 'NonA']

        return model, hist, diag_classes

    @ds.action
    def train_on_batch(self, model_name):
        '''
        Train model
        '''
        model_comp = self.get_model_by_name(model_name)
        model, hist, _ = model_comp
        train_x = np.array([x for x in self.signal]).reshape((-1, 3000, 1))
        train_y = self.get_categorical_labels(model_name)
        res = model.train_on_batch(train_x, train_y)
        pred = model.predict(train_x)
        y_pred = bt.get_pos_of_max(pred)
        hist['train_loss'].append(res)
        hist['train_metric'].append(f1_score(train_y, y_pred, average='macro'))
        return self

    @ds.action
    def validate_on_batch(self, model_name):
        '''
        Validate model
        '''
        model_comp = self.get_model_by_name(model_name)
        model, hist, _ = model_comp
        test_x = np.array([x for x in self.signal]).reshape((-1, 3000, 1))
        test_y = self.get_categorical_labels(model_name)
        pred = model.predict(test_x)
        y_pred = bt.get_pos_of_max(pred)
        hist['val_loss'].append(log_loss(test_y, pred))
        hist['val_metric'].append(f1_score(test_y, y_pred, average='macro'))
        return self

    @ds.action
    def model_summary(self, model_name):
        '''
        Print model layers
        '''
        model_comp = self.get_model_by_name(model_name)
        print(model_name)
        print(model_comp[0].summary())
        return self

    @ds.action
    def save_model(self, model_name, fname):
        '''
        Save model layers and weights
        '''
        model_comp = self.get_model_by_name(model_name)
        model = model_comp[0]
        model.save_weights(fname)
        yaml_string = model.to_yaml()
        fout = open(fname + ".layers", "w")
        fout.write(yaml_string)
        fout.close()
        return self

    @ds.action
    def load_model(self, model_name, fname):
        '''
        Load model layers and weights
        '''
        model_comp = self.get_model_by_name(model_name)
        model = model_comp[0]
        fin = open(fname + ".layers", "r")
        yaml_string = fin.read()
        fin.close()
        model = model_from_yaml(yaml_string)
        model.load_weights(fname)
        return self

    @ds.action
    def train_hmm(self, model_name):
        '''
        Train hmm model on the whole batch
        '''
        warnings.filterwarnings("ignore")
        model = self.get_model_by_name(model_name)
        train_x = np.concatenate(self.signal, axis=1).T
        lengths = [x.shape[1] for x in self.signal]
        model.fit(train_x, lengths)
        return self

    @ds.action
    def predict_hmm(self, model_name):
        '''
        Get hmm predictited classes
        '''
        model = self.get_model_by_name(model_name)
        return self.predict_all_hmm(model)

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel", target='mpc')
    def predict_all_hmm(self, model):
        '''
        Get hmm predictited classes
        '''
        _ = model
        return bt.predict_hmm_classes

    @ds.action
    def save_hmm_model(self, model_name, fname):
        '''
        Save hmm model
        '''
        model = self.get_model_by_name(model_name)
        joblib.dump(model, fname + '.pkl')
        return self

    @ds.action
    def load_hmm_model(self, model_name, fname):
        '''
        Load hmm model
        '''
        model = self.get_model_by_name(model_name)#pylint: disable=unused-variable
        model = joblib.load(fname + '.pkl')
        return self

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel", target='mpc')
    def gradient(self, order):
        """
        Compute derivative of given order and add it to annotation
        """
        _ = order
        return bt.get_gradient

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel", target='mpc')
    def convolve_layer(self, layer, kernel):
        """
        Convolve layer with kernel
        """
        _ = layer, kernel
        return bt.convolve_layer

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def merge_layers(self, list_of_layers):
        """
        Merge layers from list of layers to signal
        """
        _ = list_of_layers
        return bt.merge_list_of_layers
