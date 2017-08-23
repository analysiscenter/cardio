"""Contains ECG Batch class.""" #pylint: disable=too-many-lines

import os
import sys

import copy
import itertools
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate
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

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

from .. import dataset as ds
from . import kernels
from . import ecg_batch_tools as bt
from .keras_extra_layers import RFFT, Crop, Inception2D



class EcgBatch(ds.Batch):  # pylint: disable=too-many-public-methods
    """Class for storing batch of ECG signals."""

    def __init__(self, index, preloaded=None, unique_labels=None):
        super().__init__(index, preloaded)
        self._data = (None, None, None, None)
        self.signal = np.array([np.array([])] * len(index) + [None])[:-1]
        self.annotation = np.array([{}] * len(index))
        self.meta = np.array([{}] * len(index))
        self.target = np.array([None] * len(index))
        self._unique_labels = None
        self._label_binarizer = None
        self.unique_labels = unique_labels

    def _reraise_exceptions(self, results):
        """Reraise all exceptions in results list.

        Parameters
        ----------
        results : list
            Post function computation results.
        """
        if ds.any_action_failed(results):
            all_errors = self.get_errors(results)
            raise RuntimeError("Cannot assemble the batch", all_errors)

    @staticmethod
    def _check_2d(signal):
        """Check if given signal is 2-D.

        Parameters
        ----------
        signal : ndarray
            Signal to check.
        """
        if signal.ndim != 2:
            raise ValueError("Each signal in batch must be 2-D ndarray")

    @property
    def components(self):
        """Return data components names."""
        return "signal", "annotation", "meta", "target"

    @property
    def unique_labels(self):
        """Return unique labels in dataset."""
        return self._unique_labels

    @unique_labels.setter
    def unique_labels(self, val):
        """Set unqiue labels value to val. Updates self.label_binarizer instance.

        Parameters
        ----------
        val : 1-D ndarray
            New unique labels.
        """
        self._unique_labels = val
        if self.unique_labels is None:
            self._label_binarizer = None
        else:
            self._label_binarizer = LabelBinarizer().fit(self.unique_labels)

    @property
    def label_binarizer(self):
        """Return LabelBinarizer instance for unique labels in dataset."""
        return self._label_binarizer

    def update(self, signal=None, annotation=None, meta=None, target=None):
        """Update batch components.

        Parameters
        ----------
        signal : ndarray
            New signal component.
        annotation : ndarray
            New annotation component.
        meta : ndarray
            New meta component.
        target : ndarray
            New target component.

        Returns
        -------
        batch : EcgBatch
            Updated batch. Changes batch components inplace.
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

    @classmethod
    def merge(cls, batches, batch_size=None):
        """Concatenate list of EcgBatch instances and split the result into two batches of sizes
        (batch_size, sum(lens of batches) - batch_size).

        Parameters
        ----------
        batches : list
            List of EcgBatch instances.
        batch_size : positive int
            Length of the first resulting batch.

        Returns
        -------
        batches : tuple
            Tuple of two EcgBatch instances. Each instance contains deepcopy of input batches data.
        """
        batches = [batch for batch in batches if batch is not None]
        if len(batches) == 0:
            return None, None
        total_len = np.sum([len(batch) for batch in batches])
        if batch_size is None:
            batch_size = total_len
        elif not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size must be positive int")
        indices = np.arange(total_len)

        data = []
        for comp in batches[0].components:
            data.append(np.concatenate([batch.get(component=comp) for batch in batches]))
        data = copy.deepcopy(data)

        new_indices = indices[:batch_size]
        new_batch = cls(ds.DatasetIndex(new_indices), unique_labels=batches[0].unique_labels)
        new_batch._data = tuple(comp[:batch_size] for comp in data)  # pylint: disable=protected-access
        if total_len <= batch_size:
            rest_batch = None
        else:
            rest_indices = indices[batch_size:]
            rest_batch = cls(ds.DatasetIndex(rest_indices), unique_labels=batches[0].unique_labels)
            rest_batch._data = tuple(comp[batch_size:] for comp in data)  # pylint: disable=protected-access
        return new_batch, rest_batch

    @ds.action
    def load(self, src=None, fmt=None, components=None, *args, **kwargs):
        """Load given batch components from source.

        Parameters
        ----------
        src : misc
            Source to load components from.
        fmt : str
            Source format.
        components : iterable
            Components to load.

        Returns
        -------
        batch : EcgBatch
            Batch with loaded components. Changes components inplace.
        """
        if components is None:
            components = self.components
        components = np.asarray(components).ravel()
        if (fmt == "csv" or fmt is None and isinstance(src, pd.Series)) and np.all(components == "target"):
            return self._load_labels(src)
        elif fmt == "wfdb":
            return self._load_wfdb(src=src, components=components)
        else:
            return super().load(src, fmt, components, *args, **kwargs)

    @ds.inbatch_parallel(init="indices", post="_assemble_load", target="threads")
    def _load_wfdb(self, index, src=None, components=None):
        """Load given components from wfdb files.

        Parameters
        ----------
        src : dict
            Path to wfdb file for every batch index. If None, path from FilesIndex is used.
        components : iterable
            Components to load.

        Returns
        -------
        batch : EcgBatch
            Batch with loaded components. Changes components inplace.
        """
        if src is not None:
            path = src[index]
        elif isinstance(self.index, ds.FilesIndex):
            path = self.index.get_fullpath(index)  # pylint: disable=no-member
        else:
            raise ValueError("Source path is not specified")
        return bt.load_wfdb(path, components)

    def _assemble_load(self, results, *args, **kwargs):
        """Concatenate results of different workers and update self.

        Parameters
        ----------
        results : list
            Workers' results.

        Returns
        -------
        batch : EcgBatch
            Assembled batch. Changes components inplace.
        """
        _ = args, kwargs
        self._reraise_exceptions(results)
        components = kwargs.get("components", None)
        if components is None:
            components = self.components
        for comp, data in zip(components, zip(*results)):
            if comp == "signal":
                data = np.array(data + (None,))[:-1]
            else:
                data = np.array(data)
            setattr(self, comp, data)
        return self

    def _load_labels(self, src):
        """Load labels from csv file or pandas Series.

        Parameters
        ----------
        src : str or Series
            Path to csv file or pandas Series. File should contain 2 columns: ecg index and label.

        Returns
        -------
        batch : EcgBatch
            Batch with loaded labels. Changes self.target inplace.
        """
        if not isinstance(src, (str, pd.Series)):
            raise TypeError("Unsupported type of source")
        if self.pipeline is None:
            raise RuntimeError("Batch must be created in pipeline")
        ds_indices = self.pipeline.dataset.indices
        if isinstance(src, str):
            src = pd.read_csv(src, header=None, names=["index", "label"], index_col=0)["label"]
        self.unique_labels = np.sort(src[ds_indices].unique())
        self.update(target=src[self.indices].values)
        return self

    def _filter_batch(self, keep_mask):
        """Drop elements from batch with corresponding False values in keep_mask.

        Parameters
        ----------
        keep_mask : bool 1-D ndarray
            Filtering mask.

        Returns
        -------
        batch : EcgBatch
            Filtered batch. Creates a new EcgBatch instance.
        """
        indices = self.indices[keep_mask]
        if len(indices) == 0:
            raise ds.SkipBatchException("All batch data was dropped")
        res_batch = EcgBatch(ds.DatasetIndex(indices), unique_labels=self.unique_labels)
        res_batch.update(self.signal[keep_mask], self.annotation[keep_mask],
                         self.meta[keep_mask], self.target[keep_mask])
        return res_batch

    @ds.action
    def drop_labels(self, drop_list):
        """Drop those elements from batch, whose labels are in drop_list.

        Parameters
        ----------
        drop_list : list
            Labels to be dropped from batch.

        Returns
        -------
        batch : EcgBatch
            Filtered batch. Creates a new EcgBatch instance.
        """
        drop_arr = np.asarray(drop_list)
        self.unique_labels = np.setdiff1d(self.unique_labels, drop_arr)
        keep_mask = ~np.in1d(self.target, drop_arr)
        return self._filter_batch(keep_mask)

    @ds.action
    def keep_labels(self, keep_list):
        """Keep only those elements in batch, whose labels are in keep_list.

        Parameters
        ----------
        keep_list : list
            Labels to be kept in batch.

        Returns
        -------
        batch : EcgBatch
            Filtered batch. Creates a new EcgBatch instance.
        """
        keep_arr = np.asarray(keep_list)
        self.unique_labels = np.intersect1d(self.unique_labels, keep_arr)
        keep_mask = np.in1d(self.target, keep_arr)
        return self._filter_batch(keep_mask)

    @ds.action
    def replace_labels(self, replace_dict):
        """Replace labels in batch with corresponding values in replace_dict.

        Parameters
        ----------
        replace_dict : dict
            Dictionary containing (old label : new label) pairs.

        Returns
        -------
        batch : EcgBatch
            Batch with replaced labels. Changes self.target inplace.
        """
        self.unique_labels = np.array(sorted({replace_dict.get(t, t) for t in self.unique_labels}))
        return self.update(target=[replace_dict.get(t, t) for t in self.target])

    @ds.action
    def binarize_labels(self):
        """Binarize labels in batch in a one-vs-all fashion.

        Returns
        -------
        batch : EcgBatch
            Batch with binarized labels. Changes self.target inplace.
        """
        return self.update(target=self.label_binarizer.transform(self.target))

    @ds.action
    def drop_short_signals(self, min_length, axis=-1):
        """Drop short signals from batch.

        Parameters
        ----------
        min_length : positive int
            Minimal signal length.
        axis : int
            Axis along which length is calculated.

        Returns
        -------
        batch : EcgBatch
            Filtered batch. Creates a new EcgBatch instance.
        """
        keep_mask = np.array([sig.shape[axis] >= min_length for sig in self.signal])
        return self._filter_batch(keep_mask)

    @staticmethod
    def _pad_signal(signal, length, pad_value):
        """Pad signal to the left along axis 1 with pad value.

        Parameters
        ----------
        signal : 2-D ndarray
            Signals to pad.
        length : positive int
            Length of padded signal along axis 1.
        pad_value : float
            Padding value.

        Returns
        -------
        signal : 2-D ndarray
            Padded signals.
        """
        pad_len = length - signal.shape[1]
        sig = np.pad(signal, ((0, 0), (pad_len, 0)), "constant", constant_values=pad_value)
        return sig

    @staticmethod
    def _get_segmentation_arg(arg, arg_name, target):
        """Get segmentation step or number of segments for given signal.

        Parameters
        ----------
        arg : positive int or dict
            Segmentation step or number of segments.
        arg_name : str
            Argument name.
        target : hashable
            Signal target.

        Returns
        -------
        arg : positive int
            Segmentation step or number of segments for given signal.
        """
        if isinstance(arg, int):
            return arg
        elif isinstance(arg, dict):
            arg = arg.get(target)
            if arg is None:
                raise KeyError("Undefined {} for target {}".format(arg_name, target))
            else:
                return arg
        else:
            raise ValueError("Unsupported {} type".format(arg_name))

    @staticmethod
    def _check_segmentation_args(signal, target, length, arg, arg_name):
        """Check values of segmentation parameters.

        Parameters
        ----------
        signal : 2-D ndarray
            Signals to segment.
        target : hashable
            Signal target.
        length : positive int
            Length of each segment along axis 1.
        arg : positive int or dict
            Segmentation step or number of segments.
        arg_name : str
            Argument name.

        Returns
        -------
        arg : positive int
            Segmentation step or number of segments for given signal.
        """
        EcgBatch._check_2d(signal)
        if (length <= 0) or not isinstance(length, int):
            raise ValueError("Segment length must be positive integer")
        arg = EcgBatch._get_segmentation_arg(arg, arg_name, target)
        if (arg <= 0) or not isinstance(arg, int):
            raise ValueError("{} must be positive integer".format(arg_name))
        return arg

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def segment_signals(self, index, length, step, pad_value=0):
        """Segment signals along axis 1 with given length and step.

        If signal length along axis 1 is less than length, it is padded to the left with pad value.

        Parameters
        ----------
        length : positive int
            Length of each segment along axis 1.
        step : positive int or dict
            Segmentation step. If step is dict, segmentation step is fetched by signal target key.
        pad_value : float
            Padding value.

        Returns
        -------
        batch : EcgBatch
            Segmented batch. Changes self.signal and self.meta inplace.
        """
        i = self.get_pos(None, "signal", index)
        step = self._check_segmentation_args(self.signal[i], self.target[i], length, step, "step size")
        if self.signal[i].shape[1] < length:
            tmp_sig = self._pad_signal(self.signal[i], length, pad_value)
            self.signal[i] = tmp_sig[np.newaxis, ...]
        else:
            self.signal[i] = bt.segment_signals(self.signal[i], length, step)
        self.meta[i]["siglen"] = length

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def random_segment_signals(self, index, length, n_segments, pad_value=0):
        """Segment signals along axis 1 n_segments times with random start position and given length.

        If signal length along axis 1 is less than length, it is padded to the left with pad value.

        Parameters
        ----------
        length : positive int
            Length of each segment along axis 1.
        n_segments : positive int or dict
            Number of segments. If n_segments is dict, number of segments is fetched by signal target key.
        pad_value : float
            Padding value.

        Returns
        -------
        batch : EcgBatch
            Segmented batch. Changes self.signal and self.meta inplace.
        """
        i = self.get_pos(None, "signal", index)
        n_segments = self._check_segmentation_args(self.signal[i], self.target[i], length,
                                                   n_segments, "number of segments")
        if self.signal[i].shape[1] < length:
            tmp_sig = self._pad_signal(self.signal[i], length, pad_value)
            self.signal[i] = np.tile(tmp_sig, (n_segments, 1, 1))
        else:
            self.signal[i] = bt.random_segment_signals(self.signal[i], length, n_segments)
        self.meta[i]["siglen"] = length

    def _safe_fs_resample(self, index, fs):
        """Resample signals along axis 1 to given sampling rate.

        New sampling rate is guaranteed to be positive float.

        Parameters
        ----------
        fs : positive float
            New sampling rate.
        """
        i = self.get_pos(None, "signal", index)
        self._check_2d(self.signal[i])
        new_len = max(1, int(fs * self.signal[i].shape[1] / self.meta[i]["fs"]))
        self.meta[i]["fs"] = fs
        self.meta[i]["siglen"] = new_len
        self.signal[i] = bt.resample_signals(self.signal[i], new_len)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def resample_signals(self, index, fs):
        """Resample signals along axis 1 to given sampling rate.

        Parameters
        ----------
        fs : positive float
            New sampling rate.

        Returns
        -------
        batch : EcgBatch
            Resampled batch. Changes self.signal and self.meta inplace.
        """
        if fs <= 0:
            raise ValueError("Sampling rate must be a positive float")
        self._safe_fs_resample(index, fs)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def random_resample_signals(self, index, distr, **kwargs):
        """Resample signals along axis 1 to new sampling rate, sampled from given distribution.

        Parameters
        ----------
        distr : str or callable
            NumPy distribution name or callable to sample from.
        kwargs : misc
            Distribution parameters. If new sampling rate is negative, the signal is left unchanged.

        Returns
        -------
        batch : EcgBatch
            Resampled batch. Changes self.signal and self.meta inplace.
        """
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

        return out_batch.update(signal=batch_data,
                                annotation=batch_annot,
                                meta=batch_meta)

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

    @ds.action
    def print_ecg(self, index, start=0, end=None, fs=None, annotate=False): #pylint: disable=too-many-locals
        """ Method for printing an ECG.

        Parameters
        ----------
        index : element of self.indices
            Index of the signal in the batch to print report about.
        start : int
            From which second of the signal to start plotting.
        end : int
            Second of the signal to end plot.
        fs : float
            Sampling rate of the signal.
        annotate : bool
            Show annotation on the plot or not.

        Returns
        -------
        None
        """
        i = self.get_pos(None, "signal", index)

        sig = self.signal[i]
        annotation = self.annotation[i]
        meta = self.meta[i]

        if fs is None:
            fs = meta["fs"]

        if end is None:
            end = sig.shape[1]
        else:
            end = np.int(end*fs)
        start = np.int(start*fs)

        sig = sig[:, start:end]

        num_channels = sig.shape[0]
        fig = plt.figure(figsize=(10, 4*num_channels))
        for channel in range(num_channels):
            ax = fig.add_subplot(num_channels, 1, channel+1)
            ax.plot((np.arange(start, end) / fs), sig[channel, :])
            ax.set_xlabel("t, sec")
            ax.set_ylabel(meta["units"][channel] if "units" in meta.keys() else "mV")
            ax.grid("on", which='major')
            if annotate:
                r_starts, r_ends = bt.find_intervals_borders(annotation['hmm_annotation'][start:end],
                                                             [0, 1, 2])
                for begin, stop in zip((r_starts + start)/fs, (r_ends + start)/fs):
                    ax.axvspan(begin, stop, color='red', alpha=0.3)

                p_starts, p_ends = bt.find_intervals_borders(annotation['hmm_annotation'][start:end],
                                                             [14, 15, 16])
                for begin, stop in zip((p_starts + start)/fs, (p_ends + start)/fs):
                    ax.axvspan(begin, stop, color='green', alpha=0.3)

                t_starts, t_ends = bt.find_intervals_borders(annotation['hmm_annotation'][start:end],
                                                             [5, 6, 7, 8, 9, 10])
                for begin, stop in zip((t_starts + start)/fs, (t_ends + start)/fs):
                    ax.axvspan(begin, stop, color='blue', alpha=0.3)

    @ds.model()
    def load_hmm_annotation():
        """ Loads HMM that is trained to annotate signals"""

        try:
            model = joblib.load('/notebooks/dpodvyaznikov/clada/ecg_report/Intenship_submit/hmm_model_2.pkl')
        except FileNotFoundError:
            model = None

        return model

    @ds.action
    @ds.inbatch_parallel(init="indices", target='threads')
    def generate_hmm_annotations(self, index, cwt_scales, cwt_wavelet, model_name):
        """Annotatate signals in batch and write it to annotation component under key
        'hmm_annotation'.

        Parameters
        ----------
        cwt_scales : array_like
            Scales to use for Continuous Wavele Transformation.
        cwt_wavelet : object or str
            Wavelet to use in CWT.

        Returns
        -------
        batch : EcgBatch
            EcgBatch with annotations of signals.
        """
        i = self.get_pos(None, "signal", index)
        model = self.get_model_by_name(model_name)
        self._check_2d(self.signal[i])

        self.annotation[i]["hmm_annotation"] = bt.predict_hmm_annot(self.signal[i],
                                                                    cwt_scales,
                                                                    cwt_wavelet,
                                                                    model)

    @ds.action
    @ds.inbatch_parallel(init="indices", target='threads')
    def calc_ecg_parameters(self, index):
        """ Calculates PQ interval based on annotation and writes it in meta under key 'pq'.
        Annotation can be obtained using hmm_annotation model with method predict_hmm_annotation.

        Parameters
        ----------
        None

        Returns
        -------
        batch : EcgBatch
            Batch with report parameters stored in meta component.
        """
        i = self.get_pos(None, "signal", index)

        self.meta[i]["hr"] = bt.calc_hr(self.signal[i],
                                        self.annotation[i]['hmm_annotation'],
                                        self.meta[i]['fs'])

        self.meta[i]["pq"] = bt.calc_pq(self.annotation[i]['hmm_annotation'],
                                        self.meta[i]['fs'])

        self.meta[i]["qt"] = bt.calc_qt(self.annotation[i]['hmm_annotation'],
                                        self.meta[i]['fs'])

        self.meta[i]["qrs"] = bt.calc_qrs(self.annotation[i]['hmm_annotation'],
                                          self.meta[i]['fs'])

    @ds.action
    def print_report(self, index):
        """ Takes information from batch about specific signal by index
        and prints table with ecg report.

        Parameters
        ----------
        index : element of self.indices
            Index of the signal in the batch to print report about.

        Returns
        -------
        None
        """
        i = self.get_pos(None, "signal", index)

        print(tabulate([['ЧСС', np.round(self.meta[i]['hr'], 2), 'уд./мин.'],
                        ['QRS', np.round(self.meta[i]['qrs'], 2), 'сек.'],
                        ['PQ', np.round(self.meta[i]['pq'], 2), 'сек.'],
                        ['QT', np.round(self.meta[i]['qt'], 2), 'сек.'],
                        ['Вероятность аритмии', self.meta[i]['pred_af'], '%']],
                       headers=['Параметр', 'Значение', 'Ед.изм.'], tablefmt='orgtbl'))

    @ds.action
    def append_api_result(self, var_name):
        if var_name is not None:
            for ind in self.indices:
                res_dict = {"heart_rate": self[ind].meta['hr'],
                            "qrs_interval": self[ind].meta['qrs'],
                            "pq_interval": self[ind].meta['pq'],
                            "qt_interval": self[ind].meta['qt'],
                            "units": self[ind].meta['units'],
                            "frequency": self[ind].meta['fs'], 
                            "signal":self[ind].signal,
                            "annotation": self[ind].annotation["hmm_annotation"]}
                self.pipeline.get_variable(var_name, init=list, init_on_each_run=True).append(res_dict)
        return self
