"""Contains ECG Batch class."""

import copy

import numpy as np
import pandas as pd
import scipy

from ..import dataset as ds
from .import kernels
from .import ecg_batch_tools as bt
from .utils import LabelBinarizer


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
        new_batch : cls
            Batch of no more than batch_size first items from concatenation of input batches.
            Contains deepcopy of input batches data.
        rest_batch : cls
            Batch of the remaining items. Contains deepcopy of input batches data.
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
        res_batch = self.__class__(ds.DatasetIndex(indices), unique_labels=self.unique_labels)
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
        **kwargs : misc
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

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def repeat_signal(self, index, reps):
        '''
        Construct an array by repeating signal the number of times given by reps. New
        signal shape will be (reps, initial_signal_shape).

        Parameters
        ----------
        repeat : positive int
            Number of times to repeat signal.

        Returns
        -------
        batch : EcgBatch
            Batch with each signal repeated. Changes self.signal inplace.
        '''
        i = self.get_pos(None, "signal", index)
        self.signal[i] = np.broadcast_to(self.signal[i], (reps, *self.signal[i].shape))

    @ds.action
    def ravel(self):
        '''
        Join a sequence of arrays along axis 0.
        '''
        x = np.concatenate(self.signal)
        x = list(x)
        x.append([])
        x = np.array(x)[:-1]
        y = np.concatenate([np.tile(item.target, (item.signal.shape[0], 1)) for item in self])
        new_index = ds.DatasetIndex(np.arange(len(x), dtype=int))
        out_batch = self.__class__(new_index)
        return out_batch.update(signal=x, target=y)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def signal_transpose(self, index, axes):
        '''
        Transpose axes

        Arguments
        axes: list of new axis orders
        '''
        i = self.get_pos(None, "signal", index)
        self.signal[i] = np.transpose(self.signal[i], axes)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def tile(self, index, reps):
        '''
        Repeat signal reps times

        Arguments
        reps: the number of repetitions
        '''
        i = self.get_pos(None, "signal", index)
        self.signal[i] = np.tile(self.signal[i], reps).reshape((*self.signal[i].shape, reps))

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def squeeze(self, index, axis=None):
        '''
        Remove single-dimensional entries from the shape of a signal

        Arguments
        axis: selects a subset of the single-dimensional entries in the shape.
            If an axis is selected with shape entry greater than one, an error is raised.
        '''
        i = self.get_pos(None, "signal", index)
        self.signal[i] = np.squeeze(self.signal[i], axis)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def slice_signal(self, index, slice_index):
        '''
        Repeat signal reps times

        Arguments
        reps: the number of repetitions
        '''
        i = self.get_pos(None, "signal", index)
        self.signal[i] = self.signal[i][slice_index]

    @ds.action
    def get_triplets(self, size, siglen):#pylint: disable=too-many-locals
        '''
        Samples triplets so that
        1) anchor and positive segments are drawn at random from the same ecg
        2) negative segments is drawn from diffetent ecg
        3) exactly one ecg is drawn from arrythmia class.

        Arguments
        size: number of triplets to sample from batch.
        signel: length of signal to sample from ecg.

        Returns
        Batch of triplets [anchor, positive_sement, negative_segmant]
        '''
        ind = ds.DatasetIndex(index=np.arange(size, dtype=int))
        out_batch = self.__class__(ind)

        batch_data = []
        batch_meta = []

        ar_indices = np.array([ind for ind in self.indices if self[ind].target == 'A'])
        sin_indices = np.array([ind for ind in self.indices if self[ind].target != 'A'])

        if len(ar_indices) == 0:
            raise ValueError('There are no arrythmia signals in batch')
        if len(sin_indices) == 0:
            raise ValueError('There are no non-arrythmia signals in batch')

        ar_choice = ar_indices[np.random.randint(low=0, high=len(ar_indices), size=size)]
        sin_choice = sin_indices[np.random.randint(low=0, high=len(sin_indices), size=size)]
        pair_choice = [ar_choice, sin_choice]

        first_choice = np.random.randint(low=0, high=2, size=size)
        for i in ind.indices:
            label = first_choice[i]

            pos_index = pair_choice[label][i]
            neg_index = pair_choice[1 - label][i]

            pos_signal = self[pos_index].signal
            if pos_signal.shape[1] < siglen:
                raise ValueError('Signal is shorter than length of target segment')
            seg_1, seg_2 = np.random.randint(low=0, high=pos_signal.shape[1] - siglen, size=2)

            neg_signal = self[neg_index].signal
            if neg_signal.shape[1] < siglen:
                raise ValueError('Signal is shorter than length of target segment')
            seg_3 = np.random.randint(low=0, high=neg_signal.shape[1] - siglen)

            batch_data.append([pos_signal[:, seg_1: seg_1 + siglen],
                               pos_signal[:, seg_2: seg_2 + siglen],
                               neg_signal[:, seg_3: seg_3 + siglen]])
            batch_meta.append([label, pos_index, seg_1, seg_2, neg_index, seg_3])

        batch_data.append(None)
        batch_data = np.array(batch_data)[:-1]
        batch_meta = np.array(batch_meta)
        return out_batch.update(signal=batch_data,
                                meta=batch_meta,
                                target=np.zeros(len(batch_data), dtype=int))
