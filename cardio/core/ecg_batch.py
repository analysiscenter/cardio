"""Contains ECG Batch class."""
# pylint: disable=too-many-lines

import copy
from textwrap import dedent

import numpy as np
import pandas as pd
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import pywt

from .. import dataset as ds
from . import kernels
from . import ecg_batch_tools as bt
from .utils import partialmethod, LabelBinarizer


ACTIONS_DICT = {
    "fft": (np.fft.fft, "numpy.fft.fft", "a Discrete Fourier Transform"),
    "ifft": (np.fft.ifft, "numpy.fft.ifft", "an inverse Discrete Fourier Transform"),
    "rfft": (np.fft.rfft, "numpy.fft.rfft", "a real-input Discrete Fourier Transform"),
    "irfft": (np.fft.irfft, "numpy.fft.irfft", "a real-input inverse Discrete Fourier Transform"),
    "dwt": (pywt.dwt, "pywt.dwt", "a single level Discrete Wavelet Transform"),
    "idwt": (lambda x, *args, **kwargs: pywt.idwt(*x, *args, **kwargs), "pywt.idwt",
             "a single level inverse Discrete Wavelet Transform"),
    "wavedec": (pywt.wavedec, "pywt.wavedec", "a multilevel 1D Discrete Wavelet Transform"),
    "waverec": (lambda x, *args, **kwargs: pywt.waverec(list(x), *args, **kwargs), "pywt.waverec",
                "a multilevel 1D Inverse Discrete Wavelet Transform"),
    "pdwt": (lambda x, part, *args, **kwargs: pywt.downcoef(part, x, *args, **kwargs), "pywt.downcoef",
             "a partial Discrete Wavelet Transform data decomposition"),
    "cwt": (lambda x, *args, **kwargs: pywt.cwt(x, *args, **kwargs)[0], "pywt.cwt", "a Continuous Wavelet Transform"),
}


TEMPLATE_DOCSTRING = """
    Compute {description} for each slice of a signal over the axis 0
    (typically the channel axis).

    This method simply wraps ``apply_to_each_channel`` method by setting the
    ``func`` argument to ``{full_name}``.

    Parameters
    ----------
    src : str, optional
        Batch attribute or component name to get the data from.
    dst : str, optional
        Batch attribute or component name to put the result in.
    args : misc
        Any additional positional arguments to ``{full_name}``.
    kwargs : misc
        Any additional named arguments to ``{full_name}``.

    Returns
    -------
    batch : EcgBatch
        Transformed batch. Changes ``dst`` attribute or component.
"""
TEMPLATE_DOCSTRING = dedent(TEMPLATE_DOCSTRING).strip()


def add_actions(actions_dict, template_docstring):
    """Add new actions in ``EcgBatch`` by setting ``func`` argument in
    ``EcgBatch.apply_to_each_channel`` method to given callables.

    Parameters
    ----------
    actions_dict : dict
        A dictionary, containing new methods' names as keys and a callable,
        its full name and description for each method as values.
    template_docstring : str
        A string, that will be formatted for each new method from
        ``actions_dict`` using ``full_name`` and ``description`` parameters
        and assigned to its ``__doc__`` attribute.

    Returns
    -------
    decorator : callable
        Class decorator.
    """
    def decorator(cls):
        """Returned decorator."""
        for method_name, (func, full_name, description) in actions_dict.items():
            docstring = template_docstring.format(full_name=full_name, description=description)
            method = partialmethod(cls.apply_to_each_channel, func)
            method.__doc__ = docstring
            setattr(cls, method_name, method)
        return cls
    return decorator


@add_actions(ACTIONS_DICT, TEMPLATE_DOCSTRING)  # pylint: disable=too-many-public-methods,too-many-instance-attributes
class EcgBatch(ds.Batch):
    """Batch class for ECG signals storing.

    Contains ECG signals and additional metadata along with various processing
    methods.

    Parameters
    ----------
    index : DatasetIndex
        Unique identifiers of ECGs in the batch.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to ``None``.
    unique_labels : 1-D ndarray, optional
        Array with unique labels in a dataset.

    Attributes
    ----------
    index : DatasetIndex
        Unique identifiers of ECGs in the batch.
    signal : 1-D ndarray
        Array of 2-D ndarrays with ECG signals in channels first format.
    annotation : 1-D ndarray
        Array of dicts with different types of annotations.
    meta : 1-D ndarray
        Array of dicts with metadata about signals.
    target : 1-D ndarray
        Array with signals' labels.
    unique_labels : 1-D ndarray
        Array with unique labels in a dataset.
    label_binarizer : LabelBinarizer
        Object for label one-hot encoding.

    Note
    ----
    Some batch methods take ``index`` as their first argument after ``self``.
    You should not specify it in your code, it will be passed automatically by
    ``inbatch_parallel`` decorator. For example, ``resample_signals`` method
    with ``index`` and ``fs`` arguments should be called as
    ``batch.resample_signals(fs)``.
    """

    def __init__(self, index, preloaded=None, unique_labels=None):
        super().__init__(index, preloaded)
        self.signal = self.array_of_nones
        self.annotation = self.array_of_dicts
        self.meta = self.array_of_dicts
        self.target = self.array_of_nones
        self._unique_labels = None
        self._label_binarizer = None
        self.unique_labels = unique_labels

    @property
    def components(self):
        """tuple of str: Data components names."""
        return "signal", "annotation", "meta", "target"

    @property
    def array_of_nones(self):
        """1-D ndarray: ``NumPy`` array with ``None`` values."""
        return np.array([None] * len(self.index))

    @property
    def array_of_dicts(self):
        """1-D ndarray: ``NumPy`` array with empty ``dict`` values."""
        return np.array([{} for _ in range(len(self.index))])

    @property
    def unique_labels(self):
        """1-D ndarray: Unique labels in a dataset."""
        return self._unique_labels

    @unique_labels.setter
    def unique_labels(self, val):
        """Set unique labels value to ``val``. Updates
        ``self.label_binarizer`` instance.

        Parameters
        ----------
        val : 1-D ndarray
            New unique labels.
        """
        self._unique_labels = val
        if self.unique_labels is None or len(self.unique_labels) == 0:
            self._label_binarizer = None
        else:
            self._label_binarizer = LabelBinarizer().fit(self.unique_labels)

    @property
    def label_binarizer(self):
        """LabelBinarizer: Label binarizer object for unique labels in a
        dataset."""
        return self._label_binarizer

    def _reraise_exceptions(self, results):
        """Reraise all exceptions in the ``results`` list.

        Parameters
        ----------
        results : list
            Post function computation results.

        Raises
        ------
        RuntimeError
            If any paralleled action raised an ``Exception``.
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

        Raises
        ------
        ValueError
            If given signal is not two-dimensional.
        """
        if signal.ndim != 2:
            raise ValueError("Each signal in batch must be 2-D ndarray")

    # Input/output methods

    @ds.action
    def load(self, src=None, fmt=None, components=None, ann_ext=None, *args, **kwargs):
        """Load given batch components from source.

        Most of the ``EcgBatch`` actions work under the assumption that both
        ``signal`` and ``meta`` components are loaded. In case this assumption
        is not fulfilled, normal operation of the actions is not guaranteed.

        This method supports loading of signals from wfdb, DICOM, EDF, wav and
        blosc formats.

        Parameters
        ----------
        src : misc, optional
            Source to load components from.
        fmt : str, optional
            Source format.
        components : str or array-like, optional
            Components to load.
        ann_ext : str, optional
            Extension of the annotation file.

        Returns
        -------
        batch : EcgBatch
            Batch with loaded components. Changes batch data inplace.
        """
        if components is None:
            components = self.components
        components = np.asarray(components).ravel()
        if (fmt == "csv" or fmt is None and isinstance(src, pd.Series)) and np.all(components == "target"):
            return self._load_labels(src)
        elif fmt in ["wfdb", "dicom", "edf", "wav"]:
            return self._load_data(src=src, fmt=fmt, components=components, ann_ext=ann_ext, *args, **kwargs)
        else:
            return super().load(src, fmt, components, *args, **kwargs)

    @ds.inbatch_parallel(init="indices", post="_assemble_load", target="threads")
    def _load_data(self, index, src=None, fmt=None, components=None, *args, **kwargs):
        """Load given components from wfdb, DICOM, EDF or wav files.

        Parameters
        ----------
        src : misc, optional
            Source to load components from. If ``None``, path from
            ``FilesIndex`` is used.
        fmt : str, optional
            Source format.
        components : iterable, optional
            Components to load.
        ann_ext: str, optional
            Extension of the annotation file.

        Returns
        -------
        batch : EcgBatch
            Batch with loaded components. Changes batch data inplace.

        Raises
        ------
        ValueError
            If source path is not specified and batch's ``index`` is not a
            ``FilesIndex``.
        """
        loaders = {"wfdb": bt.load_wfdb, "dicom": bt.load_dicom,
                   "edf": bt.load_edf, "wav": bt.load_wav}

        if src is not None:
            path = src[index]
        elif isinstance(self.index, ds.FilesIndex):
            path = self.index.get_fullpath(index)  # pylint: disable=no-member
        else:
            raise ValueError("Source path is not specified")
        return loaders[fmt](path, components, *args, **kwargs)

    def _assemble_load(self, results, *args, **kwargs):
        """Concatenate results of different workers and update ``self``.

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
        """Load labels from a csv file or ``pandas.Series``.

        Parameters
        ----------
        src : str or Series
            Path to csv file or ``pandas.Series``. The file should contain two
            columns: ECG index and label. It shouldn't have a header.

        Returns
        -------
        batch : EcgBatch
            Batch with loaded labels. Changes ``self.target`` inplace.

        Raises
        ------
        TypeError
            If ``src`` is not a string or ``pandas.Series``.
        RuntimeError
            If ``unique_labels`` has not been defined and the batch was not
            created by a ``Pipeline``.
        """
        if not isinstance(src, (str, pd.Series)):
            raise TypeError("Unsupported type of source")
        if isinstance(src, str):
            src = pd.read_csv(src, header=None, names=["index", "label"], index_col=0)["label"]
        self.target = src[self.indices].values
        if self.unique_labels is None:
            if self.pipeline is None:
                raise RuntimeError("Batch with undefined unique_labels must be created in a pipeline")
            ds_indices = self.pipeline.dataset.indices
            self.unique_labels = np.sort(src[ds_indices].unique())
        return self

    def show_ecg(self, index=None, start=0, end=None, annot=None, subplot_size=(10, 4)):  # pylint: disable=too-many-locals, line-too-long
        """Plot an ECG signal.

        Optionally highlight QRS complexes along with P and T waves. Each
        channel is displayed on a separate subplot.

        Parameters
        ----------
        index : element of ``self.indices``, optional
            Index of a signal to plot. If undefined, the first ECG in the
            batch is used.
        start : int, optional
            The start point of the displayed part of the signal (in seconds).
        end : int, optional
            The end point of the displayed part of the signal (in seconds).
        annot : str, optional
            If not ``None``, specifies attribute that stores annotation
            obtained from ``cardio.models.HMModel``.
        subplot_size : tuple
            Width and height of each subplot in inches.

        Raises
        ------
        ValueError
            If the chosen signal is not two-dimensional.
        """
        i = 0 if index is None else self.get_pos(None, "signal", index)
        signal, meta = self.signal[i], self.meta[i]
        self._check_2d(signal)

        fs = meta["fs"]
        num_channels = signal.shape[0]
        start = np.int(start * fs)
        end = signal.shape[1] if end is None else np.int(end * fs)

        figsize = (subplot_size[0], subplot_size[1] * num_channels)
        _, axes = plt.subplots(num_channels, 1, squeeze=False, figsize=figsize)
        for channel, (ax,) in enumerate(axes):
            lead_name = "undefined" if meta["signame"][channel] == "None" else meta["signame"][channel]
            units = "undefined" if meta["units"][channel] is None else meta["units"][channel]
            ax.plot((np.arange(start, end) / fs), signal[channel, start:end])
            ax.set_title("Lead name: {}".format(lead_name))
            ax.set_xlabel("Time (sec)")
            ax.set_ylabel("Amplitude ({})".format(units))
            ax.grid("on", which="major")

        if annot and hasattr(self, annot):
            def fill_segments(segment_states, color):
                """Fill ECG segments with a given color."""
                starts, ends = bt.find_intervals_borders(signal_states, segment_states)
                for start_t, end_t in zip((starts + start) / fs, (ends + start) / fs):
                    for (ax,) in axes:
                        ax.axvspan(start_t, end_t, color=color, alpha=0.3)

            signal_states = getattr(self, annot)[i][start:end]
            fill_segments(bt.QRS_STATES, "red")
            fill_segments(bt.P_STATES, "green")
            fill_segments(bt.T_STATES, "blue")
        plt.tight_layout()
        plt.show()

    # Batch processing

    @classmethod
    def merge(cls, batches, batch_size=None):
        """Concatenate a list of ``EcgBatch`` instances and split the result
        into two batches of sizes ``batch_size`` and ``sum(lens of batches) -
        batch_size`` respectively.

        Parameters
        ----------
        batches : list
            List of ``EcgBatch`` instances.
        batch_size : positive int, optional
            Length of the first resulting batch. If ``None``, equals the
            length of the concatenated batch.

        Returns
        -------
        new_batch : EcgBatch
            Batch of no more than ``batch_size`` first items from the
            concatenation of input batches. Contains a deep copy of input
            batches' data.
        rest_batch : EcgBatch
            Batch of the remaining items. Contains a deep copy of input
            batches' data.

        Raises
        ------
        ValueError
            If ``batch_size`` is non-positive or non-integer.
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
        new_batch._data = tuple(comp[:batch_size] for comp in data)  # pylint: disable=protected-access, attribute-defined-outside-init, line-too-long
        if total_len <= batch_size:
            rest_batch = None
        else:
            rest_indices = indices[batch_size:]
            rest_batch = cls(ds.DatasetIndex(rest_indices), unique_labels=batches[0].unique_labels)
            rest_batch._data = tuple(comp[batch_size:] for comp in data)  # pylint: disable=protected-access, attribute-defined-outside-init, line-too-long
        return new_batch, rest_batch

    # Versatile components processing

    @ds.action
    def apply_transform(self, func, *args, src="signal", dst="signal", **kwargs):
        """Apply a function to each item in the batch.

        Parameters
        ----------
        func : callable
            A function to apply. Must accept an item of ``src`` as its first
            argument if ``src`` is not ``None``.
        src : str, array-like or ``None``, optional
            The source to get the data from. If ``src`` is ``str``, it is
            treated as the batch attribute or component name. Defaults to
            ``signal`` component.
        dst : str, writeable array-like or ``None``, optional
            The source to put the result in. If ``dst`` is ``str``, it is
            treated as the batch attribute or component name. Defaults to
            ``signal`` component.
        args : misc
            Any additional positional arguments to ``func``.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        batch : EcgBatch
            Transformed batch. If ``dst`` is ``str``, the corresponding
            attribute or component is changed inplace.
        """
        if isinstance(dst, str) and not hasattr(self, dst):
            setattr(self, dst, np.array([None] * len(self.index)))
        return super().apply_transform(func, *args, src=src, dst=dst, **kwargs)

    def _init_component(self, *args, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args
        dst = kwargs.get("dst")
        if dst is None:
            raise KeyError("dst argument must be specified")
        if not hasattr(self, dst):
            setattr(self, dst, np.array([None] * len(self.index)))
        return self.indices

    @ds.action
    @ds.inbatch_parallel(init="_init_component", src="signal", dst="signal", target="threads")
    def apply_to_each_channel(self, index, func, *args, src="signal", dst="signal", **kwargs):
        """Apply a function to each slice of a signal over the axis 0
        (typically the channel axis).

        Parameters
        ----------
        func : callable
            A function to apply. Must accept a signal as its first argument.
        src : str, optional
            Batch attribute or component name to get the data from. Defaults
            to ``signal`` component.
        dst : str, optional
            Batch attribute or component name to put the result in. Defaults
            to ``signal`` component.
        args : misc
            Any additional positional arguments to ``func``.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        batch : EcgBatch
            Transformed batch. Changes ``dst`` attribute or component.
        """
        i = self.get_pos(None, src, index)
        src_data = getattr(self, src)[i]
        dst_data = np.array([func(slc, *args, **kwargs) for slc in src_data])
        getattr(self, dst)[i] = dst_data

    # Labels processing

    def _filter_batch(self, keep_mask):
        """Drop elements from a batch with corresponding ``False`` values in
        ``keep_mask``.

        This method creates a new batch and updates only components and
        ``unique_labels`` attribute. The information stored in other
        attributes will be lost.

        Parameters
        ----------
        keep_mask : bool 1-D ndarray
            Filtering mask.

        Returns
        -------
        batch : same class as self
            Filtered batch.

        Raises
        ------
        SkipBatchException
            If all batch data was dropped. If the batch is created by a
            ``pipeline``, its processing will be stopped and the ``pipeline``
            will create the next batch.
        """
        indices = self.indices[keep_mask]
        if len(indices) == 0:
            raise ds.SkipBatchException("All batch data was dropped")
        batch = self.__class__(ds.DatasetIndex(indices), unique_labels=self.unique_labels)
        for component in self.components:
            setattr(batch, component, getattr(self, component)[keep_mask])
        return batch

    @ds.action
    def drop_labels(self, drop_list):
        """Drop elements whose labels are in ``drop_list``.

        This method creates a new batch and updates only components and
        ``unique_labels`` attribute. The information stored in other
        attributes will be lost.

        Parameters
        ----------
        drop_list : list
            Labels to be dropped from a batch.

        Returns
        -------
        batch : EcgBatch
            Filtered batch. Creates a new ``EcgBatch`` instance.

        Raises
        ------
        SkipBatchException
            If all batch data was dropped. If the batch is created by a
            ``pipeline``, its processing will be stopped and the ``pipeline``
            will create the next batch.
        """
        drop_arr = np.asarray(drop_list)
        self.unique_labels = np.setdiff1d(self.unique_labels, drop_arr)
        keep_mask = ~np.in1d(self.target, drop_arr)
        return self._filter_batch(keep_mask)

    @ds.action
    def keep_labels(self, keep_list):
        """Drop elements whose labels are not in ``keep_list``.

        This method creates a new batch and updates only components and
        ``unique_labels`` attribute. The information stored in other
        attributes will be lost.

        Parameters
        ----------
        keep_list : list
            Labels to be kept in a batch.

        Returns
        -------
        batch : EcgBatch
            Filtered batch. Creates a new ``EcgBatch`` instance.

        Raises
        ------
        SkipBatchException
            If all batch data was dropped. If the batch is created by a
            ``pipeline``, its processing will be stopped and the ``pipeline``
            will create the next batch.
        """
        keep_arr = np.asarray(keep_list)
        self.unique_labels = np.intersect1d(self.unique_labels, keep_arr)
        keep_mask = np.in1d(self.target, keep_arr)
        return self._filter_batch(keep_mask)

    @ds.action
    def rename_labels(self, rename_dict):
        """Rename labels with corresponding values from ``rename_dict``.

        Parameters
        ----------
        rename_dict : dict
            Dictionary containing ``(old label : new label)`` pairs.

        Returns
        -------
        batch : EcgBatch
            Batch with renamed labels. Changes ``self.target`` inplace.
        """
        self.unique_labels = np.array(sorted({rename_dict.get(t, t) for t in self.unique_labels}))
        self.target = np.array([rename_dict.get(t, t) for t in self.target])
        return self

    @ds.action
    def binarize_labels(self):
        """Binarize labels in a batch in a one-vs-all fashion.

        Returns
        -------
        batch : EcgBatch
            Batch with binarized labels. Changes ``self.target`` inplace.
        """
        self.target = self.label_binarizer.transform(self.target)
        return self

    # Channels processing

    @ds.inbatch_parallel(init="indices", target="threads")
    def _filter_channels(self, index, names=None, indices=None, invert_mask=False):
        """Build and apply a boolean mask for each channel of a signal based
        on provided channels ``names`` and ``indices``.

        Mask value for a channel is set to ``True`` if its name or index is
        contained in ``names`` or ``indices`` respectively. The mask can be
        inverted before its application if ``invert_mask`` flag is set to
        ``True``.

        Parameters
        ----------
        names : str or list or tuple, optional
            Channels names used to construct the mask.
        indices : int or list or tuple, optional
            Channels indices used to construct the mask.
        invert_mask : bool, optional
            Specifies whether to invert the mask before its application.

        Returns
        -------
        batch : EcgBatch
            Batch with filtered channels. Changes ``self.signal`` and
            ``self.meta`` inplace.

        Raises
        ------
        ValueError
            If both ``names`` and ``indices`` are empty.
        ValueError
            If all channels should be dropped.
        """
        i = self.get_pos(None, "signal", index)
        channels_names = np.asarray(self.meta[i]["signame"])
        mask = np.zeros_like(channels_names, dtype=np.bool)
        if names is None and indices is None:
            raise ValueError("Both names and indices cannot be empty")
        if names is not None:
            names = np.asarray(names)
            mask |= np.in1d(channels_names, names)
        if indices is not None:
            indices = np.asarray(indices)
            mask |= np.array([i in indices for i in range(len(channels_names))])
        if invert_mask:
            mask = ~mask
        if np.sum(mask) == 0:
            raise ValueError("All channels cannot be dropped")
        self.signal[i] = self.signal[i][mask]
        self.meta[i]["signame"] = channels_names[mask]

    @ds.action
    def drop_channels(self, names=None, indices=None):
        """Drop channels whose names are in ``names`` or whose indices are in
        ``indices``.

        Parameters
        ----------
        names : str or list or tuple, optional
            Names of channels to be dropped from a batch.
        indices : int or list or tuple, optional
            Indices of channels to be dropped from a batch.

        Returns
        -------
        batch : EcgBatch
            Batch with dropped channels. Changes ``self.signal`` and
            ``self.meta`` inplace.

        Raises
        ------
        ValueError
            If both ``names`` and ``indices`` are empty.
        ValueError
            If all channels should be dropped.
        """
        return self._filter_channels(names, indices, invert_mask=True)

    @ds.action
    def keep_channels(self, names=None, indices=None):
        """Drop channels whose names are not in ``names`` and whose indices
        are not in ``indices``.

        Parameters
        ----------
        names : str or list or tuple, optional
            Names of channels to be kept in a batch.
        indices : int or list or tuple, optional
            Indices of channels to be kept in a batch.

        Returns
        -------
        batch : EcgBatch
            Batch with dropped channels. Changes ``self.signal`` and
            ``self.meta`` inplace.

        Raises
        ------
        ValueError
            If both ``names`` and ``indices`` are empty.
        ValueError
            If all channels should be dropped.
        """
        return self._filter_channels(names, indices, invert_mask=False)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def rename_channels(self, index, rename_dict):
        """Rename channels with corresponding values from ``rename_dict``.

        Parameters
        ----------
        rename_dict : dict
            Dictionary containing ``(old channel name : new channel name)``
            pairs.

        Returns
        -------
        batch : EcgBatch
            Batch with renamed channels. Changes ``self.meta`` inplace.
        """
        i = self.get_pos(None, "signal", index)
        old_names = self.meta[i]["signame"]
        new_names = np.array([rename_dict.get(name, name) for name in old_names], dtype=object)
        self.meta[i]["signame"] = new_names

    # Signal processing

    @ds.action
    def convolve_signals(self, kernel, padding_mode="edge", axis=-1, **kwargs):
        """Convolve signals with given ``kernel``.

        Parameters
        ----------
        kernel : 1-D array_like
            Convolution kernel.
        padding_mode : str or function, optional
            ``np.pad`` padding mode.
        axis : int, optional
            Axis along which signals are sliced. Default value is -1.
        kwargs : misc
            Any additional named arguments to ``np.pad``.

        Returns
        -------
        batch : EcgBatch
            Convolved batch. Changes ``self.signal`` inplace.

        Raises
        ------
        ValueError
            If ``kernel`` is not one-dimensional or has non-numeric ``dtype``.
        """
        for i in range(len(self.signal)):
            self.signal[i] = bt.convolve_signals(self.signal[i], kernel, padding_mode, axis, **kwargs)
        return self

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def band_pass_signals(self, index, low=None, high=None, axis=-1):
        """Reject frequencies outside a given range.

        Parameters
        ----------
        low : positive float, optional
            High-pass filter cutoff frequency (in Hz).
        high : positive float, optional
            Low-pass filter cutoff frequency (in Hz).
        axis : int, optional
            Axis along which signals are sliced. Default value is -1.

        Returns
        -------
        batch : EcgBatch
            Filtered batch. Changes ``self.signal`` inplace.
        """
        i = self.get_pos(None, "signal", index)
        self.signal[i] = bt.band_pass_signals(self.signal[i], self.meta[i]["fs"], low, high, axis)

    @ds.action
    def drop_short_signals(self, min_length, axis=-1):
        """Drop short signals from a batch.

        Parameters
        ----------
        min_length : positive int
            Minimal signal length.
        axis : int, optional
            Axis along which length is calculated. Default value is -1.

        Returns
        -------
        batch : EcgBatch
            Filtered batch. Creates a new ``EcgBatch`` instance.
        """
        keep_mask = np.array([sig.shape[axis] >= min_length for sig in self.signal])
        return self._filter_batch(keep_mask)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def flip_signals(self, index, window_size=None, threshold=0):
        """Flip 2-D signals whose R-peaks are directed downwards.

        Each element of ``self.signal`` must be a 2-D ndarray. Signals are
        flipped along axis 1 (signal axis). For each subarray of
        ``window_size`` length skewness is calculated and compared with
        ``threshold`` to decide whether this subarray should be flipped or
        not. Then the mode of the result is calculated to make the final
        decision.

        Parameters
        ----------
        window_size : int, optional
            Signal is split into K subarrays of ``window_size`` length. If it
            is not possible, data in the end of the signal is removed. If
            ``window_size`` is not given, the whole array is checked without
            splitting.
        threshold : float, optional
            If skewness of a subarray is less than the ``threshold``, it
            "votes" for flipping the signal. Default value is 0.

        Returns
        -------
        batch : EcgBatch
            Batch with flipped signals. Changes ``self.signal`` inplace.

        Raises
        ------
        ValueError
            If given signal is not two-dimensional.
        """
        i = self.get_pos(None, "signal", index)
        self._check_2d(self.signal[i])
        sig = bt.band_pass_signals(self.signal[i], self.meta[i]["fs"], low=5, high=50)
        sig = bt.convolve_signals(sig, kernels.gaussian(11, 3))

        if window_size is None:
            window_size = sig.shape[1]

        number_of_splits = sig.shape[1] // window_size
        sig = sig[:, : window_size * number_of_splits]

        splits = np.split(sig, number_of_splits, axis=-1)
        votes = [np.where(scipy.stats.skew(subseq, axis=-1) < threshold, -1, 1).reshape(-1, 1) for subseq in splits]
        mode_of_votes = scipy.stats.mode(votes)[0].reshape(-1, 1)
        self.signal[i] *= mode_of_votes

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def slice_signals(self, index, selection_object):
        """Perform indexing or slicing of signals in a batch. Allows basic
        ``NumPy`` indexing and slicing along with advanced indexing.

        Parameters
        ----------
        selection_object : slice or int or a tuple of slices and ints
            An object that is used to slice signals.

        Returns
        -------
        batch : EcgBatch
            Batch with sliced signals. Changes ``self.signal`` inplace.
        """
        i = self.get_pos(None, "signal", index)
        self.signal[i] = self.signal[i][selection_object]

    @staticmethod
    def _pad_signal(signal, length, pad_value):
        """Pad signal with ``pad_value`` to the left along axis 1 (signal
        axis).

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
        """Get segmentation step or number of segments for a given signal.

        Parameters
        ----------
        arg : int or dict
            Segmentation step or number of segments.
        arg_name : str
            Argument name.
        target : hashable
            Signal target.

        Returns
        -------
        arg : positive int
            Segmentation step or number of segments for given signal.

        Raises
        ------
        KeyError
            If ``arg`` dict has no ``target`` key.
        ValueError
            If ``arg`` is not int or dict.
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

        Raises
        ------
        ValueError
            If:
                * given signal is not two-dimensional,
                * ``arg`` is not int or dict,
                * ``length`` or ``arg`` for a given signal is negative or
                  non-integer.
        KeyError
            If ``arg`` dict has no ``target`` key.
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
    def split_signals(self, index, length, step, pad_value=0):
        """Split 2-D signals along axis 1 (signal axis) with given ``length``
        and ``step``.

        If signal length along axis 1 is less than ``length``, it is padded to
        the left with ``pad_value``.

        Notice, that each resulting signal will be a 3-D ndarray of shape
        ``[n_segments, n_channels, length]``. If you would like to get a
        number of 2-D signals of shape ``[n_channels, length]`` as a result,
        you need to apply ``unstack_signals`` method then.

        Parameters
        ----------
        length : positive int
            Length of each segment along axis 1.
        step : positive int or dict
            Segmentation step. If ``step`` is dict, segmentation step is
            fetched by signal's target key.
        pad_value : float, optional
            Padding value. Defaults to 0.

        Returns
        -------
        batch : EcgBatch
            Batch of split signals. Changes ``self.signal`` inplace.

        Raises
        ------
        ValueError
            If:
                * given signal is not two-dimensional,
                * ``step`` is not int or dict,
                * ``length`` or ``step`` for a given signal is negative or
                  non-integer.
        KeyError
            If ``step`` dict has no signal's target key.
        """
        i = self.get_pos(None, "signal", index)
        step = self._check_segmentation_args(self.signal[i], self.target[i], length, step, "step size")
        if self.signal[i].shape[1] < length:
            tmp_sig = self._pad_signal(self.signal[i], length, pad_value)
            self.signal[i] = tmp_sig[np.newaxis, ...]
        else:
            self.signal[i] = bt.split_signals(self.signal[i], length, step)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def random_split_signals(self, index, length, n_segments, pad_value=0):
        """Split 2-D signals along axis 1 (signal axis) ``n_segments`` times
        with random start position and given ``length``.

        If signal length along axis 1 is less than ``length``, it is padded to
        the left with ``pad_value``.

        Notice, that each resulting signal will be a 3-D ndarray of shape
        ``[n_segments, n_channels, length]``. If you would like to get a
        number of 2-D signals of shape ``[n_channels, length]`` as a result,
        you need to apply ``unstack_signals`` method then.

        Parameters
        ----------
        length : positive int
            Length of each segment along axis 1.
        n_segments : positive int or dict
            Number of segments. If ``n_segments`` is dict, number of segments
            is fetched by signal's target key.
        pad_value : float, optional
            Padding value. Defaults to 0.

        Returns
        -------
        batch : EcgBatch
            Batch of split signals. Changes ``self.signal`` inplace.

        Raises
        ------
        ValueError
            If:
                * given signal is not two-dimensional,
                * ``n_segments`` is not int or dict,
                * ``length`` or ``n_segments`` for a given signal is negative
                  or non-integer.
        KeyError
            If ``n_segments`` dict has no signal's target key.
        """
        i = self.get_pos(None, "signal", index)
        n_segments = self._check_segmentation_args(self.signal[i], self.target[i], length,
                                                   n_segments, "number of segments")
        if self.signal[i].shape[1] < length:
            tmp_sig = self._pad_signal(self.signal[i], length, pad_value)
            self.signal[i] = np.tile(tmp_sig, (n_segments, 1, 1))
        else:
            self.signal[i] = bt.random_split_signals(self.signal[i], length, n_segments)

    @ds.action
    def unstack_signals(self):
        """Create a new batch in which each signal's element along axis 0 is
        considered as a separate signal.

        This method creates a new batch and updates only components and
        ``unique_labels`` attribute. Signal's data from non-``signal``
        components is duplicated using a deep copy for each of the resulting
        signals. The information stored in other attributes will be lost.

        Returns
        -------
        batch : same class as self
            Batch with split signals and duplicated other components.

        Examples
        --------
        >>> batch.signal
        array([array([[ 0,  1,  2,  3],
                      [ 4,  5,  6,  7],
                      [ 8,  9, 10, 11]])],
              dtype=object)

        >>> batch = batch.unstack_signals()
        >>> batch.signal
        array([array([0, 1, 2, 3]),
               array([4, 5, 6, 7]),
               array([ 8,  9, 10, 11])],
              dtype=object)
        """
        n_reps = [sig.shape[0] for sig in self.signal]
        signal = np.array([channel for signal in self.signal for channel in signal] + [None])[:-1]
        index = ds.DatasetIndex(np.arange(len(signal)))
        batch = self.__class__(index, unique_labels=self.unique_labels)
        batch.signal = signal
        for component_name in set(self.components) - {"signal"}:
            val = []
            component = getattr(self, component_name)
            is_object_dtype = (component.dtype.kind == "O")
            for elem, n in zip(component, n_reps):
                for _ in range(n):
                    val.append(copy.deepcopy(elem))
            if is_object_dtype:
                val = np.array(val + [None])[:-1]
            else:
                val = np.array(val)
            setattr(batch, component_name, val)
        return batch

    def _safe_fs_resample(self, index, fs):
        """Resample 2-D signal along axis 1 (signal axis) to given sampling
        rate.

        New sampling rate is guaranteed to be positive float.

        Parameters
        ----------
        fs : positive float
            New sampling rate.

        Raises
        ------
        ValueError
            If given signal is not two-dimensional.
        """
        i = self.get_pos(None, "signal", index)
        self._check_2d(self.signal[i])
        new_len = max(1, int(fs * self.signal[i].shape[1] / self.meta[i]["fs"]))
        self.meta[i]["fs"] = fs
        self.signal[i] = bt.resample_signals(self.signal[i], new_len)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def resample_signals(self, index, fs):
        """Resample 2-D signals along axis 1 (signal axis) to given sampling
        rate.

        Parameters
        ----------
        fs : positive float
            New sampling rate.

        Returns
        -------
        batch : EcgBatch
            Resampled batch. Changes ``self.signal`` and ``self.meta``
            inplace.

        Raises
        ------
        ValueError
            If given signal is not two-dimensional.
        ValueError
            If ``fs`` is negative or non-numeric.
        """
        if fs <= 0:
            raise ValueError("Sampling rate must be a positive float")
        self._safe_fs_resample(index, fs)

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def random_resample_signals(self, index, distr, **kwargs):
        """Resample 2-D signals along axis 1 (signal axis) to a new sampling
        rate, sampled from a given distribution.

        If new sampling rate is negative, the signal is left unchanged.

        Parameters
        ----------
        distr : str or callable
            ``NumPy`` distribution name or a callable to sample from.
        kwargs : misc
            Distribution parameters.

        Returns
        -------
        batch : EcgBatch
            Resampled batch. Changes ``self.signal`` and ``self.meta``
            inplace.

        Raises
        ------
        ValueError
            If given signal is not two-dimensional.
        ValueError
            If ``distr`` is not a string or a callable.
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

    # Complex ECG processing

    @ds.action
    @ds.inbatch_parallel(init="_init_component", src="signal", dst="signal", target="threads")
    def spectrogram(self, index, *args, src="signal", dst="signal", **kwargs):
        """Compute a spectrogram for each slice of a signal over the axis 0
        (typically the channel axis).

        This method is a wrapper around ``scipy.signal.spectrogram``, that
        accepts the same arguments, except the ``fs`` which is substituted
        automatically from signal's meta. The method returns only the
        spectrogram itself.

        Parameters
        ----------
        src : str, optional
            Batch attribute or component name to get the data from.
        dst : str, optional
            Batch attribute or component name to put the result in.
        args : misc
            Any additional positional arguments to
            ``scipy.signal.spectrogram``.
        kwargs : misc
            Any additional named arguments to ``scipy.signal.spectrogram``.

        Returns
        -------
        batch : EcgBatch
            Transformed batch. Changes ``dst`` attribute or component.
        """
        i = self.get_pos(None, src, index)
        fs = self.meta[i]["fs"]
        src_data = getattr(self, src)[i]
        dst_data = np.array([scipy.signal.spectrogram(slc, fs, *args, **kwargs)[-1] for slc in src_data])
        getattr(self, dst)[i] = dst_data

    @ds.action
    @ds.inbatch_parallel(init="_init_component", src="signal", dst="signal", target="threads")
    def standardize(self, index, axis=None, eps=1e-10, *, src="signal", dst="signal"):
        """Standardize data along specified axes by removing the mean and
        scaling to unit variance.

        Parameters
        ----------
        axis : ``None`` or int or tuple of ints, optional
            Axis or axes along which standardization is performed. The default
            is to compute for the flattened array.
        eps: float
            Small addition to avoid division by zero.
        src : str, optional
            Batch attribute or component name to get the data from.
        dst : str, optional
            Batch attribute or component name to put the result in.

        Returns
        -------
        batch : EcgBatch
            Transformed batch. Changes ``dst`` attribute or component.
        """
        i = self.get_pos(None, src, index)
        src_data = getattr(self, src)[i]
        dst_data = ((src_data - np.mean(src_data, axis=axis, keepdims=True)) /
                    np.std(src_data, axis=axis, keepdims=True) + eps)
        getattr(self, dst)[i] = dst_data

    @ds.action
    @ds.inbatch_parallel(init="indices", target="threads")
    def calc_ecg_parameters(self, index, src=None):
        """Calculate ECG report parameters and write them to the ``meta``
        component.

        Calculates PQ, QT, QRS intervals along with their borders and the
        heart rate value based on the annotation and writes them to the
        ``meta`` component.

        Parameters
        ----------
        src : str
            Batch attribute or component name to get the annotation from.

        Returns
        -------
        batch : EcgBatch
            Batch with report parameters stored in the ``meta`` component.

        Raises
        ------
        ValueError
            If ``src`` is ``None`` or is not an attribute of a batch.
        """
        if not (src and hasattr(self, src)):
            raise ValueError("Batch does not have an attribute or component {}!".format(src))

        i = self.get_pos(None, "signal", index)
        src_data = getattr(self, src)[i]

        self.meta[i]["hr"] = bt.calc_hr(self.signal[i],
                                        src_data,
                                        np.float64(self.meta[i]["fs"]),
                                        bt.R_STATE)

        self.meta[i]["pq"] = bt.calc_pq(src_data,
                                        np.float64(self.meta[i]["fs"]),
                                        bt.P_STATES,
                                        bt.Q_STATE,
                                        bt.R_STATE)

        self.meta[i]["qt"] = bt.calc_qt(src_data,
                                        np.float64(self.meta[i]["fs"]),
                                        bt.T_STATES,
                                        bt.Q_STATE,
                                        bt.R_STATE)

        self.meta[i]["qrs"] = bt.calc_qrs(src_data,
                                          np.float64(self.meta[i]["fs"]),
                                          bt.S_STATE,
                                          bt.Q_STATE,
                                          bt.R_STATE)

        self.meta[i]["qrs_segments"] = np.vstack(bt.find_intervals_borders(src_data,
                                                                           bt.QRS_STATES))

        self.meta[i]["p_segments"] = np.vstack(bt.find_intervals_borders(src_data,
                                                                         bt.P_STATES))

        self.meta[i]["t_segments"] = np.vstack(bt.find_intervals_borders(src_data,
                                                                         bt.T_STATES))
