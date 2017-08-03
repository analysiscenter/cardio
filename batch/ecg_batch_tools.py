"""Сontains ECG processing tools."""

import os
import numpy as np

import wfdb
from numba import njit
from scipy.signal import resample_poly


def load_wfdb(path):
    """
    Load signal and meta, loading of annotation should be added
    """
    path = os.path.splitext(path)[0]
    record = wfdb.rdsamp(path)
    signal = record.__dict__.pop("p_signals").T
    meta = record.__dict__
    return [signal, {}, meta, None]


def load_npz(path):
    """
    Load signal and meta, loading of annotation should be added
    """
    data = np.load(path)
    if set(data.files) != {"signal", "annotation", "meta", "target"}:
        raise ValueError("File " + path + " has wrong components")
    signal = data["signal"]
    annotation = data["annotation"].item()
    meta = data["meta"].item()
    target = data["target"].item()
    return [signal, annotation, meta, target]


def dump_npz(item, path):
    """
    Save ecg in a separate file as 'path/<index>.<fmt>'
    """
    np.savez(path, signal=item.signal, annotation=item.annotation, meta=item.meta, target=item.target)


def convolve(signals, kernel, axis=-1, padding_mode="edge", **kwargs):
    """Convolve signals with given kernel.

    Parameters
    ----------
    signals : ndarray
        Signals to convolve.
    kernel : array_like
        Convolution kernel.
    axis : int
        Axis along which signal is sliced.
    padding_mode : str or function
        np.pad padding mode.
    **kwargs :
        Any additional named argments to np.pad.

    Returns
    -------
    signals : ndarray
        Convolved signals.
    """
    kernel = np.asarray(kernel)
    if len(kernel.shape) == 0:
        kernel = kernel.ravel()
    if len(kernel.shape) != 1:
        raise ValueError("Kernel must be 1-D array")
    if not np.issubdtype(kernel.dtype, np.number):
        raise ValueError("Kernel must have numeric dtype")
    pad = len(kernel) // 2

    def conv_func(x):
        """Convolve padded signal."""
        x_pad = np.pad(x, pad, padding_mode, **kwargs)
        conv = np.convolve(x_pad, kernel, "same")
        if pad > 0:
            conv = conv[pad:-pad]
        return conv

    signals = np.apply_along_axis(conv_func, arr=signals, axis=axis)
    return signals


def band_pass_filter(signals, freq, axis=-1, low=None, high=None):
    """Reject frequencies outside given range.

    Parameters
    ----------
    signals : ndarray
        Signals to filter.
    freq : positive float
        Sampling rate.
    axis : int
        Axis along which signal is sliced.
    low : positive float
        High-pass filter cutoff frequency (Hz).
    high : positive float
        Low-pass filter cutoff frequency (Hz).

    Returns
    -------
    signals : ndarray
        Filtered signals.
    """
    if freq <= 0:
        raise ValueError("Sampling rate must be a positive float")
    sig_rfft = np.fft.rfft(signals, axis=axis)
    sig_freq = np.fft.rfftfreq(signals.shape[axis], 1 / freq)
    mask = np.zeros(len(sig_freq), dtype=bool)
    if low is not None:
        mask |= (sig_freq <= low)
    if high is not None:
        mask |= (sig_freq >= high)
    slc = [slice(None)] * signals.ndim
    slc[axis] = mask
    sig_rfft[slc] = 0
    return np.fft.irfft(sig_rfft, n=signals.shape[axis], axis=axis)


@njit(nogil=True)
def get_pos_of_max(pred):
    '''
    Returns position of maximal element in a row.

    Arguments
    pred: 2d array.
    '''
    labels = np.zeros(pred.shape)
    for i in range(len(labels)):
        labels[i, pred[i].argmax()] = 1
    return labels

def resample_signal(signal, annot, meta, index, new_fs):
    """
    Resample signal along axis=1 to new sampling rate. Retruns resampled signal with modified meta.
    Resampling of annotation will be implemented in the future.

    Arguments
    signal, annot, meta, index: componets of ecg signal.
    new_fs: target signal sampling rate in Hz.
    """
    fs = meta['fs']
    new_len = int(new_fs * len(signal[0]) / fs)
    if new_len < 1:
        print('Error: new_len should be >= 1. Try to change new_fs')
        return None
    signal = resample_poly(signal, new_len, len(signal[0]), axis=1)
    out_meta = {**meta, 'fs': new_fs, 'siglen': new_len}
    return [signal, annot, out_meta, index]

def segment_signal(signal, annot, meta, index, length, step, pad, return_copy):
    """
    Segment signal along axis=1 with constant step to segments with constant length.
    If signal is shorter than target segment length, signal is zero-padded on the left if
    pad is True or raise ValueError if pad is False.
    Segmentation of annotation will be implemented in the future.

    Arguments
    signal, annot, meta, index: componets of ecg signal.
    length: length of segment.
    step: step along axis=1.
    pad: whether to apply zero-padding to short signals.
    return_copy: if True, a copy of segments is returned and segments become intependent. If False,
                 segments are not independent, but segmentation runtime becomes almost indepentent on
                 signal length.

    Attention: segmentation of meta and annotation is not implemented yet.
    """
    if signal.ndim != 2:
        raise ValueError('Signal should have ndim = 2, found ndim = {0}'.format(signal.ndim))

    if signal.shape[1] < length:
        if pad:
            pad_len = length - signal.shape[1]
            segments = np.lib.pad(signal, ((0, 0), (pad_len, 0)),
                                  'constant', constant_values=(0, 0))[np.newaxis, :, :]
        else:
            raise ValueError('Signal is shorter than segment length: %i < %i'
                             % (signal.shape[1], length))
    else:
        shape = signal.shape[:-1] + (signal.shape[-1] - length + 1, length)
        strides = signal.strides + (signal.strides[-1],)
        segments = np.lib.stride_tricks.as_strided(signal, shape=shape,
                                                   strides=strides)[:, ::step, :]
        segments = np.transpose(segments, (1, 0, 2))

    _ = annot
    if return_copy:
        segments = segments.copy()
    out_meta = {**meta, 'siglen': length}
    return [segments, {}, out_meta, index]

def augment_fs_signal(signal, annot, meta, index, distr, params):
    '''
    Augmentation of signal to random sampling rate. New sampling rate is sampled
    from given probability distribution with specified parameters.

    Arguments
    signal, annot, meta, index: componets of ecg signal.
    distr: distribution type, either a name of any distribution from np.random, or
           callable, or 'none', or 'delta'.
    params: dict of parameters and values for distr. ignored if distr='none'.
    '''
    if hasattr(np.random, distr):
        distr_fn = getattr(np.random, distr)
        new_fs = distr_fn(**params)
    elif callable(distr):
        new_fs = distr_fn(**params)
    elif distr == 'none':
        return [signal, annot, meta, index]
    elif distr == 'delta':
        new_fs = params['loc']
    if new_fs <= 0:
        return None
    return resample_signal(signal, annot, meta, index, new_fs)

def augment_fs_signal_mult(signal, annot, meta, index, list_of_distr):
    '''
    Multiple augmentation of signal to random sampling rates. New sampling rates are sampled
    from list of probability distributions with specified parameters.

    Arguments
    signal, annot, meta, index: componets of ecg signal.
    list_of_distr: list of tuples (distr, params). See augment_fssignal for details.
    '''
    res = [augment_fs_signal(signal, annot, meta, index, distr_type, params)
           for (distr_type, params) in list_of_distr]
    out_sig = [x[0] for x in res]
    out_annot = [x[1] for x in res]
    out_meta = [x[2] for x in res]
    out_sig.append([])
    return [np.array(out_sig)[:-1], out_annot, out_meta, index]

def predict_hmm_classes(signal, annot, meta, index, model):
    '''
    Get hmm predicted classes
    '''
    res = np.array(model.predict(signal.T)).reshape((1, -1))
    annot.update({'hmm_predict': res})
    return [signal, annot, meta, index]

def get_gradient(signal, annot, meta, index, order):
    '''
    Compute derivative of given order

    Arguments
    signal, annot, meta, index: componets of ecg signal.
    order: order of derivative to compute.
    '''
    grad = np.gradient(signal, axis=1)
    for i in range(order - 1):#pylint: disable=unused-variable
        grad = np.gradient(grad, axis=1)
    annot.update({'grad_{0}'.format(order): grad})
    return [signal, annot, meta, index]

def convolve_layer(signal, annot, meta, index, layer, kernel):
    '''
    Convolve squared data with kernel

    Arguments
    signal, annot, meta, index: componets of ecg signal.
    layer: name of layer that will be convolved. Can be 'signal' or key from annotation keys.
    kernel: kernel for convolution.
    '''
    if layer == 'signal':
        data = signal
    else:
        data = annot[layer]
    res = np.apply_along_axis(np.convolve, 1, data**2, v=kernel, mode='same')
    annot.update({layer + '_conv': res})
    return [signal, annot, meta, index]

def merge_list_of_layers(signal, annot, meta, index, list_of_layers):
    '''
    Merge layers from list of layers to signal

    Arguments
    signal, annot, meta, index: componets of ecg signal.
    list_of_layers: list of name layers that will be merged. Can contain 'signal' or keys from annotation.
    '''
    res = []
    for layer in list_of_layers:
        if layer == 'signal':
            data = signal
        else:
            data = annot[layer]
        res.append(data)
    res = np.concatenate(res, axis=0)[np.newaxis, :, :]
    return [res, annot, meta, index]
