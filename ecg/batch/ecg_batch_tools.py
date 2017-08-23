"""Сontains ECG processing tools."""

import os
import numpy as np

import pywt
from numba import njit

import wfdb


def load_wfdb(path, components):
    """Load given components from wfdb file.

    Parameters
    ----------
    path : str
        Path to .hea file.
    components : iterable
        Components to load.

    Returns
    -------
    signal_data : list
        List of signal components.
    """
    path = os.path.splitext(path)[0]
    record = wfdb.rdsamp(path)
    signal = record.__dict__.pop("p_signals").T
    meta = record.__dict__
    data = {"signal": signal,
            "annotation": {},
            "meta": meta}
    return [data[comp] for comp in components]


@njit(nogil=True)
def segment_signals(signals, length, step):
    """Segment signals along axis 1 with given length and step.

    Parameters
    ----------
    signals : 2-D ndarray
        Signals to segment.
    length : positive int
        Length of each segment along axis 1.
    step : positive int
        Segmentation step.

    Returns
    -------
    signals : 3-D ndarray
        Segmented signals.
    """
    res = np.empty(((signals.shape[1] - length) // step + 1, signals.shape[0], length), dtype=signals.dtype)
    for i in range(res.shape[0]):
        res[i, :, :] = signals[:, i * step : i * step + length]
    return res


@njit(nogil=True)
def random_segment_signals(signals, length, n_segments):
    """Segment signals along axis 1 n_segments times with random start position and given length.

    Parameters
    ----------
    signals : 2-D ndarray
        Signals to segment.
    length : positive int
        Length of each segment along axis 1.
    n_segments : positive int
        Number of segments.

    Returns
    -------
    signals : 3-D ndarray
        Segmented signals.
    """
    res = np.empty((n_segments, signals.shape[0], length), dtype=signals.dtype)
    for i in range(res.shape[0]):
        ix = np.random.randint(0, signals.shape[1] - length + 1)
        res[i, :, :] = signals[:, ix : ix + length]
    return res


@njit(nogil=True)
def resample_signals(signals, new_length):
    """Resample signals to new length along axis 1 using linear interpolation.

    Parameters
    ----------
    signals : 2-D ndarray
        Signals to resample.
    new_length : positive int
        New signals shape along axis 1.

    Returns
    -------
    signals : 2-D ndarray
        Resampled signals.
    """
    arg = np.linspace(0, signals.shape[1] - 1, new_length)
    x_left = arg.astype(np.int32)  # pylint: disable=no-member
    x_right = x_left + 1
    x_right[-1] = x_left[-1]
    alpha = arg - x_left
    y_left = signals[:, x_left]
    y_right = signals[:, x_right]
    return y_left + (y_right - y_left) * alpha


def convolve_signals(signals, kernel, padding_mode="edge", axis=-1, **kwargs):
    """Convolve signals with given kernel.

    Parameters
    ----------
    signals : ndarray
        Signals to convolve.
    kernel : array_like
        Convolution kernel.
    axis : int
        Axis along which signals are sliced.
    padding_mode : str or function
        np.pad padding mode.
    kwargs : misc
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


def band_pass_signals(signals, freq, low=None, high=None, axis=-1):
    """Reject frequencies outside given range.

    Parameters
    ----------
    signals : ndarray
        Signals to filter.
    freq : positive float
        Sampling rate.
    low : positive float
        High-pass filter cutoff frequency (Hz).
    high : positive float
        Low-pass filter cutoff frequency (Hz).
    axis : int
        Axis along which signals are sliced.

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

def predict_hmm_annot(signal, cwt_scales, cwt_wavelet, model):
    """ Predict peak for the signal based on features generated by gen_hmm_features.

    Parameters
    ----------
    signal : numpy.array
        Ecg signal.
    cwt_scales : array_like
        Scales to use for Continuous Wavele Transformation.
    cwt_wavelet : object or str
        Wavelet to use in CWT.

    Returns
    -------
    prediction : numpy.array
        Annotation of the signal.
    """
    # NOTE: Currently works on first lead signal only
    sig = signal[0, :]

    cwtmatr = pywt.cwt(sig, np.array(cwt_scales), cwt_wavelet)[0]
    features = ((cwtmatr - np.mean(cwtmatr, axis=1).reshape(-1, 1))/
                np.std(cwtmatr, axis=1).reshape(-1, 1)).T

    prediction = model.predict(features).reshape((1, -1)).flatten()
    return prediction

@njit(nogil=True)
def find_intervals_borders(hmm_annotation, inter_val):
    """ Finds starts and ends of the intervals with values from inter_val.

    Parameters
    ----------
    hmm_annotation : numpy.array
        Annotation for the signal from hmm_annotation model.
    inter_val : array_like
        Values that form interval of interest.

    Returns
    -------
    starts : numpy.array
        Indices of the starts of the intervals.
    ends : numpy.array
        Indices of the ens of the intervals.
    """
    intervals = np.zeros(hmm_annotation.shape, dtype=np.int8)
    for val in inter_val:
        intervals = np.logical_or(intervals, (hmm_annotation==val).astype(np.int8)).astype(np.int8)    
    masque = np.diff(intervals)
    starts = np.where(masque == 1)[0] + 1
    ends = np.where(masque == -1)[0] + 1
    if np.any(inter_val==hmm_annotation[:1]):
        ends = ends[1:]
    if np.any(inter_val==hmm_annotation[-1:]):
        starts = starts[:-1]
    return starts, ends

@njit(nogil=True)
def find_maxes(signal, starts, ends):
    """ Find index of the maximum of the segment.

    Parameters
    ----------
    signal : numpy.array
        Ecg signal.
    starts : numpy.array
        Indices of the starts of the intervals.
    ends : numpy.array
        Indices of the ens of the intervals.
    maxes : numpy.array
        Empty array for numba acceleration.

    Returns
    -------
    maxes : numpy.array
        Indices of max values of each interval.
    """
    maxes = np.empty(starts.shape)
    for i in range(maxes.shape[0]):
        maxes[i] = starts[i] + np.argmax(signal[0][starts[i]:ends[i]])
    
    return maxes

def calc_hr(signal, hmm_annotation, fs):
    """ Calculate heart rate based on HMM prediction.

    Parameters
    ----------
    signal : numpy.array
        Ecg signal.
    hmm_annotation : numpy.array
        Annotation for the signal from hmm_annotation model.
    fs : float
        Sampling rate of the signal.

    Returns
    -------
    hr_val : float
        Heart rate in beats per minute.
    """

    starts, ends = find_intervals_borders(hmm_annotation, (1,))
    # NOTE: Currently works on first lead signal only
    maxes = find_maxes(signal, starts, ends, maxes=np.empty_like(starts))

    hr_val = (np.median(np.diff(maxes) / fs) ** -1) * 60

    return hr_val

def calc_pq(hmm_annotation, fs):
    """ Calculate PQ based on HMM prediction.

    Parameters
    ----------
    hmm_annotation : numpy.array
        Annotation for the signal from hmm_annotation model.
    fs : float
        Sampling rate of the signal.

    Returns
    -------
    pq_val : float
        Duration of PQ interval in seconds.
    """

    p_starts, _ = find_intervals_borders(hmm_annotation, (14, 15, 16))
    q_starts, _ = find_intervals_borders(hmm_annotation, (0,))
    r_starts, _ = find_intervals_borders(hmm_annotation, (1,))

    p_final = []
    q_final = []

    for i in range(len(r_starts)-1):
        low = r_starts[i]
        high = r_starts[i+1]

        p_vals = p_starts[(low < p_starts) & (p_starts < high)]
        q_vals = q_starts[(low < q_starts) & (q_starts < high)]

        if len(p_vals) < 1 or len(q_vals) < 1:
            continue
        elif len(p_vals) > 1 or len(q_vals) > 1:
            print("More than one peak!")
        else:
            p_final.append(p_vals[0])
            q_final.append(q_vals[0])

    p_final = np.array(p_final)
    q_final = np.array(q_final)

    pq_intervals = q_final-p_final

    pq_val = np.median(pq_intervals) / fs

    if (pq_val < 0) or (pq_val > 0.3):
        pq_val = "-"

    return pq_val

def calc_qt(hmm_annotation, fs):
    """ Calculate QT interval based on HMM prediction.

    Parameters
    ----------
    hmm_annotation : numpy.array
        Annotation for the signal from hmm_annotation model.
    fs : float
        Sampling rate of the signal.

    Returns
    -------
    qt_val : float
        Duration of QT interval in seconds.
    """

    _, t_ends = find_intervals_borders(hmm_annotation, (5, 6, 7, 8, 9, 10))
    q_starts, _ = find_intervals_borders(hmm_annotation, (0,))
    r_starts, _ = find_intervals_borders(hmm_annotation, (1,))

    t_final = []
    q_final = []

    for i in range(len(r_starts)-1):
        low = r_starts[i]
        high = r_starts[i+1]

        t_vals = t_ends[(low < t_ends) & (t_ends < high)]
        q_vals = q_starts[(low < q_starts) & (q_starts < high)]

        if len(t_vals) < 1 or len(q_vals) < 1:
            continue
        elif len(t_vals) > 1 or len(q_vals) > 1:
            print("More than one peak!")
        else:
            t_final.append(t_vals[0])
            q_final.append(q_vals[0])

    t_final = np.array(t_final[1:])
    q_final = np.array(q_final[:-1])

    qt_intervals = t_final-q_final

    qt_val = np.median(qt_intervals) / fs

    if (qt_val < 0) or (qt_val > 0.7):
        qt_val = "-"

    return qt_val

def calc_qrs(hmm_annotation, fs):
    """ Calculate QRS interval based on HMM prediction.

    Parameters
    ----------
    hmm_annotation : numpy.array
        Annotation for the signal from hmm_annotation model.
    fs : float
        Sampling rate of the signal.

    Returns
    -------
    qrs_val : float
        Duration of QRS interval in seconds.
    """

    _, s_ends = find_intervals_borders(hmm_annotation, (2,))
    q_starts, _ = find_intervals_borders(hmm_annotation, (0,))
    r_starts, _ = find_intervals_borders(hmm_annotation, (1,))

    s_final = []
    q_final = []

    for i in range(len(r_starts)-1):
        low = r_starts[i]
        high = r_starts[i+1]

        s_vals = s_ends[(low < s_ends) & (s_ends < high)]
        q_vals = q_starts[(low < q_starts) & (q_starts < high)]

        if len(s_vals) < 1 or len(q_vals) < 1:
            continue
        elif len(s_vals) > 1 or len(q_vals) > 1:
            print("More than one peak!")
        else:
            s_final.append(s_vals[0])
            q_final.append(q_vals[0])

    s_final = np.array(s_final[1:])
    q_final = np.array(q_final[:-1])

    qs_intervals = s_final-q_final

    qrs_val = np.median(qs_intervals) / fs

    if (qrs_val < 0) or (qrs_val > 0.25):
        qrs_val = "-"

    return qrs_val
