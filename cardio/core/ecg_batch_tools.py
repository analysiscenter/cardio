"""Сontains ECG processing tools."""

import os
import numpy as np

import pywt
from numba import njit

import wfdb

# Constants

# This is the predefined keys of the meta component. 
# Each key is initialized with None.
META_KEYS = [
"age",
"sex",
"timestamp",
"comments",
"heart_rate",
"RR_interval",
"PQ_interval",
"QT_interval",
"P_duration",
"T_durarion",
"QRS_duration",
"heart_axis",
"nsig",
"siglen",
"fs",
"filter_low_freq", #* num_ch
"filter_high_freq", #* num_ch
"signame", #* num_ch
"units", #* num_ch
"units_factor", #* num_ch
"filename",
]

# This is the mapping from inner HMM states to human-understandable
# cardiological terms.
P_STATES = np.array([14, 15, 16], np.int64)
T_STATES = np.array([5, 6, 7, 8, 9, 10], np.int64)
QRS_STATES = np.array([0, 1, 2], np.int64)
Q_STATE = np.array([0], np.int64)
R_STATE = np.array([1], np.int64)
S_STATE = np.array([2], np.int64)


def load_wfdb(path, components, ann_ext=None):
    """Load given components from wfdb file.

    Parameters
    ----------
    path : str
        Path to .hea file.
    components : iterable
        Components to load.
    ann_ext: str
        Extension of the annotation file.

    Returns
    -------
    ecg_data : list
        List of ecg data components.
    """
    path = os.path.splitext(path)[0]
    record = wfdb.rdsamp(path)
    signal = record.__dict__.pop("p_signals").T
    record_meta = record.__dict__
    if "annotation" in components and ann_ext is not None:
        annotation = wfdb.rdann(path, ann_ext)
        annot = {"annsamp": annotation.annsamp,
                 "anntype": annotation.anntype}
    else:
        annot = {}

    # Initialize meta with defined keys, load values from record
    # meta and preprocess to our format.
    meta = dict(zip(META_KEYS, [None] * len(META_KEYS)))
    meta.update(record_meta)
    meta["filename"] = meta["filename"][0]

    data = {"signal": signal,
            "annotation": annot,
            "meta": meta}
    
    return [data[comp] for comp in components]


@njit(nogil=True)
def split_signals(signals, length, step):
    """Split signals along axis 1 with given ``length`` and ``step``.

    Parameters
    ----------
    signals : 2-D ndarray
        Signals to split.
    length : positive int
        Length of each segment along axis 1.
    step : positive int
        Segmentation step.

    Returns
    -------
    signals : 3-D ndarray
        Split signals stacked along new axis with index 0.
    """
    res = np.empty(((signals.shape[1] - length) // step + 1, signals.shape[0], length), dtype=signals.dtype)
    for i in range(res.shape[0]):
        res[i, :, :] = signals[:, i * step : i * step + length]
    return res


@njit(nogil=True)
def random_split_signals(signals, length, n_segments):
    """Split signals along axis 1 ``n_segments`` times with random start
    position and given ``length``.

    Parameters
    ----------
    signals : 2-D ndarray
        Signals to split.
    length : positive int
        Length of each segment along axis 1.
    n_segments : positive int
        Number of segments.

    Returns
    -------
    signals : 3-D ndarray
        Split signals stacked along new axis with index 0.
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
    """Convolve signals with given ``kernel``.

    Parameters
    ----------
    signals : ndarray
        Signals to convolve.
    kernel : array_like
        Convolution kernel.
    padding_mode : str or function
        ``np.pad`` padding mode.
    axis : int
        Axis along which signals are sliced.
    kwargs : misc
        Any additional named argments to ``np.pad``.

    Returns
    -------
    signals : ndarray
        Convolved signals.

    Raises
    ------
    ValueError
        If ``kernel`` is not one-dimensional or has non-numeric ``dtype``.
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

    Raises
    ------
    ValueError
        If ``freq`` is negative or non-numeric.
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


def wavelet_transform(signal, cwt_scales, cwt_wavelet):
    """Generate wavelet transformation from the signal.

    Parameters
    ----------
    signal : numpy.array
        Ecg signal.
    cwt_scales : array_like
        Scales to use for Continuous Wavele Transformation.
    cwt_wavelet : Wavelet object or name
        Wavelet to use in CWT.

    Returns
    -------
    features : numpy.array
        Features generated by wavelet from the signal.
    """
    # NOTE: Currently works on first lead signal only
    sig = signal[0, :]

    cwtmatr = pywt.cwt(sig, np.array(cwt_scales), cwt_wavelet)[0]
    wavelets = ((cwtmatr - np.mean(cwtmatr, axis=1).reshape(-1, 1)) /
                np.std(cwtmatr, axis=1).reshape(-1, 1)).T

    return wavelets


@njit(nogil=True)
def find_intervals_borders(hmm_annotation, inter_val):
    """Find starts and ends of the intervals.

    This function finds starts and ends of continuous intervals of values
    from inter_val in hmm_annotation.

    Parameters
    ----------
    hmm_annotation : numpy.array
        Annotation for the signal from hmm_annotation model.
    inter_val : array_like
        Values that form interval of interest.

    Returns
    -------
    starts : 1-D ndarray
        Indices of the starts of the intervals.
    ends : 1-D ndarray
        Indices of the ends of the intervals.
    """
    intervals = np.zeros(hmm_annotation.shape, dtype=np.int8)
    for val in inter_val:
        intervals = np.logical_or(intervals, (hmm_annotation == val).astype(np.int8)).astype(np.int8)
    masque = np.diff(intervals)
    starts = np.where(masque == 1)[0] + 1
    ends = np.where(masque == -1)[0] + 1
    if np.any(inter_val == hmm_annotation[:1]):
        ends = ends[1:]
    if np.any(inter_val == hmm_annotation[-1:]):
        starts = starts[:-1]
    return starts, ends


@njit(nogil=True)
def find_maxes(signal, starts, ends):
    """ Find index of the maximum of the segment.

    Parameters
    ----------
    signal : 2-D ndarray
        ECG signal.
    starts : 1-D ndarray
        Indices of the starts of the intervals.
    ends : 1-D ndarray
        Indices of the ens of the intervals.

    Returns
    -------
    maxes : 1-D ndarray
        Indices of max values of each interval.

    Notes
    -----
    Currently works with first lead only.
    """

    maxes = np.empty(starts.shape, dtype=np.float64)
    for i in range(maxes.shape[0]):
        maxes[i] = starts[i] + np.argmax(signal[0][starts[i]:ends[i]])

    return maxes


@njit(nogil=True)
def calc_hr(signal, hmm_annotation, fs, r_state=R_STATE):
    """ Calculate heart rate based on HMM prediction.

    Parameters
    ----------
    signal : 2-D ndarray
        ECG signal.
    hmm_annotation : 1-D ndarray
        Annotation for the signal from hmm_annotation model.
    fs : float
        Sampling rate of the signal.
    r_state : 1-D ndarray
        Array with values that represent R peak.
        Default value is R_STATE, which is a constant of this module.

    Returns
    -------
    hr_val : float
        Heart rate in beats per minute.
    """

    starts, ends = find_intervals_borders(hmm_annotation, r_state)
    # NOTE: Currently works on first lead signal only
    maxes = find_maxes(signal, starts, ends)
    diff = maxes[1:] - maxes[:-1]
    hr_val = (np.median(diff / fs) ** -1) * 60

    return hr_val


@njit(nogil=True)
def calc_pq(hmm_annotation, fs, p_states=P_STATES, q_state=Q_STATE, r_state=R_STATE):
    """ Calculate PQ based on HMM prediction.

    Parameters
    ----------
    hmm_annotation : numpy.array
        Annotation for the signal from hmm_annotation model.
    fs : float
        Sampling rate of the signal.
    p_states : 1-D ndarray
        Array with values that represent P peak.
        Default value is P_STATES, which is a constant of this module.
    q_state : 1-D ndarray
        Array with values that represent Q peak.
        Default value is Q_STATE, which is a constant of this module.
    r_state : 1-D ndarray
        Array with values that represent R peak.
        Default value is R_STATE, which is a constant of this module.

    Returns
    -------
    pq_val : float
        Duration of PQ interval in seconds.
    """

    p_starts, _ = find_intervals_borders(hmm_annotation, p_states)
    q_starts, _ = find_intervals_borders(hmm_annotation, q_state)
    r_starts, _ = find_intervals_borders(hmm_annotation, r_state)

    p_final = - np.ones(r_starts.shape[0] - 1)
    q_final = - np.ones(r_starts.shape[0] - 1)

    maxlen = hmm_annotation.shape[0]

    if not p_starts.shape[0] * q_starts.shape[0] * r_starts.shape[0]:
        return 0.00

    temp_p = np.zeros(maxlen)
    temp_p[p_starts] = 1
    temp_q = np.zeros(maxlen)
    temp_q[q_starts] = 1

    for i in range(len(r_starts) - 1):
        low = r_starts[i]
        high = r_starts[i + 1]

        inds_p = np.where(temp_p[low:high])[0] + low
        inds_q = np.where(temp_q[low:high])[0] + low

        if inds_p.shape[0] == 1 and inds_q.shape[0] == 1:
            p_final[i] = inds_p[0]
            q_final[i] = inds_q[0]

    p_final = p_final[p_final > -1]
    q_final = q_final[q_final > -1]

    intervals = q_final - p_final

    return np.median(intervals) / fs


@njit(nogil=True)
def calc_qt(hmm_annotation, fs, t_states=T_STATES, q_state=Q_STATE, r_state=R_STATE):
    """ Calculate QT interval based on HMM prediction.

    Parameters
    ----------
    hmm_annotation : numpy.array
        Annotation for the signal from hmm_annotation model.
    fs : float
        Sampling rate of the signal.
    t_states : 1-D ndarray
        Array with values that represent T peak.
        Default value is T_STATES, which is a constant of this module.
    q_state : 1-D ndarray
        Array with values that represent Q peak.
        Default value is Q_STATE, which is a constant of this module.
    r_state : 1-D ndarray
        Array with values that represent R peak.
        Default value is R_STATE, which is a constant of this module.

    Returns
    -------
    qt_val : float
        Duration of QT interval in seconds.
    """

    _, t_ends = find_intervals_borders(hmm_annotation, t_states)
    q_starts, _ = find_intervals_borders(hmm_annotation, q_state)
    r_starts, _ = find_intervals_borders(hmm_annotation, r_state)

    t_final = - np.ones(r_starts.shape[0] - 1)
    q_final = - np.ones(r_starts.shape[0] - 1)

    maxlen = hmm_annotation.shape[0]

    if not t_ends.shape[0] * q_starts.shape[0] * r_starts.shape[0]:
        return 0.00

    temp_t = np.zeros(maxlen)
    temp_t[t_ends] = 1
    temp_q = np.zeros(maxlen)
    temp_q[q_starts] = 1

    for i in range(len(r_starts) - 1):
        low = r_starts[i]
        high = r_starts[i + 1]

        inds_t = np.where(temp_t[low:high])[0] + low
        inds_q = np.where(temp_q[low:high])[0] + low

        if inds_t.shape[0] == 1 and inds_q.shape[0] == 1:
            t_final[i] = inds_t[0]
            q_final[i] = inds_q[0]

    t_final = t_final[t_final > -1][1:]
    q_final = q_final[q_final > -1][:-1]

    intervals = t_final - q_final

    return np.median(intervals) / fs


@njit(nogil=True)
def calc_qrs(hmm_annotation, fs, s_state=S_STATE, q_state=Q_STATE, r_state=R_STATE):
    """ Calculate QRS interval based on HMM prediction.

    Parameters
    ----------
    hmm_annotation : numpy.array
        Annotation for the signal from hmm_annotation model.
    fs : float
        Sampling rate of the signal.
    s_state : 1-D ndarray
        Array with values that represent S peak.
        Default value is S_STATE, which is a constant of this module.
    q_state : 1-D ndarray
        Array with values that represent Q peak.
        Default value is Q_STATE, which is a constant of this module.
    r_state : 1-D ndarray
        Array with values that represent R peak.
        Default value is R_STATE, which is a constant of this module.

    Returns
    -------
    qrs_val : float
        Duration of QRS complex in seconds.
    """
    _, s_ends = find_intervals_borders(hmm_annotation, s_state)
    q_starts, _ = find_intervals_borders(hmm_annotation, q_state)
    r_starts, _ = find_intervals_borders(hmm_annotation, r_state)

    s_final = - np.ones(r_starts.shape[0] - 1)
    q_final = - np.ones(r_starts.shape[0] - 1)

    maxlen = hmm_annotation.shape[0]

    if not s_ends.shape[0] * q_starts.shape[0] * r_starts.shape[0]:
        return 0.00

    temp_s = np.zeros(maxlen)
    temp_s[s_ends] = 1
    temp_q = np.zeros(maxlen)
    temp_q[q_starts] = 1

    for i in range(len(r_starts) - 1):
        low = r_starts[i]
        high = r_starts[i + 1]

        inds_s = np.where(temp_s[low:high])[0] + low
        inds_q = np.where(temp_q[low:high])[0] + low

        if inds_s.shape[0] == 1 and inds_q.shape[0] == 1:
            s_final[i] = inds_s[0]
            q_final[i] = inds_q[0]

    s_final = s_final[s_final > -1][1:]
    q_final = q_final[q_final > -1][:-1]

    intervals = s_final - q_final

    return np.median(intervals) / fs
