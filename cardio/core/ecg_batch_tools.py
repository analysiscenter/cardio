"""Сontains ECG processing tools."""

import os
import struct

import numpy as np
from numba import njit
from scipy.io import wavfile
import dicom
import pyedflib
import wfdb

# Constants

# This is the predefined keys of the meta component.
# Each key is initialized with None.
META_KEYS = [
    "age",
    "sex",
    "timestamp",
    "comments",
    "fs",
    "signame",
    "units",
]

# This is the mapping from inner HMM states to human-understandable
# cardiological terms.
P_STATES = np.array([14, 15, 16], np.int64)
T_STATES = np.array([5, 6, 7, 8, 9, 10], np.int64)
QRS_STATES = np.array([0, 1, 2], np.int64)
Q_STATE = np.array([0], np.int64)
R_STATE = np.array([1], np.int64)
S_STATE = np.array([2], np.int64)


def check_signames(signame, nsig):
    """Check that signame is in proper format.

    Check if signame is a list of values that can be casted
    to string, othervise generate new signame list with numbers
    0 to `nsig`-1 as strings.

    Parameters
    ----------
    signame : misc
        Signal names from file.
    nsig : int
        Number of signals / channels.

    Returns
    -------
    signame : list
        List of string names of signals / channels.
    """
    if isinstance(signame, (tuple, list)) and len(signame) == nsig:
        signame = [str(name) for name in signame]
    else:
        signame = [str(number) for number in range(nsig)]
    return np.array(signame)


def check_units(units, nsig):
    """Check that units are in proper format.

    Check if units is a list of values with lenght
    equal to number of channels.

    Parameters
    ----------
    units : misc
        Units from file.
    nsig : int
        Number of signals / channels.

    Returns
    -------
    units : list
        List of units of signal / channel.
    """
    if not (isinstance(units, (tuple, list)) and len(units) == nsig):
        units = [None for number in range(nsig)]
    return np.array(units)


def load_wfdb(path, components, *args, **kwargs):
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
    _ = args

    ann_ext = kwargs.get("ann_ext")

    path = os.path.splitext(path)[0]
    record = wfdb.rdsamp(path)
    signal = record.__dict__.pop("p_signals").T
    record_meta = record.__dict__
    nsig = record_meta["nsig"]

    if "annotation" in components and ann_ext is not None:
        annotation = wfdb.rdann(path, ann_ext)
        annot = {"annsamp": annotation.sample,
                 "anntype": annotation.symbol}
    else:
        annot = {}

    # Initialize meta with defined keys, load values from record
    # meta and preprocess to our format.
    meta = dict(zip(META_KEYS, [None] * len(META_KEYS)))
    meta.update(record_meta)

    meta["signame"] = check_signames(meta["signame"], nsig)
    meta["units"] = check_units(meta["units"], nsig)

    data = {"signal": signal,
            "annotation": annot,
            "meta": meta}
    return [data[comp] for comp in components]


def load_dicom(path, components, *args, **kwargs):
    """
    Load given components from DICOM file.

    Parameters
    ----------
    path : str
        Path to .hea file.
    components : iterable
        Components to load.

    Returns
    -------
    ecg_data : list
        List of ecg data components.
    """

    def signal_decoder(record, nsig):
        """
        Helper function to decode signal from binaries when reading from dicom.
        """
        definition = record.WaveformSequence[0].ChannelDefinitionSequence
        data = record.WaveformSequence[0].WaveformData

        unpack_fmt = "<{}h".format(int(len(data) / 2))
        factor = np.ones(nsig)
        baseline = np.zeros(nsig)

        for i in range(nsig):

            assert definition[i].WaveformBitsStored == 16

            channel_sens = definition[i].get("ChannelSensitivity")
            channel_sens_cf = definition[i].get("ChannelSensitivityCorrectionFactor")
            if channel_sens is not None and channel_sens_cf is not None:
                factor[i] = float(channel_sens) * float(channel_sens_cf)

            channel_bl = definition[i].get("ChannelBaseline")
            if channel_bl is not None:
                baseline[i] = float(channel_bl)

        unpacked_data = struct.unpack(unpack_fmt, data)

        signals = np.asarray(unpacked_data, dtype=np.float32).reshape(-1, nsig)
        signals = ((signals + baseline) * factor).T

        return signals

    _ = args, kwargs

    record = dicom.read_file(path)

    sequence = record.WaveformSequence[0]

    assert sequence.WaveformSampleInterpretation == 'SS'
    assert sequence.WaveformBitsAllocated == 16

    nsig = sequence.NumberOfWaveformChannels

    annot = {}

    meta = dict(zip(META_KEYS, [None] * len(META_KEYS)))

    if record.PatientAge[-1] == "Y":
        age = np.int(record.PatientAge[:-1])
    else:
        age = np.int(record.PatientAge[:-1]) / 12.0

    meta["age"] = age
    meta["sex"] = record.PatientSex
    meta["timestamp"] = record.AcquisitionDateTime
    meta["comments"] = [section.UnformattedTextValue for section in
                        record.WaveformAnnotationSequence if section.AnnotationGroupNumber == 0]
    meta["fs"] = sequence.SamplingFrequency
    meta["signame"] = [section.ChannelSourceSequence[0].CodeMeaning for section in
                       sequence.ChannelDefinitionSequence]
    meta["units"] = [section.ChannelSensitivityUnitsSequence[0].CodeValue for section in
                     sequence.ChannelDefinitionSequence]

    meta["signame"] = check_signames(meta["signame"], nsig)
    meta["units"] = check_units(meta["units"], nsig)

    signal = signal_decoder(record, nsig)

    data = {"signal": signal,
            "annotation": annot,
            "meta": meta}
    return [data[comp] for comp in components]


def load_edf(path, components, *args, **kwargs):
    """
    Load given components from EDF file.

    Parameters
    ----------
    path : str
        Path to .hea file.
    components : iterable
        Components to load.

    Returns
    -------
    ecg_data : list
        List of ecg data components.
    """
    _ = args, kwargs

    record = pyedflib.EdfReader(path)

    annot = {}
    meta = dict(zip(META_KEYS, [None] * len(META_KEYS)))

    meta["sex"] = record.getGender() if record.getGender() != '' else None
    meta["timestamp"] = record.getStartdatetime().strftime("%Y%m%d%H%M%S")
    nsig = record.signals_in_file

    if len(np.unique(record.getNSamples())) != 1:
        raise ValueError("Different signal lenghts are not supported!")

    if len(np.unique(record.getSampleFrequencies())) == 1:
        meta["fs"] = record.getSampleFrequencies()[0]
    else:
        raise ValueError("Different sampling rates are not supported!")

    meta["signame"] = record.getSignalLabels()
    meta["units"] = [record.getSignalHeader(sig)["dimension"] for sig in range(nsig)]

    meta.update(record.getHeader())

    meta["signame"] = check_signames(meta["signame"], nsig)
    meta["units"] = check_units(meta["units"], nsig)

    signal = np.array([record.readSignal(i) for i in range(nsig)])

    data = {"signal": signal,
            "annotation": annot,
            "meta": meta}
    return [data[comp] for comp in components]


def load_wav(path, components, *args, **kwargs):
    """
    Load given components from wav file.

    Parameters
    ----------
    path : str
        Path to .hea file.
    components : iterable
        Components to load.

    Returns
    -------
    ecg_data : list
        List of ecg data components.
    """
    _ = args, kwargs

    fs, signal = wavfile.read(path)
    if signal.ndim == 1:
        nsig = 1
        signal = signal.reshape([-1, 1])
    elif signal.ndim == 2:
        nsig = signal.shape[1]
    else:
        raise ValueError("Unexpected number of dimensions in signal array: {}".format(signal.ndim))

    signal = signal.T

    annot = {}
    meta = dict(zip(META_KEYS, [None] * len(META_KEYS)))

    meta["fs"] = fs
    meta["signame"] = check_signames(meta["signame"], nsig)
    meta["units"] = check_units(meta["units"], nsig)

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
