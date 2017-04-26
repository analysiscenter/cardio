# pylint: disable=invalid-name
""" contain methods for signal processing """

import warnings
import numpy as np

from scipy.signal import butter, lfilter, resample_poly, gaussian, correlate
from scipy.stats import bayes_mvs
from hmmlearn import hmm
import pywt


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Find frequence interval
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """
    Apply butter_bandpass_filter
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, signal)


def hmm_estimate(signal, n_components=3, n_iter=25):
    """
    Find n_components in signal
    """
    warnings.filterwarnings("ignore")
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full",
                            n_iter=n_iter)
    model.fit(signal)
    pred = model.predict(signal)
    return pred
'''    
def convolve(signal, kernel, axis=-1):
    return np.apply_along_axis(np.convolve, axis, signal**2, v=kernel, mode='same')'''


def get_r_peaks(pred, target):
    """
    Find component within n_components that corresponds to r_peaks
    """
    probe_class = []
    for i in range(max(pred)+1):
        probe_class.append(np.correlate((pred == i).astype(int), target)[0])
    rpeak_class = np.array(probe_class).argsort()[-1]
    rpeak = (pred == rpeak_class)
    return rpeak.astype(int)


def stack_segments(signal, start, stop, unify=True, rate=None):
    """
    Divide signal into set of segments
    """
    stacked = []
    if rate is None:
        rate = 100
    p_start = np.arange(len(signal))[start.astype(bool)]
    p_stop = np.arange(len(signal))[stop.astype(bool)]
    if len(p_start) == 0:
        return np.array([[np.nan] * rate])
    
    for a, b in zip(p_start, p_stop):
        segment = signal[a: b + 1]
        if unify:
            stacked.append(resample_poly(segment, rate, len(segment)))
        else:
            stacked.append(segment)
    return np.array(stacked)


def segment_profile(signal, start, stop, rate=None, return_ci=True):
    """
    Mean and ci for singal segments
    """
    if rate is None:
        rate = 100
    stacked = stack_segments(signal, start, stop, rate)

    if return_ci:
        if len(stacked) < 2:
            print('segment_profile needs at least 2 segments')
            return None
        stats = [bayes_mvs(x)[0] for x in stacked.T]
        mean = np.array([x[0] for x in stats])
        interval = np.array([x[1] for x in stats])
        return mean, interval
    else:
        return stacked.mean(axis=0)


def segment_features(signal, start, stop):
    """
    Statistics of signal within sogments
    """
    p_start = np.arange(len(signal))[start.astype(bool)]
    p_stop = np.arange(len(signal))[stop.astype(bool)] 

    if len(p_start) == 0:
        return np.array([]).reshape((5,0))

    sig_segments = [signal[p_start[i]: p_stop[i] + 1] for i in range(len(p_start))]

    length = (p_stop - p_start).astype(int) 
    ampl = np.array([np.abs(x).max() for x in sig_segments])
    area = np.array([np.abs(x).sum() for x in sig_segments])
    mean = np.array([x.mean() for x in sig_segments])
    std = np.array([x.std() for x in sig_segments])

    return length, ampl, area, mean, std


def show_segment_profile(mean, interval, ax, rate=None):
    """
    Plot mean signal
    """
    if rate is None:
        rate = 1
    ax.fill_between(np.linspace(0, rate, len(mean)), interval[:, 0],
                    interval[:, 1])
    ax.plot(np.linspace(0, rate, len(mean)), mean, c='r')


def show_hist(start, stop, ax, bins=None):
    """
    Histogram of segment lengh
    """
    if bins is None:
        bins = np.linspace(0, 400, 20)
    p_start = np.arange(len(start))[start.astype(bool)]
    p_stop = np.arange(len(stop))[stop.astype(bool)]
    data = p_stop - p_start
    hist, edg = np.histogram(data, bins=bins, normed=True)
    width = 0.9 * (edg[1] - edg[0])
    center = (edg[:-1] + edg[1:]) / 2
    ax.bar(center, hist, align='center', width=width)


def segment_alignment(signal, max_iter=10, align_to='median'):
    """
    To be added
    """
    n_iter = 0
    shifted = signal.copy()
    L = shifted.shape[-1]
    while n_iter < max_iter:
        if align_to == 'median':
            mean = np.median(shifted, axis=0)
        elif align_to == 'mean':
            mean = np.mean(shifted, axis=0)
        else:
            raise KeyError('align_to supports mean or median only')
        delay = []
        for sig in shifted:
            d = np.argmax(correlate(mean, sig, 'full'))
            delay.append(L - d)
        delay = np.array(delay)
        if np.abs(delay).max() <= 1:
            return shifted

        for i in range(len(delay)):
            if delay[i] >= 0:
                shifted[i] = np.pad(shifted[i, delay[i]:], (0, delay[i]),
                                    'constant', constant_values=0)
            else:
                shifted[i] = np.pad(shifted[i, :delay[i]], (-delay[i], 0),
                                    'constant', constant_values=0)
        n_iter += 1
    return shifted


def segment_centering(start, stop, target_length):
    """
    Wavelet spectrogram of signal along given axis, default is -1
    """
    p_start = np.arange(len(start))[start.astype(bool)]
    p_stop = np.arange(len(stop))[stop.astype(bool)]
    length = p_stop - p_start + 1
    out = target_length - length

    np_start = p_start - np.floor(out / 2).astype(int)
    np_stop = p_stop + np.ceil(out / 2).astype(int)
    valid_ind = (np_start >= 0) & (np_stop < len(stop))
    np_start = np_start[valid_ind]
    np_stop = np_stop[valid_ind]

    res_start = np.zeros_like(start, dtype=int)
    res_start[np_start] = 1
    res_stop = np.zeros_like(stop, dtype=int)
    res_stop[np_stop] = 1
    return np.array([res_start, res_stop])

def wavelet_spectrogram(signal, wavelet, axis=None):
    """
    Wavelet spectrogram of signal along given axis, default is -1
    """
    if axis is None:
        axis = -1
    new_len = 2**int(np.log2(signal.shape[axis]))
    w_coef = pywt.wavedec(resample_poly(signal, new_len, signal.shape[axis], axis=axis),
                          wavelet=wavelet, mode='per', axis=axis)
    res_t = []
    for i in range(1, len(w_coef)):
        res_t.append(np.repeat(w_coef[-i][0], 2**i))
        res_t.append(np.repeat(w_coef[0][0], 2**(len(w_coef) - 1)))

    time = np.linspace(0, signal.shape[axis], new_len)
    scale = np.arange(1, len(res_t) + 1)
    time_ax, scale_ax = np.meshgrid(time, scale)
    power = np.stack(res_t)
    return time_ax, scale_ax, power

def fft_spectrogram(signal, window, overlay=0, axis=None):
    """
    Windowed FFT along given axis, default is -1. Returns spectrogram as [time, scale, power].
    """
    if axis is None:
        axis = -1
    if (overlay < 0) or (overlay > 1):
        raise ValueError("Overlay should be in [0, 1]", overlay)

    start = 0
    end = start + window
    res_t = []
    while end < signal.shape[axis]:
        segment = signal[axis, start: end]
        coef = np.fft.rfft(segment)
        res_t.append(coef)
        start += int(window * (1 - overlay))
        end = start + window
    time = np.linspace(0, signal.shape[axis], len(res_t))
    scale = np.arange(len(res_t[0]))
    time_ax, scale_ax = np.meshgrid(time, scale)
    power = np.stack(res_t).transpose()
    return time_ax, scale_ax, power

def loc_segments(seq, segment_type):
    """
    Returns start and stop positons of requested segment_type
    seq is a sequence of 0 and 1
    peak is a continious segment of 1
    period is a segment within beginnings of two successive peaks
    """
    available_segment_types = {'peak', 'period'}
    if segment_type not in available_segment_types:
        raise KeyError('Unknown segmetnl type {0}.' \
                       'Available types are'.format(segment_type),
                       available_segment_types)
    start_ind = np.where(np.diff(seq) == 1)[0] + 1
    if segment_type == 'peak':
        stop_ind = np.where(np.diff(seq) == -1)[0]
    else:
        stop_ind = start_ind - 1
    if (len(start_ind) == 0) or (len(stop_ind) == 0):
        start_ind = []
        stop_ind = []
    elif (len(start_ind) == 1) and (len(stop_ind) == 1):
        if start_ind[0] > stop_ind[0]:
            start_ind = []
            stop_ind = []
    else:
        if start_ind[0] > stop_ind[0]:
            stop_ind = stop_ind[1:]
        if stop_ind[-1] < start_ind[-1]:
            start_ind = start_ind[:-1]
    start = np.zeros_like(seq, dtype=int)
    start[start_ind] = 1
    stop = np.zeros_like(seq, dtype=int)
    stop[stop_ind] = 1
    return start, stop
