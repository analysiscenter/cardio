""" contain methods for signal processing """

import warnings
import numpy as np

from scipy.signal import butter, lfilter, resample_poly, gaussian
from scipy.stats import bayes_mvs
from hmmlearn import hmm


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


def hmm_estimate(signal, fs, kernel=None, n_iter=25, n_components=3):
    """
    Find n_components in signal
    """
    warnings.filterwarnings("ignore")
    if kernel is None:
        kernel = gaussian(int(100 * fs / 300), int(10 * fs / 300))

    grad1 = np.gradient(signal)
    grad1_sm = np.convolve(grad1**2, kernel, mode='same')

    grad2 = np.gradient(np.gradient(signal))
    grad2_sm = np.convolve(grad2**2, kernel, mode='same')

    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full",
                                n_iter=n_iter)
    train = list(zip(signal, grad1, grad1_sm, grad2, grad2_sm))
    model.fit(train)

    pred = model.predict(train)
    return pred


def get_r_peaks(signal, pred, fs, kernel=None):
    """
    Find component within n_components that corresponds to r_peaks
    """
    if kernel is None:
        kernel = gaussian(int(100 * fs / 300), int(10 * fs / 300))

    grad2 = np.gradient(np.gradient(signal))**2
    grad2 = np.convolve(grad2, kernel, mode='same')

    probe_class = []
    for i in range(max(pred)+1):
        probe_class.append(np.correlate((pred == i).astype(int), grad2)[0])
    rpeak_class = np.array(probe_class).argsort()[-1]
    rpeak = (pred == rpeak_class)
    return rpeak.astype(int)


def stack_segments(signal, start, stop, rate=None):
    """
    Divide signal into set of segments
    """
    stacked = []
    if rate is None:
        rate = 100
    p_start = np.arange(len(signal))[start.astype(bool)]
    p_stop = np.arange(len(signal))[stop.astype(bool)]
    for a, b in zip(p_start, p_stop):
        segment = signal[a: b]
        stacked.append(resample_poly(segment, rate, len(segment)))
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
        return None

    sig_segments = [signal[p_start[i]: p_stop[i]] for i in range(len(p_start))]

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
