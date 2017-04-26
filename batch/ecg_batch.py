# pylint: disable=invalid-name
""" contain Batch class for processing ECGs """

import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import pywt

from scipy.signal import resample_poly, gaussian
from scipy.stats import threshold
from sklearn.decomposition import PCA
from collections import ChainMap
from dataset import *
from functools import partial

sys.path.append('..')

from signal_process import signal_process as sps


def get_ecg(i, fields):
    data, annot, meta = fields
    pos = meta[i]['__pos']
    return (data[pos],
           {k: v[pos] for k, v in annot.items()},
           meta[i])

def hmm_estimate_compile(i, fields, kernel, n_iter):
    signal, annot, meta = get_ecg(i, fields)
    pred = sps.hmm_estimate(signal[0], meta['fs'], kernel, n_iter)
    r_peak = sps.get_r_peaks(signal[0], pred, meta['fs'], kernel).reshape((1, -1))
    annot['R_peaks'] = r_peak
    return [signal, annot, {i: meta}]

def resample_compile(i, fields, new_fs):
    """
    Resample signals to new fixed rate given by new_fs
    """
    signal, annot, meta = get_ecg(i, fields)
    fs = meta['fs']
    new_len = int(new_fs * len(signal[0]) / fs)
    signal = resample_poly(signal, new_len, len(signal[0]), axis=1)
    meta['fs'] = new_fs
    return [signal, annot, {i: meta}]
        

class EcgBatch(Batch):
    """
    Batch of ECG data
    """
    def __init__(self, index, *args, **kwargs):
        super().__init__(index)

        self._data = None
        self._annotation = self.create_annotation_df()
        self._meta = dict()
        self.history = []

    @staticmethod
    def create_annotation_df(data=None):
        """ Create a pandas dataframe with ECG annotations """
        return {}

    @action
    def load(self, src=None, fmt="wfdb"):
        #print('Load', self.indices)
        """
        Loads data from different sources
        src is not used yet, so files locations are defined by the index
        """
        if fmt == "wfdb":
            list_of_arrs, list_of_annotations, meta = self._load_wfdb(src)
        elif fmt == "npz":
            list_of_arrs, list_of_annotations, meta = self._load_npz(src)
        else:
            raise TypeError("Incorrect type of source")

        # ATTENTION!
        # Construction below is used to overcome numpy bug:
        # adding empty array to list of arrays, then generating array
        # of arrays and removing the last item (empty array)
        list_of_arrs.append(np.array([]))
        self._data = np.array(list_of_arrs)[:-1]
        # ATTENTION!
        # Annotation should be loaded with a separate function
        self._annotation = list_of_annotations
        self._meta = meta

        # add info in self.history
        info = dict()
        info['method'] = 'load'
        info['params'] = {}
        self.history.append(info)

        return self

    def _load_wfdb(self, src):
        """
        Load signal and meta, loading of annotation should be added
        """
        list_of_arrs = []
        list_of_annotations = {}
        meta = {}
        for pos, ecg in np.ndenumerate(self.indices):
            if src is None:
                path = self.index.get_fullpath(ecg)
            else:
                path = src[ecg]
            record = wfdb.rdsamp(os.path.splitext(path)[0])
            signal = record.__dict__.pop('p_signals')
            fields = record.__dict__
            signal = signal.T
            # try:
            #     annot = wfdb.rdann(path, "atr")
            # except FileNotFoundError:
            #     annot = {}
            list_of_arrs.append(signal)
            # list_of_annotations.append(annot)
            fields.update({"__pos": pos[0]})
            meta.update({ecg: fields})

        return list_of_arrs, list_of_annotations, meta

    def _load_npz(self, src):
        """
        Load signal and meta, loading of annotation should be added
        """
        list_of_arrs = []
        list_of_annotations = []
        meta = {}
        for pos, ecg in np.ndenumerate(self.indices):
            if src is None:
                path = self.index.get_fullpath(ecg)
            else:
                path = src + '/' + ecg + '.npz'
            data = np.load(path)
            list_of_arrs.append(data["signal"])
            list_of_annotations.append(data["annotation"].tolist())
            fields = data["meta"].tolist()
            fields.update({"__pos": pos[0]})
            meta.update({ecg: fields})

        keys = list(list_of_annotations[0].keys())
        annot = {k: [] for k in keys}
        for x in list_of_annotations:
            for k in keys:
                annot[k].append(x[k])
        for k in keys:
            annot[k].append(np.array([]))
            annot[k] = np.array(annot[k])[:-1]

        return list_of_arrs, annot, meta

    @action
    def dump(self, dst, fmt="npz"):
        """
        Save each ecg in its own file named as '<index>.<fmt>'
        """
        if fmt == "npz":
            for ecg in self.indices:
                signal, ann, meta = self[ecg]
                saved_meta = meta.copy()
                del saved_meta['__pos']
                np.savez(os.path.join(dst, ecg + "." + fmt),
                         signal=signal,
                         annotation=ann,
                         meta=saved_meta)
        elif fmt == 'hmm':
            df = pd.DataFrame([self.indices, self._annotation['R_peaks']]).T
            df = df.rename(columns={0: "ecg", 1: 'R_peaks'})
            df.to_csv(os.path.join(dst, "hmm_dump.csv"))
        else:
            raise NotImplementedError("The format is not supported yet")

    def __getitem__(self, index):
        if index in self.indices:
            pos = self._meta[index]['__pos']
            return (self._data[pos],
                    {k: v[pos] for k, v in self._annotation.items()},
                    self._meta[index])
        else:
            raise IndexError("There is no such index in the batch", index)
    
    def update(self, data=None, annot=None, meta=None):
        """
        Update content of ecg_batch
        """
        if data is not None:
            self._data = np.array(data)
        if annot is not None:
            self._annotation = annot
        if meta is not None:
            self._meta = meta
        return self

    def init_parallel(self, *args, **kwargs):
        return list(map(list, zip(self.indices,
                                  np.tile([self._data,
                                           self._annotation,
                                           self._meta], len(self.indices)).reshape((-1, 3)))))
    
    def _init_parallel(self, *args, **kwargs):
        print('init', args)
        return np.arange(len(self.indices))
    
    def _post_parallel(self, all_results, *args, **kwargs):
        #print(self)
        print("done:", all_results)
        #print("not_done:", not_done)
        return self

    def post_parallel(self, all_results, *args, **kwargs):
        if any([isinstance(res, Exception) for res in all_results]):
            print([res for res in all_results if isinstance(res, Exception)])
            return self
        out_batch = EcgBatch(self.index)
        list_of_arrs = [x[0] for x in all_results]
        list_of_arrs.append(np.array([]))
        data = np.array(list_of_arrs)[:-1]
        if len(all_results) > 0:
            keys = list(all_results[0][1].keys())
        else:
            keys = []
        annot = {}
        for k in keys:
            list_of_arrs = [x[1][k] for x in all_results]
            list_of_arrs.append(np.array([]))
            annot[k] = np.array(list_of_arrs)[:-1]
        meta = dict(ChainMap(*[x[2] for x in all_results]))
        return out_batch.update(data=data,
                                annot=annot,
                                meta=meta)

    
    @action
    @inbatch_parallel(init="init_parallel", post="post_parallel", target='mpc')
    def resample_mpc(self, new_fs=100):
        """
        Resample signals to new fixed rate given by new_fs
        """
        return partial(resample_compile, new_fs=new_fs)

    @action
    def pad(self, target_length=None, pad_annotation=False):
        """
        Left zero padding upto target_length or batch_maximal_length
        """
        out_batch = EcgBatch(self.index)
        list_of_arrs = []
        cur_meta = self._meta.copy()

        if target_length is None:
            target_length = max(len(x[0]) for x in self._data)

        for ecg in self.indices:
            signal, _, _ = self[ecg]
            if len(signal[0]) > target_length:
                list_of_arrs.append(signal[:, -target_length:])
                left_pad =  target_length - len(signal[0])
            else:
                left_pad = target_length - len(signal[0])
                list_of_arrs.append(np.pad(signal, pad_width=((0, 0), (left_pad, 0)),
                                           mode='constant', constant_values=0))
            cur_meta[ecg]['pad'] = left_pad

        annot = self._annotation.copy()
        if pad_annotation:
            for key in annot:
                pad_arrs = []
                ann_arrs = annot[key]
                for arr in ann_arrs:
                    if len(arr[0]) > target_length:
                        arr = arr[:, -target_length:]
                    else:
                        arr = np.pad(arr, pad_width=((0, 0), (left_pad, 0)),
                                     mode='constant', constant_values=0)
                    pad_arrs.append(arr)
                annon[key] = np.array(pad_arrs)
        out_batch.update(data=np.array(list_of_arrs),
                         annot=annot,
                         meta=cur_meta)
        return out_batch

    @action
    def add_ref(self, path):
        """
        Loads labels for Challenge dataset from file REFERENCE.csv
        """
        ref = pd.read_csv(path, header=None)
        ref.columns = ['file', 'diag']
        ref = ref.set_index('file')  #pylint: disable=no-member
        for ecg in self.indices:
            self._meta[ecg]['diag'] = ref.ix[ecg]['diag']
        return self
    
    @action
    def gradient(self, order):
        """
        ...
        """
        out_batch = EcgBatch(self.index)
        out_annot = self._annotation.copy()
        list_of_arrs = []

        for ecg in self.indices:
            signal, annot, meta = self[ecg]
            grad = np.gradient(signal, axis=1)
            for i in range(order-1):
                grad = np.gradient(grad, axis=1)
            list_of_arrs.append(grad)

        list_of_arrs.append(np.array([]))
        out_annot['grad_{0}'.format(order)] = np.array(list_of_arrs)[:-1]
        out_batch.update(data=self._data.copy(),
                         annot=out_annot,
                         meta=self._meta.copy())
        return out_batch

    @action
    def convolve(self, layer, kernel=None):
        """
        ...
        """
        out_batch = EcgBatch(self.index)
        out_annot = self._annotation.copy()
        list_of_arrs = []
 
        if kernel is None:
            fs = 300
            kernel = gaussian(int(100 * fs / 300), int(10 * fs / 300))

        if layer == 'signal':
            data = self._data
        else:
            data = self._annotation[layer]

        for signal in data:
            res = np.apply_along_axis(np.convolve, 1, signal**2, v=kernel, mode='same')
            list_of_arrs.append(res)

        list_of_arrs.append(np.array([]))
        out_annot[layer+'_conv'] = np.array(list_of_arrs)[:-1]
        out_batch.update(data=self._data.copy(),
                         annot=out_annot,
                         meta=self._meta.copy())
        return out_batch

    @action
    def hmm_estimate(self, layers, n_components=3, n_iter=25):
        """
        Computes r-peaks using hmm
        """
        out_batch = EcgBatch(self.index)
        out_annot = self._annotation.copy()

        list_of_arrs = []
        for ecg in self.indices:
            signal, annot, meta = self[ecg]
            arrs = []
            for layer in layers:
                if layer == 'signal':
                    arrs.extend(signal)
                else:
                    arrs.extend(annot[layer])
            pred = sps.hmm_estimate(np.array(arrs).T, n_components, n_iter)
            list_of_arrs.append(pred.reshape((1, -1)))

        list_of_arrs.append(np.array([]))
        out_annot['hmm_estimate'] = np.array(list_of_arrs)[:-1]
        out_batch.update(data=self._data.copy(),
                         annot=out_annot,
                         meta=self._meta.copy())
        return out_batch

    @action
    def get_r_peaks(self, prior):
        """
        Computes r-peaks using hmm
        """
        out_batch = EcgBatch(self.index)
        out_annot = self._annotation.copy()

        list_of_arrs = []
        for ecg in self.indices:
            signal, annot, meta = self[ecg]
            arrs = []
            r_peak = sps.get_r_peaks(annot['hmm_estimate'][0], annot[prior][0])
            list_of_arrs.append(r_peak.reshape((1, -1)))

        list_of_arrs.append(np.array([]))
        out_annot['R_peaks'] = np.array(list_of_arrs)[:-1]
        out_batch.update(data=self._data.copy(),
                         annot=out_annot,
                         meta=self._meta.copy())
        return out_batch

    @action
    def clean_annotation(self, keys=None):
        """
        ...
        """
        out_batch = EcgBatch(self.index)
        out_annot = self._annotation.copy()

        if keys is None:
            out_annot.clear()
        else:
            for key in keys:
                out_annot.pop(key)
        out_batch.update(data=self._data.copy(),
                         annot=out_annot,
                         meta=self._meta.copy())
        return out_batch

    @action
    def add_hmm_pred(self, pred_set):
        """
        Add r-peaks from pred_set
        """
        ann_data = []
        for pred, ecg in zip(pred_set, self.indices):
            signal, _, meta = self[ecg]
            r_peak = sps.get_r_peaks(signal[0], pred, meta['fs'])
            ann_data.append(r_peak.reshape((1, -1)))
        ann_data.append(np.array([]))
        ann_data = np.array(ann_data)[:-1]
        self._annotation['R_peaks'] = ann_data
        return self

    @action
    def butter_bandpass_filter(self, lowcut, highcut, order=5):
        """
        Apply butter_bandpass_filter
        """
        out_batch = EcgBatch(self.index)
        list_of_arrs = []
        for ecg in self.indices:
            signal, _, meta = self[ecg]
            fs = meta['fs']
            arr = sps.butter_bandpass_filter(signal[0], lowcut,
                                             highcut, fs, order)
            list_of_arrs.append(arr.reshape(1, -1))
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         annot=self._annotation.copy(),
                         meta=self._meta.copy())
        return out_batch

    @action
    def resample(self, new_fs):
        """
        Resample signals to new fixed rate given by new_fs
        """
        out_batch = EcgBatch(self.index)
        cur_meta = self._meta.copy()
        list_of_arrs = []
        for ecg in self.indices:
            signal, _, meta = self[ecg]
            fs = meta['fs']
            new_len = int(new_fs * len(signal[0]) / fs)
            list_of_arrs.append(resample_poly(signal, new_len,
                                              len(signal[0]), axis=1))
            cur_meta[ecg]['fs'] = new_fs
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         annot=self._annotation.copy(),
                         meta=cur_meta)
        return out_batch

    @action
    def wavelet_coefs(self, wavelet, resample=True):
        """
        Multilevel discrete wavelet transform. Returns ordered list of
        coefficients arrays [cA_n, cD_n, cD_n-1, ..., cD2, cD1] where n
        denotes the level of decomposition
        """
        out_batch = EcgBatch(self.index)
        list_of_arrs = []
        for ecg in self.indices:
            signal, _, _ = self[ecg]
            if resample:
                new_len = 2**int(np.log2(len(signal[0])))
                w_coef = pywt.wavedec(resample_poly(signal, new_len,
                                                    len(signal[0]), axis=1),
                                      wavelet=wavelet, mode='per', axis=1)
            else:
                w_coef = pywt.wavedec(signal, wavelet=wavelet, mode='per', axis=1)
            list_of_arrs.append(w_coef)
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         annot=self._annotation.copy(),
                         meta=self._meta.copy())
        return out_batch
    
    @action
    def wavelet_filter(self, wavelet, from_levels=None, drop_levels=None, 
                       sigma=None, t_mode=None, crop=False, t_low=10, t_high=90):
        """
        ...
        """
        out_batch = EcgBatch(self.index)
        out_annot = self._annotation.copy()
        
        list_of_arrs = []
        for ecg in self.indices:
            signal, _, _ = self[ecg]
            if crop:
                new_len = 2**int(np.log2(len(signal[0])))
                w_coef = pywt.wavedec(signal[0][:new_len].reshape((1, -1)), 
                                      wavelet=wavelet, mode='per', axis=1)
            else:
                w_coef = pywt.wavedec(signal, wavelet=wavelet, mode='per', axis=1)

            to_zero = np.arange(len(w_coef), dtype = int)
            if from_levels is not None:
                to_zero = np.delete(to_zero, from_levels[from_levels < len(w_coef)])
            elif drop_levels is not None:
                to_zero = drop_levels[drop_levels < len(w_coef)]
            else:
                to_zero = []
            for i in to_zero:
                w_coef[-i] *= 0
            '''  
            if sigma is not None:
                uthresh = sigma * np.sqrt(2 * np.log(len(signal[0])))
                if t_mode is None:
                    t_mode = 'soft'
                w_coef = [pywt.threshold(x[0], value=uthresh, mode=t_mode).reshape((1, -1)) for x in w_coef]
                '''
            for i in range(len(w_coef)):
                sign = 2 * (w_coef[i] >= 0).astype(int) - 1
                low, high = np.percentile(np.abs(w_coef[i]), [t_low, t_high])
                w_coef[i] = sign * threshold(threshold(np.abs(w_coef[i]), threshmin=low, newval=0), 
                                             threshmax=high, newval=high)

            rec_signal = pywt.waverec(w_coef, wavelet=wavelet, mode='per', axis=1)
            list_of_arrs.append(rec_signal)

        list_of_arrs.append(np.array([]))
        out_annot['DWT_filter'] = np.array(list_of_arrs)[:-1]
        out_batch.update(data=self._data.copy(),
                         annot=out_annot,
                         meta=self._meta.copy())
        return out_batch

    @action
    def wavelet_spectrogram(self, wavelet):
        """
        Multilevel discrete wavelet transform. Returns spectrogram as
        [time, scale, power].
        """
        out_batch = EcgBatch(self.index)
        list_of_arrs = []
        for ecg in self.indices:
            signal, _, _ = self[ecg]
            time_ax, scale_ax, power = sps.wavelet_spectrogram(signal, wavelet)
            list_of_arrs.append([time_ax, scale_ax, power])
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         annot=self._annotation.copy(),
                         meta=self._meta.copy())
        return out_batch


    @action
    def drop_nan(self):
        """
        ...
        """
        list_of_arrs = []
        list_of_indices = []
        list_of_pos = []
        for ecg in self.indices:
            signal, _, meta = self[ecg]
            if np.isnan(signal).any():
                continue
            list_of_indices.append(ecg)
            list_of_pos.append(meta['__pos'])

        list_of_indices = np.array(list_of_indices)
        list_of_pos = np.array(list_of_pos)

        out_batch = EcgBatch(list_of_indices)
        out_meta = {i: self._meta[i] for i in list_of_indices}
        for i, ind in enumerate(list_of_indices):
            out_meta[ind]['__pos'] = i
        out_batch.update(data=self._data[list_of_pos],
                         annot={k: v[list_of_pos] for k, v in self._annotation.items()},
                         meta=out_meta)
        return out_batch


    @action
    def pca_reduce(self, n_components):
        """
        ...
        """
        out_batch = EcgBatch(self.index)
        list_of_arrs = []
        pca = PCA(n_components=n_components)
        for ecg in self.indices:
            signal, _, _ = self[ecg]
            reduced = []
            for channel in signal:
                try:
                    reduced.append(pca.fit_transform(channel))
                except ValueError:
                    print(channel)
                    print(ecg)
            list_of_arrs.append(np.array(reduced))
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         meta=self._meta.copy())
        return out_batch


    @action
    def fft_spectrogram(self, window, overlay=0):
        """
        Windowed FFT. Returns spectrogram as [time, scale, power].
        """
        if (overlay < 0) or (overlay > 1):
            raise ValueError("Overlay should be in [0, 1]", overlay)

        out_batch = EcgBatch(self.index)
        list_of_arrs = []
        for ecg in self.indices:
            signal, _, _ = self[ecg]
            time_ax, scale_ax, power = sps.fft_spectrogram(signal, window, overlay)
            list_of_arrs.append([time_ax, scale_ax, power])
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         annot=self._annotation.copy(),
                         meta=self._meta.copy())
        return out_batch


    @action
    def stack_r_peaks(self, target_length):
        """
        Returns segments of length target_length centered at r-pears
        """
        out_batch = EcgBatch(self.index)
        list_of_arrs = []

        for ecg in self.indices:
            signal, annot, _ = self[ecg]
            start = annot['r_peak_start'][0]
            stop = annot['r_peak_stop'][0]
            c_start, c_stop = sps.segment_centering(start, stop, target_length)
            stacked = sps.stack_segments(signal[0], c_start, c_stop,
                                         unify=False, rate=target_length)
            list_of_arrs.append(stacked.T)
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         meta=self._meta.copy())
        return out_batch

    @action
    def segment_align(self, max_iter=10, align_to='median'):
        """
        ...
        """
        out_batch = EcgBatch(self.index)
        list_of_arrs = []

        for ecg in self.indices:
            signal, _, _ = self[ecg]
            aligned = sps.segment_alignment(signal.T, max_iter=max_iter, 
                                            align_to=align_to)
            list_of_arrs.append(aligned.T)
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         annot=self._annotation.copy(),
                         meta=self._meta.copy())
        return out_batch

    @action
    def loc_segments(self, segment_type):
        """
        Returns start and stop positons of requested segment_type
        """
        available_segment_types = {'r_peak', 'rr_interval', 'period'}
        if segment_type not in available_segment_types:
            raise KeyError('Unknown interval type {0}.' 
                           'Available types are'.format(segment_type),
                           available_segment_types)

        out_batch = EcgBatch(self.index)
        cur_annot = self._annotation.copy()

        ann_start = []
        ann_stop = []

        for ecg in self.indices:
            _, annot, _ = self[ecg]
            rpeak = annot['R_peaks'][0]
            if segment_type == 'r_peak':
                start, stop = sps.loc_segments(rpeak, 'peak')
            if segment_type == 'rr_interval':
                start, stop = sps.loc_segments(-rpeak+1, 'peak')
            if segment_type == 'period':
                start, stop = sps.loc_segments(rpeak, 'period')

            ann_start.append(start.reshape((1, -1)))
            ann_stop.append(stop.reshape((1, -1)))

        ann_start.append(np.array([]))
        ann_start = np.array(ann_start)[:-1]
        cur_annot[segment_type + '_start'] = ann_start

        ann_stop.append(np.array([]))
        ann_stop = np.array(ann_stop)[:-1]
        cur_annot[segment_type + '_stop'] = ann_stop

        out_batch.update(data=self._data.copy(),
                         annot=cur_annot,
                         meta=self._meta.copy())
        return out_batch

    @action
    def mean_profile(self, segment_type, rate=None):
        """
        Returns mean signal profile within requested segment_type
        """
        available_segment_types = {'r_peak', 'rr_interval', 'period'}
        if segment_type not in available_segment_types:
            raise KeyError('Unknown interval type {0}.' 
                           'Available types are'.format(segment_type),
                           available_segment_types)
        if rate is None:
            rate = 100

        out_batch = EcgBatch(self.index)
        list_of_arrs = []

        for ecg in self.indices:
            signal, annot, _ = self[ecg]
            start = annot[segment_type + '_start'][0]
            stop = annot[segment_type + '_stop'][0]
            stacked = sps.stack_segments(signal[0], start, stop, rate)
            list_of_arrs.append(stacked.mean(axis=0).reshape((1, -1)))

        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         meta=self._meta.copy())
        return out_batch

    @action
    def stack_segments(self, segment_type, rate=None):
        """
        Returns stacked segments of given segment_type
        """
        available_segment_types = {'r_peak', 'rr_interval', 'period'}
        if segment_type not in available_segment_types:
            raise KeyError('Unknown interval type {0}.' 
                           'Available types are'.format(segment_type),
                           available_segment_types)
        if rate is None:
            rate = 100

        out_batch = EcgBatch(self.index)
        list_of_arrs = []

        for ecg in self.indices:
            signal, annot, _ = self[ecg]
            start = annot[segment_type + '_start'][0]
            stop = annot[segment_type + '_stop'][0]
            stacked = sps.stack_segments(signal[0], start, stop, rate)
            list_of_arrs.append(stacked.T)

        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         meta=self._meta.copy())
        return out_batch

    @action
    def segment_describe(self, segment_type):
        """
        Returns some statistics of signal within requested segment_type
        """
        available_segment_types = {'r_peak', 'rr_interval', 'period'}
        if segment_type not in available_segment_types:
            raise KeyError('Unknown interval type {0}.' 
                           'Available types are'.format(segment_type),
                           available_segment_types)

        out_batch = EcgBatch(self.index)
        list_of_arrs = []

        for ecg in self.indices:
            signal, annot, _ = self[ecg]
            start = annot[segment_type + '_start'][0]
            stop = annot[segment_type + '_stop'][0]
            statistics = sps.segment_features(signal[0], start, stop)
            list_of_arrs.append(np.array(statistics))

        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         meta=self._meta.copy())
        return out_batch

    @action
    def show_signal(self, ecg, ax=None):
        """
        Plot ecg signal
        """
        #%matplotlib notebook
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self[ecg][0][0])
        plt.show()

    @action
    def ecg_viewer(self, ecg, axarr=None):
        """
        Plot ecg signal and distribution of signal within period
        """
        #%matplotlib notebook
        if axarr is None:
            _, axarr = plt.subplots(2, 2)

        fontsize = 9
        signal, annot, meta = self[ecg]
        axarr[0, 0].plot(signal[0])
        axarr[0, 0].set_title('ECG signal', fontsize=fontsize)

        start = annot['period_start']
        stop = annot['period_stop']
        mean, interval = sps.segment_profile(signal[0], start[0],
                                             stop[0], rate=100)
        sps.show_segment_profile(mean, interval, axarr[0, 1])
        axarr[0, 1].set_title('Mean RR cycle with CI', fontsize=fontsize)

        sps.show_hist(start[0], stop[0], ax=axarr[1, 0])
        axarr[1, 0].set_title('Distribution of RR cycle length',
                              fontsize=fontsize)

        stacked = sps.stack_segments(signal[0], start[0], stop[0], rate=500)
        im = stacked / np.abs(stacked).max()
        axarr[1, 1].imshow(im, origin='low', aspect='auto')
        axarr[1, 1].set_title('Stacked RR cycles', fontsize=fontsize)

        plt.suptitle('Type - %s' % meta['diag'])
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=None, hspace=0.8)
        plt.show()
