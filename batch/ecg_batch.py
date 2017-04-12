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

from scipy.signal import resample_poly
from dataset import Batch, action

sys.path.append('..')

from signal_process import signal_process as sps


class EcgBatch(Batch):
    """
    Batch of ECG data
    """
    def __init__(self, index):
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
        """
        Loads data from different sources
        src is not used yet, so files locations are defined by the index
        """
        if fmt == "wfdb":
            list_of_arrs, list_of_annotations, meta = self._load_wfdb(src)
        elif fmt == "npz":
            list_of_arrs, list_of_annotations, meta = self._load_npz()
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
            signal, fields = wfdb.rdsamp(os.path.splitext(path)[0])
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

    def _load_npz(self):
        """
        Load signal and meta, loading of annotation should be added
        """
        list_of_arrs = []
        list_of_annotations = {}
        meta = {}
        for pos, ecg in np.ndenumerate(self.indices):
            path = self.index.get_fullpath(ecg)
            data = np.load(path)
            list_of_arrs.append(data["signal"])
            # annot = self.create_annotation_df(data["annotation"])
            # list_of_annotations.append(annot)
            fields = data["meta"].item()
            fields.update({"__pos": pos[0]})
            meta.update({ecg: fields})
        return list_of_arrs, list_of_annotations, meta

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

    @action
    def dump(self, dst, fmt="npz"):
        """
        Save each ecg in its own file named as '<index>.<fmt>'
        """
        if fmt == "npz":
            for ecg in self.indices:
                signal, ann, meta = self[ecg]
                # del meta["__pos"]
                np.savez(os.path.join(dst, ecg + "." + fmt),
                         signal=signal,
                         annotation=ann, meta=meta.pop('__pos'))
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
                left_pad = len(signal[0]) - target_length
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
    def wavelet_coefs(self, wavelet):
        """
        Multilevel discrete wavelet transform. Returns ordered list of
        coefficients arrays [cA_n, cD_n, cD_n-1, ..., cD2, cD1] where n
        denotes the level of decomposition
        """
        out_batch = EcgBatch(self.index)
        list_of_arrs = []
        for ecg in self.indices:
            signal, _, _ = self[ecg]
            new_len = 2**int(np.log2(len(signal[0])))
            w_coef = pywt.wavedec(resample_poly(signal, new_len,
                                                len(signal[0]), axis=1),
                                  wavelet=wavelet, mode='per', axis=1)
            list_of_arrs.append(w_coef)
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         annot=self._annotation.copy(),
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
            power = np.stack(res_t)
            list_of_arrs.append([time_ax, scale_ax, power])
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         annot=self._annotation.copy(),
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
            time_ax, scale_ax, power = fft_spectrogram(signal, window, overlay)
            list_of_arrs.append([time_ax, scale_ax, power])
        list_of_arrs.append(np.array([]))
        out_batch.update(data=np.array(list_of_arrs)[:-1],
                         annot=self._annotation.copy(),
                         meta=self._meta.copy())
        return out_batch

    @action
    def hmm_estimate(self, kernel=None, n_iter=25):
        """
        Computes r-peaks using hmm
        """
        out_batch = EcgBatch(self.index)
        annot = copy.deepcopy(self._annotation)

        ann_data = []
        for ecg in self.indices:
            signal, _, meta = self[ecg]
            pred = sps.hmm_estimate(signal[0], meta['fs'], kernel, n_iter)
            r_peak = sps.get_r_peaks(signal[0], pred, meta['fs'], kernel)
            ann_data.append(r_peak.reshape((1, -1)))

        ann_data.append(np.array([]))
        ann_data = np.array(ann_data)[:-1]
        annot['R_peaks'] = ann_data
        out_batch.update(data=self._data.copy(),
                         annot=annot,
                         meta=self._meta.copy())
        return out_batch

    def loc_segments(self, segment_type):
        """
        Returns start and stop positons of requested segment_type
        """
        available_segment_types = {'r_peak', 'rr_interval', 'period'}
        if segment_type not in available_segment_types:
            raise KeyError('Unknown interval type {0}.' \
                           'Available types are'.format(segment_type),
                           available_segment_types)

        out_batch = EcgBatch(self.index)
        cur_annot = self._annotation.copy()

        ann_start = []
        ann_stop = []

        for ecg in self.indices:
            signal, annot, _ = self[ecg]
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

    def mean_profile(self, segment_type, rate=None):
        """
        Returns mean signal profile within requested segment_type
        """
        available_segment_types = {'r_peak', 'rr_interval', 'period'}
        if segment_type not in available_segment_types:
            raise KeyError('Unknown interval type {0}.' \
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

    def segment_describe(self, segment_type):
        """
        Returns some statistics of signal within requested segment_type
        """
        available_segment_types = {'r_peak', 'rr_interval', 'period'}
        if segment_type not in available_segment_types:
            raise KeyError('Unknown interval type {0}.' \
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

    def show_signal(self, ecg, ax=None):
        """
        Plot ecg signal
        """
        #%matplotlib notebook
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self[ecg][0][0])
        plt.show()

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
        axarr[1, 0].set_title('Distribution of RR cycle length', fontsize=fontsize)

        stacked = sps.stack_segments(signal[0], start[0], stop[0], rate=500)
        im = stacked / np.abs(stacked).max()
        axarr[1, 1].imshow(im, origin='low', aspect='auto')
        axarr[1, 1].set_title('Stacked RR cycles', fontsize=fontsize)

        plt.suptitle('Type - %s' % meta['diag'])
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=None, hspace=0.8)
        plt.show()
