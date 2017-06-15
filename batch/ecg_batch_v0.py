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
from dataset_img import *
from dataset_img import action, model
from functools import partial

from keras.layers.recurrent import LSTM
from keras.layers.core import Dropout
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2
from keras.utils import np_utils
from keras.layers import *
from keras.models import Model, Sequential
from keras import backend as K
from sklearn.metrics import classification_report, f1_score, log_loss
from keras.models import model_from_yaml
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler

sys.path.append('..')


def Inception2D(x, base_dim, nb_filters, size_1, size_2,
                activation='linear', padding='same'):
    '''
    Inception block for 2D spectrogram.
    '''
    conv_1 = Convolution2D(base_dim, (1, 1),
                           activation=activation, padding=padding)(x)

    conv_2 = Convolution2D(base_dim, (1, 1),
                           activation=activation, padding=padding)(x)
    conv_2a = Convolution2D(nb_filters, (size_1, size_1),
                            activation=activation, padding=padding)(conv_2)

    conv_3 = Convolution2D(base_dim, (1, 1),
                           activation=activation, padding='same')(x)
    conv_3a = Convolution2D(nb_filters, (size_2, size_2),
                            activation=activation, padding=padding)(conv_3)

    pool = MaxPooling2D(strides=(1, 1), padding=padding)(x)
    conv_4 = Convolution2D(nb_filters, (1, 1),
                           activation=activation, padding=padding)(pool)

    concat = concatenate([conv_1, conv_2a, conv_3a, conv_4], axis=-1)
    return concat


def fft(x):
    '''
    tf fft
    '''
    import tensorflow as tf
    z = tf.map_fn(tf.transpose, tf.cast(x, dtype=tf.complex64))
    z2 = tf.cast(tf.abs(tf.fft(z)), dtype=tf.float32)
    return tf.map_fn(tf.transpose, z2)


def rfft(x):
    '''
    tf fft
    '''
    res = fft(x)
    half = int(res.get_shape().as_list()[1] / 2)
    return res[:, :half, :]


def crop(x, a, b):
    '''
    Crop
    '''
    return x[:, a: b, :]


def get_ecg(i, fields):
    data, annot, meta = fields
    pos = meta[i]['__pos']
    return (data[pos],
            {k: v[pos] for k, v in annot.items()},
            meta[i])


def back_to_annot(arr, annot):
    res = []
    for x in arr:
        res.append(annot[x == 1][0])
    return np.array(res)


def validate(pred, testY, unq_classes):
    labels = np.zeros(pred.shape, dtype=int)
    for i in range(len(labels)):
        labels[i, np.argmax(pred[i])] = 1

    y_pred = back_to_annot(labels, unq_classes)
    if testY.ndim > 1:
        y_true = back_to_annot(testY, unq_classes)
    else:
        y_true = testY

    print(classification_report(y_true, y_pred))
    print("f1_score", f1_score(y_true, y_pred, average='macro'))


def f1_metric(pred, testY, unq_classes):
    labels = np.zeros(pred.shape, dtype=int)
    for i in range(len(labels)):
        labels[i, np.argmax(pred[i])] = 1

    y_pred = back_to_annot(labels, unq_classes)
    if testY.ndim > 1:
        y_true = back_to_annot(testY, unq_classes)
    else:
        y_true = testY

    return f1_score(y_true, y_pred, average='macro')


def resample_signal(signal, annot, meta, index, new_fs):
    """
    Resample signal to new_fs
    """
    fs = meta['fs']
    new_len = int(new_fs * len(signal[0]) / fs)
    signal = resample_poly(signal, new_len, len(signal[0]), axis=1)
    meta['fs'] = new_fs
    return [signal, annot, {index: meta}]


def segment_signal(signal, annot, meta, index, length, step, pad):
    """
    Segment signal
    """
    diag = meta['diag']
    start = 0
    segments = []
    if(len(signal[0]) < length):
        if pad:
            pad_len = length - len(signal[0])
            segments.append(np.lib.pad(signal, ((pad_len, 0), (0, 0)),
                                       'constant', constant_values=(0, 0)))
            print(segments)
            return [np.array(segments), meta['diag'], index]
        else:
            raise ValueError('Signal is shorter than segment length: %i < %i'
                             % (len(signal[0]), length))
    while(start + length <= len(signal[0])):
        segments.append(signal[:, start: start + length])
        start += step
    return [np.array(segments), meta['diag'], index]


class EcgBatch(Batch):
    """
    Batch of ECG data
    """
    def __init__(self, index, *args, **kwargs):
        super().__init__(index)
        self._data = (None, self.create_annotation_df(), dict())
        self.history = []

    @property
    def _signal(self):
        """ Signal """
        return self._data[0] if self._data is not None else None

    @_signal.setter
    def _signal(self, value):
        """ Set labels """
        data = list(self._data)
        data[0] = value
        self._data = data

    @property
    def _annotation(self):
        """ Annotation """
        return self._data[1] if self._data is not None else None

    @_annotation.setter
    def _annotation(self, value):
        """ Set labels """
        data = list(self._data)
        data[1] = value
        self._data = data

    @property
    def _meta(self):
        """ Meta """
        return self._data[2] if self._data is not None else None

    @_meta.setter
    def _meta(self, value):
        """ Set labels """
        data = list(self._data)
        data[2] = value
        self._data = data

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
            list_of_arrs, list_of_annotations, meta = self._load_npz(src)
        else:
            raise TypeError("Incorrect type of source")

        # ATTENTION!
        # Construction below is used to overcome numpy bug:
        # adding empty array to list of arrays, then generating array
        # of arrays and removing the last item (empty array)
        list_of_arrs.append(np.array([]))
        self._signal = np.array(list_of_arrs)[:-1]
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
        Save each ecg in a separate file as '<ecg_index>.<fmt>'
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
        else:
            raise NotImplementedError("The format is not supported yet")
        return self

    def __getitem__(self, index):
        if index in self.indices:
            pos = self._meta[index]['__pos']
            return (self._signal[pos],
                    {k: v[pos] for k, v in self._annotation.items()},
                    self._meta[index])
        else:
            raise IndexError("There is no such index in the batch", index)

    def update(self, data=None, annot=None, meta=None):
        """
        Update content of ecg_batch
        """
        if data is not None:
            self._signal = np.array(data)
        if annot is not None:
            self._annotation = annot
        if meta is not None:
            self._meta = meta
        return self

    def init_parallel(self, *args, **kwargs):
        return [list(self[i]) + [i] for i in self.indices]

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

    def post_parallel_segment(self, all_results, *args, **kwargs):
        if any([isinstance(res, Exception) for res in all_results]):
            print([res for res in all_results if isinstance(res, Exception)])
            return self

        list_of_arrs = [x[0] for x in all_results]
        list_of_lens = [len(x[0]) for x in all_results]

        list_of_labels = [x[1] for x in all_results]

        list_of_origs = [x[2] for x in all_results]

        ind = DatasetIndex(index=np.arange(sum(list_of_lens), dtype=int))

        out_batch = EcgBatch(ind)
        data = np.concatenate(list_of_arrs)
        labels = np.repeat(list_of_labels, list_of_lens)
        origins = np.repeat(list_of_origs, list_of_lens)
        meta = {i: {'__pos': i, 'diag': labels[i], 'origin': origins[i]}
                for i in range(len(labels))}

        return out_batch.update(data=data,
                                annot={},
                                meta=meta)

    @action
    @inbatch_parallel(init="init_parallel", post="post_parallel", target='mpc')
    def resample(self, new_fs):
        """
        Resample all signals in batch to new_fs
        """
        return resample_signal

    @action
    @inbatch_parallel(init="init_parallel", post="post_parallel_segment",
                      target='mpc')
    def segment(self, length, step, pad):
        """
        Segment all signals
        """
        return segment_signal

    @action
    def add_ref(self, path):
        """
        Load labels from file REFERENCE.csv
        """
        ref = pd.read_csv(path, header=None)
        ref.columns = ['file', 'diag']
        ref = ref.set_index('file')  #pylint: disable=no-member
        for ecg in self.indices:
            self._meta[ecg]['diag'] = ref.ix[ecg]['diag']
        return self

    @model()
    def fft_inception():
        input = Input((3000, 1))

        conv_1 = Convolution1D(4, 4, activation='relu')(input)
        mp_1 = MaxPooling1D()(conv_1)

        conv_2 = Convolution1D(8, 4, activation='relu')(mp_1)
        mp_2 = MaxPooling1D()(conv_2)
        conv_3 = Convolution1D(16, 4, activation='relu')(mp_2)
        mp_3 = MaxPooling1D()(conv_3)
        conv_4 = Convolution1D(32, 4, activation='relu')(mp_3)

        fft_1 = Lambda(rfft)(conv_4)
        shape_fft = fft_1.get_shape().as_list()
        crop_1 = Lambda(crop,
                        arguments={'a': 0, 'b': int(shape_fft[1] / 3)})(fft_1)

        shape_1d = crop_1.get_shape().as_list()[1:]
        shape_1d.append(1)
        to2d = Reshape(shape_1d)(crop_1)

        incept_1 = Inception2D(to2d, 4, 4, 3, 5)
        mp2d_1 = MaxPooling2D(pool_size=(4, 2))(incept_1)

        incept_2 = Inception2D(mp2d_1, 4, 8, 3, 5)
        mp2d_2 = MaxPooling2D(pool_size=(4, 2))(incept_2)

        incept_3 = Inception2D(mp2d_2, 4, 12, 3, 3)

        pool = GlobalMaxPooling2D()(incept_3)

        fc_1 = Dense(8, kernel_initializer='uniform', activation='relu')(pool)
        drop = Dropout(0.2)(fc_1)

        fc_2 = Dense(2, kernel_initializer='uniform',
                     activation='softmax')(drop)

        opt = Adam()
        model = Model(inputs=input, outputs=fc_2)
        model.compile(optimizer=opt, loss="categorical_crossentropy")

        hist = {'train_loss': [], 'val_loss': [], 'val_metric': []}
        diag_code = {'A': 'A', 'N': 'nonA', 'O': 'nonA'}

        lr_schedule = [[0, 50, 100], [0.01, 0.001, 0.0001]]

        return model, hist, diag_code, lr_schedule

    @action()
    def get_labels(self, encode=None):
        dsY = []
        for ind in self.indices:
            diag = self._meta[ind]['diag']
            if encode is None:
                dsY.extend([diag])
            else:
                dsY.extend([encode[diag]])

        if encode is None:
            pos_labels = []
        else:
            pos_labels = list(np.unique(list(encode.values())))
        dsY = pos_labels + dsY
        unq_classes, num_labels = np.unique(dsY, return_inverse=True)
        catY = np_utils.to_categorical(num_labels)[len(pos_labels):]
        return catY, unq_classes

    @action(model='fft_inception', singleton=True)
    def train_fft_inception(self, model_comp, nb_epoch, batch_size):
        model, hist, code, lr_s = model_comp
        trainX = self._signal.reshape((-1, 3000, 1))
        trainY, enq_classes = self.get_labels(encode=code)
        epoch_num = len(hist['train_loss'])
        if epoch_num in lr_s[0]:
            new_lr = lr_s[1][lr_s[0].index(epoch_num)]
            opt = Adam(lr=new_lr)
            model.compile(optimizer=opt, loss="categorical_crossentropy")

        res = model.fit(trainX, trainY,
                        epochs=nb_epoch,
                        batch_size=batch_size)
        hist['train_loss'].append(res.history["loss"][0])
        return self

    @action(model='fft_inception')
    def print_summary(self, model_comp):
        print(model_comp[0].summary())
        return self

    @action(model='fft_inception')
    def calc_loss(self, model_comp):
        model, hist, code, _ = model_comp
        testX = self._signal.reshape((-1, 3000, 1))
        testY, unq_classes = self.get_labels(encode=code)
        pred = model.predict(testX)
        hist['val_loss'].append(log_loss(testY, pred))
        hist['val_metric'].append(f1_metric(pred, testY, unq_classes))
        #print(model.evaluate(testX, testY, batch_size=batch_size))
        return self

    @action(model='fft_inception')
    def train_report(self, model_comp):
        model, hist, code, _ = model_comp
        if len(hist['train_loss']) == 0:
            print('Train history is empty')
        else:
            print('Epoch', len(hist['train_loss']))
            print('train_loss: %3.2f   val_loss: %3.2f   val_metric: %3.2f'
                  % (hist['train_loss'][-1], hist['val_loss'][-1],
                     hist['val_metric'][-1]))
        return self

    @action(model='fft_inception')
    def print_accuracy(self, model_comp):
        model, hist, code, _ = model_comp
        testX = self._signal.reshape((-1, 3000, 1))
        testY, unq_classes = self.get_labels(encode=code)
        pred = model.predict(testX)
        validate(pred, testY, unq_classes)
        return self

    @action(model='fft_inception')
    def show_loss(self, model_comp):
        model, hist, _, _ = model_comp
        fig = plt.figure()
        plt.plot(hist["train_loss"], "r", label="train loss")
        plt.plot(hist["val_loss"], "b", label="validation loss")
        plt.legend()
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()
