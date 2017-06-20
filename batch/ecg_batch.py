""" contain Batch class for processing ECGs """

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import resample_poly
from sklearn.metrics import classification_report, f1_score

from keras.layers import (Input, Conv1D, Conv2D,
                          MaxPooling1D, MaxPooling2D, Lambda,
                          Reshape, Dense, GlobalMaxPooling2D)
from keras.models import model_from_yaml, Model
from keras.layers.core import Dropout
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.engine.topology import Layer

import wfdb

import dataset as ds


sys.path.append('..')


class Inception2D(Layer):
    '''
    Keras inception layer.
    '''
    def __init__(self, base_dim, nb_filters,#pylint: disable=too-many-arguments
                 kernel_size_1, kernel_size_2,
                 activation=None, padding=None, *agrs, **kwargs):
        self.base_dim = base_dim
        self.nb_filters = nb_filters
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        if activation is None:
            self.activation = 'linear'
        else:
            self.activation = activation
        if padding is None:
            self.padding = 'same'
        else:
            self.padding = padding
        super(Inception2D, self).__init__(*agrs, **kwargs)

    def build(self, input_shape):
        super(Inception2D, self).build(input_shape)

    def call(self, x, mask=None):
        conv_1 = Conv2D(self.base_dim, (1, 1),
                        activation=self.activation, padding=self.padding)(x)

        conv_2 = Conv2D(self.base_dim, (1, 1),
                        activation=self.activation, padding=self.padding)(x)
        conv_2a = Conv2D(self.nb_filters, (self.kernel_size_1, self.kernel_size_1),
                         activation=self.activation, padding=self.padding)(conv_2)

        conv_3 = Conv2D(self.base_dim, (1, 1),
                        activation=self.activation, padding=self.padding)(x)
        conv_3a = Conv2D(self.nb_filters, (self.kernel_size_2, self.kernel_size_2),
                         activation=self.activation, padding=self.padding)(conv_3)

        pool = MaxPooling2D(strides=(1, 1), padding=self.padding)(x)
        conv_4 = Conv2D(self.nb_filters, (1, 1),
                        activation=self.activation, padding=self.padding)(pool)

        return Concatenate(axis=-1)([conv_1, conv_2a, conv_3a, conv_4])

    def compute_output_shape(self, input_shape):
        '''
        Get output shape of inception layer
        '''
        return (input_shape[0], input_shape[1], input_shape[2], self.base_dim + 3 * self.nb_filters)


def fft(x):
    '''
    tf fft
    '''
    import tensorflow as tf
    resh = tf.map_fn(tf.transpose, tf.cast(x, dtype=tf.complex64))
    res = tf.cast(tf.abs(tf.fft(resh)), dtype=tf.float32)
    return tf.map_fn(tf.transpose, res)


def rfft(x):
    '''
    tf fft
    '''
    res = fft(x)
    half = int(res.get_shape().as_list()[1] / 2)
    return res[:, :half, :]


def crop(x, lc, rc):
    '''
    Crop
    '''
    return x[:, lc: rc, :]


def get_ecg(i, fields):
    '''
    Return ecg signal, annot and meta by index
    '''
    data, annot, meta = fields
    pos = meta[i]['__pos']
    return (data[pos],
            {k: v[pos] for k, v in annot.items()},
            meta[i])


def back_to_annot(arr, annot):
    '''
    Convert categorical array to labeled array
    '''
    res = []
    for x in arr:
        res.append(annot[x == 1][0])
    return np.array(res)


def get_pred_classes(pred, y_true, unq_classes):
    '''
    Returns labeled prediction and true labeles
    '''
    labels = np.zeros(pred.shape, dtype=int)
    for i in range(len(labels)):
        labels[i, np.argmax(pred[i])] = 1

    y_pred = back_to_annot(labels, unq_classes)
    if y_true.ndim > 1:
        y_true = back_to_annot(y_true, unq_classes)
    return y_true, y_pred


def resample_signal(signal, annot, meta, index, new_fs):
    """
    Resample signal to new_fs
    """
    fs = meta['fs']
    new_len = int(new_fs * len(signal[0]) / fs)
    signal = resample_poly(signal, new_len, len(signal[0]), axis=1)
    out_meta = meta.copy()
    out_meta['fs'] = new_fs
    return [signal, annot, out_meta, index]


def segment_signal(signal, annot, meta, index, length, step, pad):
    #pylint: disable=too-many-arguments
    """
    Segment signal
    """
    diag = meta['diag']
    start = 0
    segments = []
    if len(signal[0]) < length:
        if pad:
            pad_len = length - len(signal[0])
            segments.append(np.lib.pad(signal, ((0, 0), (pad_len, 0)),
                                       'constant', constant_values=(0, 0)))
            return [np.array(segments), {}, {'diag': diag}, index]
        else:
            raise ValueError('Signal is shorter than segment length: %i < %i'
                             % (len(signal[0]), length))
    while start + length <= len(signal[0]):
        segments.append(signal[:, start: start + length])
        start += step
    _ = annot
    out_annot = {} #TODO: resample annotation
    out_meta = {'diag': diag}
    return [np.array(segments), out_annot, out_meta, index]


def noise_filter(signal, annot, meta, index):
    '''
    Drop signals labeled as noise
    '''
    if meta['diag'] == '~':
        return None
    else:
        return [signal, annot, meta, index]


def augment_fs_signal(signal, annot, meta, index, distr_type, params):
    '''
    Return resampled signal
    '''
    if distr_type == 'none':
        return [signal, annot, meta, index]
    np.random.seed()
    if distr_type == 'normal':
        new_fs = np.random.normal(**params)
    elif distr_type == 'uniform':
        new_fs = np.random.uniform(**params)
    elif distr_type == 'delta':
        new_fs = params['loc']
    new_sig, _, new_meta, _ = resample_signal(signal, annot,
                                              meta, index, new_fs)
    return [new_sig, {}, new_meta, index]


def augment_fs_signal_mult(signal, annot, meta, index, list_of_distr):
    '''
    Returns many resampled signals
    '''
    res = [augment_fs_signal(signal, annot, meta, index, distr_type, params)
           for (distr_type, params) in list_of_distr]
    out_sig = [x[0] for x in res]
    out_annot = [x[1] for x in res]
    out_meta = [x[2] for x in res]
    out_sig.append([])
    return [np.array(out_sig)[:-1], out_annot, out_meta, index]


class EcgBatch(ds.Batch):
    """
    Batch of ECG data
    """
    def __init__(self, index):
        super().__init__(index)
        self._data = (None, {}, dict())
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

    @ds.action
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
        #self._signal = np.array(list(itertools.chain([x for y in list_of_arrs for x in y])))
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
        for ecg in self.index.indices:
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
            meta.update({ecg: fields})

        return list_of_arrs, list_of_annotations, meta

    def _load_npz(self, src):
        """
        Load signal and meta, loading of annotation should be added
        """
        list_of_arrs = []
        list_of_annotations = []
        meta = {}
        for ecg in self.index.indices:
            if src is None:
                path = self.index.get_fullpath(ecg)
            else:
                path = src + '/' + ecg + '.npz'
            data = np.load(path)
            list_of_arrs.append(data["signal"])
            list_of_annotations.append(data["annotation"].tolist())
            fields = data["meta"].tolist()
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

    @ds.action
    def dump(self, dst, fmt="npz"):
        """
        Save each ecg in a separate file as '<ecg_index>.<fmt>'
        """
        if fmt == "npz":
            for ecg in self.indices:
                signal, ann, meta = self[ecg]
                np.savez(os.path.join(dst, ecg + "." + fmt),
                         signal=signal,
                         annotation=ann,
                         meta=meta)
        else:
            raise NotImplementedError("The format is not supported yet")
        return self

    def __getitem__(self, index):
        try:
            pos = np.where(self.index.indices == index)[0][0]
        except IndexError:
            raise IndexError("There is no such index in the batch: {0}"
                             .format(index))
        return (self._signal[pos],
                {k: v[pos] for k, v in self._annotation.items()},
                self._meta[index])

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
        '''
        Return array of ecg with index
        '''
        _ = args, kwargs
        return [[*self[i], i] for i in self.index.indices]

    def post_parallel(self, all_results, *args, **kwargs):
        #pylint: disable=too-many-locals
        #pylint: disable=too-many-branches
        '''
        Build ecg_batch
        Broadcasting is supported
        '''
        _ = args, kwargs
        if any([isinstance(res, Exception) for res in all_results]):
            print([res for res in all_results if isinstance(res, Exception)])
            return self

        valid_results = [res for res in all_results if res is not None]

        list_of_arrs = [x[0] for x in valid_results]
        list_of_lens = np.array([len(x[0]) for x in valid_results])
        list_of_annot = np.array([x[1] for x in valid_results]).ravel()
        list_of_meta = np.array([x[2] for x in valid_results]).ravel()
        list_of_origs = np.array([x[3] for x in valid_results]).ravel()

        if max(list_of_lens) <= 1:
            ind = ds.DatasetIndex(index=np.array(list_of_origs))
        else:
            ind = ds.DatasetIndex(index=np.arange(sum(list_of_lens), dtype=int))
        out_batch = EcgBatch(ind)

        if list_of_arrs[0].ndim in [1, 3]:
            list_of_arrs = list(itertools.chain([x for y in list_of_arrs
                                                 for x in y]))
            list_of_arrs.append([])
            batch_data = np.array(list_of_arrs)[:-1]
        elif list_of_arrs[0].ndim == 2:
            list_of_arrs.append([])
            batch_data = np.array(list_of_arrs)[:-1]
        else:
            raise ValueError('Signal is expected to have ndim = 1, 2 or 3, found ndim = {0}'
                             .format(list_of_arrs[0].ndim))

        if len(ind.indices) == len(list_of_origs):
            origins = list_of_origs
        else:
            origins = np.repeat(list_of_origs, list_of_lens)

        if len(ind.indices) == len(list_of_meta):
            metas = list_of_meta
        else:
            metas = np.repeat(list_of_meta, list_of_lens)
        for i in range(len(batch_data)):
            metas[i].update({'origin': origins[i]})
        batch_meta = dict(zip(ind.indices, metas))

        if len(ind.indices) == len(list_of_annot):
            annots = list_of_annot
        else:
            annots = np.repeat(list_of_annot, list_of_lens)
        if len(annots) > 0:
            keys = list(annots[0].keys())
        else:
            keys = []
        batch_annot = {}
        for k in keys:
            list_of_arrs = [x[k] for x in annots]
            list_of_arrs.append(np.array([]))
            batch_annot[k] = np.array(list_of_arrs)[:-1]

        return out_batch.update(data=batch_data,
                                annot=batch_annot,
                                meta=batch_meta)

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel", target='mpc')
    def resample(self, new_fs):
        """
        Resample all signals in batch to new_fs
        """
        _ = new_Fs
        return resample_signal

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def augment_fs(self, list_of_distr):
        """
        Segment all signals
        """
        _ = list_of_distr
        return augment_fs_signal_mult

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def segment(self, length, step, pad):
        """
        Segment all signals
        """
        _ = length, step, pad
        return segment_signal

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def drop_noise(self):
        """
        Segment all signals
        """
        return noise_filter

    @ds.action
    def add_ref(self, path):
        """
        Load labels from file REFERENCE.csv
        """
        ref = pd.read_csv(path, header=None)
        ref.columns = ['file', 'diag']
        ref = ref.set_index('file')  #pylint: disable=no-member
        for ecg in self.index.indices:
            self._meta[ecg]['diag'] = ref.ix[ecg]['diag']
        return self

    @ds.model()
    def fft_inception():#pylint: disable=too-many-locals, no-method-argument
        '''
        fft inception model
        '''
        x = Input((3000, 1))

        conv_1 = Conv1D(4, 4, activation='relu')(x)
        mp_1 = MaxPooling1D()(conv_1)

        conv_2 = Conv1D(8, 4, activation='relu')(mp_1)
        mp_2 = MaxPooling1D()(conv_2)
        conv_3 = Conv1D(16, 4, activation='relu')(mp_2)
        mp_3 = MaxPooling1D()(conv_3)
        conv_4 = Conv1D(32, 4, activation='relu')(mp_3)

        fft_1 = Lambda(rfft)(conv_4)
        shape_fft = fft_1.get_shape().as_list()
        crop_1 = Lambda(crop,
                        arguments={'lc': 0, 'rc': int(shape_fft[1] / 3)})(fft_1)

        shape_1d = crop_1.get_shape().as_list()[1:]
        shape_1d.append(1)
        to2d = Reshape(shape_1d)(crop_1)

        incept_1 = Inception2D(4, 4, 3, 5)(to2d)
        mp2d_1 = MaxPooling2D(pool_size=(4, 2))(incept_1)

        incept_2 = Inception2D(4, 8, 3, 5)(mp2d_1)
        mp2d_2 = MaxPooling2D(pool_size=(4, 2))(incept_2)

        incept_3 = Inception2D(4, 12, 3, 3)(mp2d_2)

        pool = GlobalMaxPooling2D()(incept_3)

        fc_1 = Dense(8, kernel_initializer='uniform', activation='relu')(pool)
        drop = Dropout(0.2)(fc_1)

        fc_2 = Dense(2, kernel_initializer='uniform',
                     activation='softmax')(drop)

        opt = Adam()
        model = Model(inputs=x, outputs=fc_2)
        model.compile(optimizer=opt, loss="categorical_crossentropy")

        hist = {'train_loss': [], 'val_loss': [],
                'val_metric': [], 'batch_size': None}
        diag_code = {'A': 'A', 'N': 'nonA', 'O': 'nonA'}

        lr_schedule = [[0, 50, 100], [0.01, 0.001, 0.0001]]

        return model, hist, diag_code, lr_schedule

    @ds.action()
    def get_labels(self, encode=None):
        '''
        Get categorical labels
        '''
        labels = []
        for ind in self.indices:
            diag = self._meta[ind]['diag']
            if encode is None:
                labels.extend([diag])
            else:
                labels.extend([encode[diag]])

        if encode is None:
            encode_labels = []
        else:
            encode_labels = list(np.unique(list(encode.values())))
        labels = encode_labels + labels
        unq_classes, num_labels = np.unique(labels, return_inverse=True)
        ctg_labels = np_utils.to_categorical(num_labels)[len(encode_labels):]
        return ctg_labels, unq_classes

    @ds.action(model='fft_inception')
    def train_fft_inception(self, model_comp, nb_epoch, batch_size):
        '''
        fft_incaption model
        '''
        model, hist, code, lr_s = model_comp
        hist['batch_size'] = batch_size
        train_x = np.array([x for x in self._signal]).reshape((-1, 3000, 1))
        train_y, _ = self.get_labels(encode=code)
        epoch_num = len(hist['train_loss'])
        if epoch_num in lr_s[0]:
            new_lr = lr_s[1][lr_s[0].index(epoch_num)]
            opt = Adam(lr=new_lr)
            model.compile(optimizer=opt, loss="categorical_crossentropy")

        res = model.fit(train_x, train_y,
                        epochs=nb_epoch,
                        batch_size=batch_size)
        hist['train_loss'].append(res.history["loss"][0])
        return self

    @ds.action(model='fft_inception')
    def print_summary(self, model_comp):
        '''
        Print model layers
        '''
        print(model_comp[0].summary())
        return self

    @ds.action(model='fft_inception')
    def calc_loss(self, model_comp):
        '''
        Add current val_loss and val_metric to training history
        '''
        model, hist, code, _ = model_comp
        test_x = np.array([x for x in self._signal]).reshape((-1, 3000, 1))
        test_y, unq_classes = self.get_labels(encode=code)
        pred = model.predict(test_x)
        batch_size = hist['batch_size']
        hist['val_loss'].append(model.evaluate(test_x, test_y,
                                               batch_size=batch_size,
                                               verbose=0))
        y_true, y_pred = get_pred_classes(pred, test_y, unq_classes)
        hist['val_metric'].append(f1_score(y_true, y_pred, average='macro'))
        return self

    @ds.action(model='fft_inception')
    def train_report(self, model_comp):
        '''
        Print loss and metrics at the end of epoch
        '''
        hist = model_comp[1]
        if len(hist['train_loss']) == 0:
            print('Train history is empty')
        else:
            print('Epoch', len(hist['train_loss']))
            print('train_loss: %3.4f   val_loss: %3.4f   val_metric: %3.4f'
                  % (hist['train_loss'][-1], hist['val_loss'][-1],
                     hist['val_metric'][-1]))
        return self

    @ds.action(model='fft_inception')
    def print_accuracy(self, model_comp):
        '''
        Print accuracy
        '''
        model, _, code, _ = model_comp
        test_x = np.array([x for x in self._signal]).reshape((-1, 3000, 1))
        test_y, unq_classes = self.get_labels(encode=code)
        pred = model.predict(test_x)
        y_true, y_pred = get_pred_classes(pred, test_y, unq_classes)
        print(classification_report(y_true, y_pred))
        print("f1_score", f1_score(y_true, y_pred, average='macro'))
        return self

    @ds.action(model='fft_inception')
    def show_loss(self, model_comp):
        '''
        Plot train and validation loss
        '''
        hist = model_comp[1]
        plt.plot(hist["train_loss"], "r", label="train loss")
        plt.plot(hist["val_loss"], "b", label="validation loss")
        plt.legend()
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()
        return self

    @ds.action(model='fft_inception')
    def save_model(self, model_comp, fname):
        '''
        Save model layers and weights
        '''
        model = model_comp[0]
        model.save_weights(fname)
        yaml_string = model.to_yaml()
        fout = open(fname + ".layers", "w")
        fout.write(yaml_string)
        fout.close()
        return self

    @ds.action(model='fft_inception')
    def load_model(self, model_comp, fname):
        '''
        Load model layers and weights
        '''
        model = model_comp[0]
        fin = open(fname + ".layers", "r")
        yaml_string = fin.read()
        fin.close()
        model = model_from_yaml(yaml_string)
        model.load_weights(fname)
        return self
