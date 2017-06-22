""" contain Batch class for processing ECGs """

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import resample_poly
from sklearn.metrics import classification_report, f1_score, log_loss
from numba import jit

from keras.engine.topology import Layer
from keras.layers import Input, Conv1D, Conv2D, \
                         MaxPooling1D, MaxPooling2D, Lambda, \
                         Reshape, Dense, GlobalMaxPooling2D
from keras.layers.core import Dropout
from keras.layers.merge import Concatenate
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam
from keras.utils import np_utils

import wfdb

import dataset as ds


sys.path.append('..')

class RFFT(Layer):
    '''
    Keras layer for one-dimensional discrete Fourier Transform for real input.
    Computes rfft transforn on each slice along last dim.

    Arguments
    None

    Input shape
    3D tensor (batch_size, signal_length, nb_channels)

    Output shape
    3D tensor (batch_size, int(signal_length / 2), nb_channels)
    '''
    def __init__(self, *agrs, **kwargs):
        super(RFFT, self).__init__(*agrs, **kwargs)

    def fft(self, x):
        '''
        Computes one-dimensional discrete Fourier Transform on each slice along last dim.
        Returns amplitude spectrum.

        Arguments
        x: 3D tensor (batch_size, signal_length, nb_channels)

        Retrun
        out: 3D tensor (batch_size, signal_length, nb_channels) of type tf.float32
        '''
        import tensorflow as tf
        resh = tf.map_fn(tf.transpose, tf.cast(x, dtype=tf.complex64))
        spec = tf.cast(tf.abs(tf.fft(resh)), dtype=tf.float32)
        out = tf.map_fn(tf.transpose, spec)
        return out

    def call(self, x):
        res = Lambda(self.fft)(x)
        half = int(res.get_shape().as_list()[1] / 2)
        return res[:, :half, :]

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (input_shape[0], int(input_shape[1] / 2), input_shape[2])


class Crop(Layer):
    '''
    Keras layer returns cropped signal.

    Arguments
    begin: begin of the cropped segment
    size: size of the cropped segment

    Input shape
    3D tensor (batch_size, signal_length, nb_channels)

    Output shape
    3D tensor (batch_size, size, nb_channels)
    '''
    def __init__(self, begin, size, *agrs, **kwargs):
        self.begin = begin
        self.size = size
        super(Crop, self).__init__(*agrs, **kwargs)

    def call(self, x):
        return x[:, self.begin: self.begin + self.size, :]

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (input_shape[0], self.size, input_shape[2])


class To2D(Layer):
    '''
    Keras layer add dim to 1D signal and returns 2D image.

    Arguments
    None

    Input shape
    3D tensor (batch_size, signal_length, nb_channels)

    Output shape
    4D tensor (batch_size, size, nb_channels, 1)
    '''
    def __init__(self, *agrs, **kwargs):
        super(To2D, self).__init__(*agrs, **kwargs)

    def call(self, x):
        shape_1d = x.get_shape().as_list()[1:]
        shape_1d.append(1)
        to2d = Reshape(shape_1d)(x)
        return to2d

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (*input_shape, 1)


class Inception2D(Layer):
    '''
    Keras layer implements inception block.

    Arguments
    base_dim: nb_filters for the first convolution layers.
    nb_filters: nb_filters for the second convolution layers.
    kernel_size_1: kernel_size for the second convolution layer.
    kernel_size_2: kernel_size for the second convolution layer.
    activation: activation function for each convolution, default is 'linear'.

    Input shape
    4D tensor (batch_size, width, height, nb_channels)

    Output shape
    4D tensor (batch_size, width, height, 3 * nb_filters + base_dim)
    '''
    def __init__(self, base_dim, nb_filters,#pylint: disable=too-many-arguments
                 kernel_size_1, kernel_size_2,
                 activation=None, *agrs, **kwargs):
        self.base_dim = base_dim
        self.nb_filters = nb_filters
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.activation = activation if activation is not None else 'linear'
        super(Inception2D, self).__init__(*agrs, **kwargs)

    def call(self, x):
        conv_1 = Conv2D(self.base_dim, (1, 1),
                        activation=self.activation, padding='same')(x)

        conv_2 = Conv2D(self.base_dim, (1, 1),
                        activation=self.activation, padding='same')(x)
        conv_2a = Conv2D(self.nb_filters, (self.kernel_size_1, self.kernel_size_1),
                         activation=self.activation, padding='same')(conv_2)

        conv_3 = Conv2D(self.base_dim, (1, 1),
                        activation=self.activation, padding='same')(x)
        conv_3a = Conv2D(self.nb_filters, (self.kernel_size_2, self.kernel_size_2),
                         activation=self.activation, padding='same')(conv_3)

        pool = MaxPooling2D(strides=(1, 1), padding='same')(x)
        conv_4 = Conv2D(self.nb_filters, (1, 1),
                        activation=self.activation, padding='same')(pool)

        return Concatenate(axis=-1)([conv_1, conv_2a, conv_3a, conv_4])

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (*input_shape[:-1], self.base_dim + 3 * self.nb_filters)


#@jit(nogil=True)
def back_to_categorical(data, col_names):
    '''
    Convert dummy matrix to categorical array. Returns array with categorical labels.

    Arguments
    data: dummy matrix of shape (num_items, num_labels). Only one element in a row is 1, other
          elements should be 0.
    col_names: array or list of len = num_labels. Contains names of each colunm in data.
    '''
    res = np.array([col_names[x] for x in data.astype(bool)]).ravel()
    return res


#@jit(nogil=True)
def get_pred_classes(pred, y_true, unq_classes):
    '''
    Returns predicted and true labeles.

    Arguments
    pred: ndarray of shape (nb_items, nb_classes) with probability of each class for each item.
    y_true: dummy matrix or array with true classes.
    unq_classes: rray or list of len = nb_classes. Contains names of each colunm in pred.
    '''
    labels = np.zeros(pred.shape, dtype=int)
    for i in range(len(labels)):
        labels[i, np.argmax(pred[i])] = 1

    y_pred = back_to_categorical(labels, unq_classes)
    if y_true.ndim > 1:
        y_true = back_to_categorical(y_true, unq_classes)
    return y_true, y_pred


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
    signal = resample_poly(signal, new_len, len(signal[0]), axis=1)
    out_meta = {**meta, 'fs': new_fs}
    return [signal, annot, out_meta, index]


#@jit(nogil=True)
def segment_signal(signal, annot, meta, index, length, step, pad):
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
    """
    if signal.ndim != 2:
        raise ValueError('Signal should have ndim = 2, found ndim = {0}'.format(signal.ndim))

    if len(signal[0]) < length:
        if pad:
            pad_len = length - len(signal[0])
            segments = np.lib.pad(signal, ((0, 0), (pad_len, 0)),
                                  'constant', constant_values=(0, 0))[np.newaxis, :, :]
            return [segments, {}, {'diag': meta['diag']}, index]
        else:
            raise ValueError('Signal is shorter than segment length: %i < %i'
                             % (len(signal[0]), length))

    shape = signal.shape[:-1] + (signal.shape[-1] - length + 1, length)
    strides = signal.strides + (signal.strides[-1],)
    segments = np.lib.stride_tricks.as_strided(signal, shape=shape,
                                               strides=strides)[:, ::step, :]
    segments = np.transpose(segments, (1, 0, 2))

    _ = annot
    out_annot = {} #TODO: segment annotation
    out_meta = {'diag': meta['diag']}
    return [segments, out_annot, out_meta, index]


def drop_noise(signal, annot, meta, index):
    '''
    Drop signals labeled as noise in meta. Retruns input if signal is not labeles as noise and
    retruns None otherwise.

    Arguments
    signal, annot, meta, index: componets of ecg signal.
    '''
    if meta['diag'] == '~':
        return None
    else:
        return [signal, annot, meta, index]


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
    return resample_signal(signal, annot, meta, index, new_fs)


def augment_fs_signal_mult(signal, annot, meta, index, list_of_distr):
    '''
    Multiple augmentation of signal to random sampling rates. New sampling rates are sampled
    from list of probability distributions with specified parameters.

    Arguments
    signal, annot, meta, index: componets of ecg signal.
    list_of_distr: list of tuples (distr, params). See augment_fs_signal for details.
    '''
    res = [augment_fs_signal(signal, annot, meta, index, distr_type, params)
           for (distr_type, params) in list_of_distr]
    out_sig = [x[0] for x in res]
    out_annot = [x[1] for x in res]
    out_meta = [x[2] for x in res]
    out_sig.append([])
    return [np.array(out_sig)[:-1], out_annot, out_meta, index]


class EcgBatch(ds.Batch):#pylint: disable=too-many-public-methods
    """
    Batch of ECG data
    """
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded)
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
            path = self.index.get_fullpath(ecg) if src is None else src[ecg]
            fullpath, _ = os.path.splitext(path)
            record = wfdb.rdsamp(fullpath)
            signal = record.__dict__.pop('p_signals')
            fields = record.__dict__
            signal = signal.T
            list_of_arrs.append(signal)
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
                path = os.path.join(src, ecg + '.npz')
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
            #pos = self.index.get_pos(index)
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
        Build ecg_batch from a list of items either [signal, annot, meta] or None.
        All Nones are ignored.
        Signal can be either a single signal or a list of signals.
        If signal is a list of signals, annot and meta can be a single annot and meta
        or a list of annots and metas of the same lentgh as the list of signals. In the
        first case annot and meta are broadcasted to each signal in the list of signals.

        Arguments
        all results: list of items either [signal, annot, meta] or None
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

        if list_of_arrs[0].ndim > 3:
            raise ValueError('Signal is expected to have ndim = 1, 2 or 3, found ndim = {0}'
                             .format(list_of_arrs[0].ndim))
        if list_of_arrs[0].ndim in [1, 3]:
            list_of_arrs = list(itertools.chain([x for y in list_of_arrs
                                                 for x in y]))
        list_of_arrs.append([])
        batch_data = np.array(list_of_arrs)[:-1]

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
        '''
        Resample all signals in batch along axis=1 to new sampling rate. Retruns resampled batch with modified meta.
        Resampling of annotation will be implemented in the future.

        Arguments
        new_fs: target signal sampling rate in Hz.
        '''
        _ = new_fs
        return resample_signal

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def augment_fs(self, list_of_distr):
        '''
        Multiple augmentation of signals in batch to random sampling rates. New sampling rates are sampled
        from list of probability distributions with specified parameters.

        Arguments
        list_of_distr: list of tuples (distr, params). See augment_fs_signal for details.
        '''
        _ = list_of_distr
        return augment_fs_signal_mult

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def segment(self, length, step, pad):
        """
        Segment signals in batch along axis=1 with constant step to segments with constant length.
        If signal is shorter than target segment length, signal is zero-padded on the left if
        pad is True or raise ValueError if pad is False.
        Segmentation of annotation will be implemented in the future.

        Arguments
        length: length of segment.
        step: step along axis=1 of the signal.
        pad: whether to apply zero-padding to short signals.
        """
        _ = length, step, pad
        return segment_signal

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def drop_noise(self):
        '''
        Drop signals labeled as noise from the batch.

        Arguments
        None
        '''
        return drop_noise

    @ds.action
    def load_labels(self, path):
        """
        Load labels from file with signal labels. File should have a csv format
        and contain 2 columns: index of ecg and label.

        Arguments
        path: path to the file with labels
        """
        ref = pd.read_csv(path, header=None)
        ref.columns = ['index', 'diag']
        ref = ref.set_index('index')  #pylint: disable=no-member
        for ecg in self.index.indices:
            self._meta[ecg]['diag'] = ref.ix[ecg]['diag']
        return self

    @ds.model()
    def fft_inception():#pylint: disable=too-many-locals
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

        fft_1 = RFFT()(conv_4)
        crop_1 = Crop(0, 128)(fft_1)
        to2d = To2D()(crop_1)

        incept_1 = Inception2D(4, 4, 3, 5, activation='relu')(to2d)
        mp2d_1 = MaxPooling2D(pool_size=(4, 2))(incept_1)

        incept_2 = Inception2D(4, 8, 3, 5, activation='relu')(mp2d_1)
        mp2d_2 = MaxPooling2D(pool_size=(4, 2))(incept_2)

        incept_3 = Inception2D(4, 12, 3, 3, activation='relu')(mp2d_2)

        pool = GlobalMaxPooling2D()(incept_3)

        fc_1 = Dense(8, kernel_initializer='uniform', activation='relu')(pool)
        drop = Dropout(0.2)(fc_1)

        fc_2 = Dense(2, kernel_initializer='uniform',
                     activation='softmax')(drop)

        opt = Adam()
        model = Model(inputs=x, outputs=fc_2)
        model.compile(optimizer=opt, loss="categorical_crossentropy")

        hist = {'train_loss': [], 'train_metric': [],
                'val_loss': [], 'val_metric': []}
        diag_code = {'A': 'A', 'N': 'nonA', 'O': 'nonA'}

        lr_schedule = [[0, 50, 100], [0.01, 0.001, 0.0001]]

        return model, hist, diag_code, lr_schedule

    @ds.action()
    def get_categorical_labels(self, new_labels=None):
        '''
        Returns a dummy matrix given an array of categorical variables and list of categories.
        Original labels will be replaced by new labels if encode is not None.

        Arguments
        encode: None or dict with new labels
        '''
        labels = []
        for ind in self.indices:
            diag = self._meta[ind]['diag']
            if new_labels is None:
                labels.extend([diag])
            else:
                labels.extend([new_labels[diag]])

        if new_labels is None:
            encode_labels = []
        else:
            encode_labels = list(np.unique(list(new_labels.values())))
        labels = encode_labels + labels
        unq_classes, num_labels = np.unique(labels, return_inverse=True)
        ctg_labels = np_utils.to_categorical(num_labels)[len(encode_labels):]
        return ctg_labels, unq_classes

    @ds.action()
    def train_on_batch(self, model_name):
        '''
        Train model
        '''
        model_comp = self.get_model_by_name(model_name)
        model, hist, code, lr_s = model_comp
        train_x = np.array([x for x in self._signal]).reshape((-1, 3000, 1))
        train_y, unq_classes = self.get_categorical_labels(new_labels=code)
        epoch_num = len(hist['train_loss'])
        if epoch_num in lr_s[0]:
            new_lr = lr_s[1][lr_s[0].index(epoch_num)]
            opt = Adam(lr=new_lr)
            model.compile(optimizer=opt, loss="categorical_crossentropy")

        res = model.train_on_batch(train_x, train_y)
        pred = model.predict(train_x)
        y_true, y_pred = get_pred_classes(pred, train_y, unq_classes)
        hist['train_loss'].append(res)
        hist['train_metric'].append(f1_score(y_true, y_pred, average='macro'))
        return self

    @ds.action()
    def validate_on_batch(self, model_name):
        '''
        Validate model
        '''
        model_comp = self.get_model_by_name(model_name)
        model, hist, code, lr_s = model_comp
        test_x = np.array([x for x in self._signal]).reshape((-1, 3000, 1))
        test_y, unq_classes = self.get_categorical_labels(new_labels=code)
        pred = model.predict(test_x)
        y_true, y_pred = get_pred_classes(pred, test_y, unq_classes)
        hist['val_loss'].append(log_loss(test_y, pred))
        hist['val_metric'].append(f1_score(y_true, y_pred, average='macro'))
        return self

    @ds.action()
    def model_summary(self, model_name):
        '''
        Print model layers
        '''
        model_comp = self.get_model_by_name(model_name)
        print(model_name)
        print(model_comp[0].summary())
        return self

    @ds.action()
    def save_model(self, model_name, fname):
        '''
        Save model layers and weights
        '''
        model_comp = self.get_model_by_name(model_name)
        model = model_comp[0]
        model.save_weights(fname)
        yaml_string = model.to_yaml()
        fout = open(fname + ".layers", "w")
        fout.write(yaml_string)
        fout.close()
        return self

    @ds.action()
    def load_model(self, model_name, fname):
        '''
        Load model layers and weights
        '''
        model_comp = self.get_model_by_name(model_name)
        model = model_comp[0]
        fin = open(fname + ".layers", "r")
        yaml_string = fin.read()
        fin.close()
        model = model_from_yaml(yaml_string)
        model.load_weights(fname)
        return self
