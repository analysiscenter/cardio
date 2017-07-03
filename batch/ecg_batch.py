""" contain Batch class for processing ECGs """

import os
import sys
import copy
import itertools
import warnings
import numpy as np
import pandas as pd

from scipy.signal import resample_poly
from sklearn.metrics import f1_score, log_loss
from sklearn.externals import joblib
from numba import njit

from keras.engine.topology import Layer
from keras.layers import Input, Conv1D, Conv2D, \
                         MaxPooling1D, MaxPooling2D, Lambda, \
                         Reshape, Dense, GlobalMaxPooling2D
from keras.layers.core import Dropout
from keras.layers.merge import Concatenate
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam
import keras.backend as K

import wfdb
from hmmlearn import hmm

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

    def fft(self, x, fft_fn):
        '''
        Computes one-dimensional discrete Fourier Transform on each slice along last dim.
        Returns amplitude spectrum.

        Arguments
        x: 3D tensor (batch_size, signal_length, nb_channels)
        fft_fn: function that performs fft

        Retrun
        out: 3D tensor (batch_size, signal_length, nb_channels) of type tf.float32
        '''
        resh = K.cast(K.map_fn(K.transpose, x), dtype='complex64')
        spec = K.abs(K.map_fn(fft_fn, resh))
        out = K.cast(K.map_fn(K.transpose, spec), dtype='float32')
        return out

    def call(self, x):
        res = Lambda(self.fft, arguments={'fft_fn': K.tf.fft})(x)
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


class Inception2D(Layer):#pylint: disable=too-many-instance-attributes
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

    def build(self, input_shape):
        x = Input(input_shape[1:])
        self.conv_1 = Conv2D(self.base_dim, (1, 1),#pylint: disable=attribute-defined-outside-init
                             activation=self.activation, padding='same')
        _ = self.conv_1(x)
        self.trainable_weights.extend(self.conv_1.trainable_weights)

        self.conv_2 = Conv2D(self.base_dim, (1, 1),#pylint: disable=attribute-defined-outside-init
                             activation=self.activation, padding='same')
        out = self.conv_2(x)
        self.trainable_weights.extend(self.conv_2.trainable_weights)
        self.conv_2a = Conv2D(self.nb_filters,#pylint: disable=attribute-defined-outside-init
                              (self.kernel_size_1, self.kernel_size_1),
                              activation=self.activation, padding='same')
        out = self.conv_2a(out)
        self.trainable_weights.extend(self.conv_2a.trainable_weights)

        self.conv_3 = Conv2D(self.base_dim, (1, 1),#pylint: disable=attribute-defined-outside-init
                             activation=self.activation, padding='same')
        out = self.conv_3(x)
        self.trainable_weights.extend(self.conv_3.trainable_weights)
        self.conv_3a = Conv2D(self.nb_filters,#pylint: disable=attribute-defined-outside-init
                              (self.kernel_size_2, self.kernel_size_2),
                              activation=self.activation, padding='same')
        out = self.conv_3a(out)
        self.trainable_weights.extend(self.conv_3a.trainable_weights)

        self.conv_4 = Conv2D(self.nb_filters, (1, 1),#pylint: disable=attribute-defined-outside-init
                             activation=self.activation, padding='same')
        _ = self.conv_4(x)
        self.trainable_weights.extend(self.conv_4.trainable_weights)

        return super(Inception2D, self).build(input_shape)

    def call(self, x):
        conv_1 = self.conv_1(x)

        conv_2 = self.conv_2(x)
        conv_2a = self.conv_2a(conv_2)

        conv_3 = self.conv_3(x)
        conv_3a = self.conv_3a(conv_3)

        pool = MaxPooling2D(strides=(1, 1), padding='same')(x)
        conv_4 = self.conv_4(pool)

        return Concatenate(axis=-1)([conv_1, conv_2a, conv_3a, conv_4])

    def compute_output_shape(self, input_shape):
        '''
        Get output shape
        '''
        return (*input_shape[:-1], self.base_dim + 3 * self.nb_filters)

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

def segment_signal(signal, annot, meta, index, length, step, pad, return_copy):
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
    return_copy: if True, a copy of segments is returned and segments become intependent. If False,
                 segments are not independent, but segmentation runtime becomes almost indepentent on
                 signal length.

    Attention: segmentation of meta and annotation is not implemented yet.
    """
    if signal.ndim != 2:
        raise ValueError('Signal should have ndim = 2, found ndim = {0}'.format(signal.ndim))

    if signal.shape[1] < length:
        if pad:
            pad_len = length - signal.shape[1]
            segments = np.lib.pad(signal, ((0, 0), (pad_len, 0)),
                                  'constant', constant_values=(0, 0))[np.newaxis, :, :]
            return [segments, {}, meta, index]
        else:
            raise ValueError('Signal is shorter than segment length: %i < %i'
                             % (signal.shape[1], length))

    shape = signal.shape[:-1] + (signal.shape[-1] - length + 1, length)
    strides = signal.strides + (signal.strides[-1],)
    segments = np.lib.stride_tricks.as_strided(signal, shape=shape,
                                               strides=strides)[:, ::step, :]
    segments = np.transpose(segments, (1, 0, 2))

    _ = annot
    if return_copy:
        segments = segments.copy()
    return [segments, {}, meta, index]

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

def replace_labels_in_meta(signal, annot, meta, index, new_labels):
    '''
    Replaces diag label by new label.

    Arguments
    new_labels: dict of previous and corresponding new labels.
    '''
    meta.update({'diag': new_labels[meta['diag']]})
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
    res = np.concatenate(res, axis=0)
    return [res, annot, meta, index]

def load_wfdb(index, path):
    """
    Load signal and meta, loading of annotation should be added
    """
    record = wfdb.rdsamp(path)
    signal = record.__dict__.pop('p_signals')
    meta = record.__dict__
    signal = signal.T
    return [signal, {}, meta, index]

def load_npz(index, path):
    """
    Load signal and meta, loading of annotation should be added
    """
    data = np.load(path)
    signal = data["signal"]
    annot = data["annotation"].tolist()
    meta = data["meta"].tolist()
    return [signal, annot, meta, index]

def dump_ecg_signal(signal, annot, meta, index, path, fmt):
    """
    Save ecg in a separate file as 'path/<index>.<fmt>'
    """
    if fmt == "npz":
        np.savez(os.path.join(path, index + "." + fmt),
                 signal=signal,
                 annotation=annot,
                 meta=meta)
    else:
        raise NotImplementedError("The format is not supported yet")
    return [signal, annot, meta, index]

class EcgBatch(ds.Batch):#pylint: disable=too-many-public-methods
    """
    Batch of ECG data
    """
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded)
        self._data = (np.array([]), {}, dict())
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
    @ds.inbatch_parallel(init='indices', post="post_parallel", target='threads')
    def load_ecg(self, index, src, fmt):
        """
        Loads data from different sources
        src is not used yet, so files locations are defined by the index
        """
        if fmt == 'wfdb':
            return load_wfdb(index, src[index])
        elif fmt == 'npz':
            return load_npz(index, src[index])
        else:
            raise TypeError("Incorrect type of source")

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel", target='threads')
    def dump_ecg(self, signal, annot, meta, index, path, fmt):
        """
        Save each ecg in a separate file as 'path/<index>.<fmt>'
        """
        return dump_ecg_signal(signal, annot, meta, index, path, fmt)

    def __getitem__(self, index):
        try:
            pos = self.get_pos(None, None, index)
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
        return [[*self[i], i] for i in self.indices]

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
            metas = []
            for i, rep in enumerate(list_of_lens):
                for j in range(rep):#pylint: disable=unused-variable
                    metas.append(copy.deepcopy(list_of_meta[i]))
            metas = np.array(metas)
        for i in range(len(batch_data)):
            metas[i].update({'origin': origins[i]})
        batch_meta = dict(zip(ind.indices, metas))

        if len(ind.indices) == len(list_of_annot):
            annots = list_of_annot
        else:
            annots = []
            for i, rep in enumerate(list_of_lens):
                for j in range(rep):#pylint: disable=unused-variable
                    annots.append(copy.deepcopy(list_of_annot[i]))
            annots = np.array(annots)
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
    def split_to_segments(self, length, step, pad, return_copy):
        """
        Split signals along axis=1 to segments with constant length.
        If signal is shorter than target segment length, signal is zero-padded on the left if
        pad is True or raise ValueError if pad is False.
        Segmentation of annotation will be implemented in the future.

        Arguments
        length: length of segment.
        step: step along axis=1 of the signal.
        pad: whether to apply zero-padding to short signals.
        return_copy: if True, a copy of segments is returned and segments become intependent. If False,
                 segments are not independent, but segmentation runtime becomes almost indepentent on
                 signal length.
        """
        _ = length, step, pad, return_copy
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
        for ecg in self.indices:
            self._meta[ecg]['diag'] = ref.ix[ecg]['diag']
        return self

    @ds.model()
    def hmm_learn():
        """
        Hidden Markov Model to find n_components in signal
        """
        n_components = 3
        n_iter = 10
        warnings.filterwarnings("ignore")
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="full",
                                n_iter=n_iter)
        return model

    @ds.action()
    def set_new_model(self, model_name, new_model):
        '''
        Replace base model by new model.

        Arguments
        model_name: name of the model where to set new model.
        new_model: new model to replace previous.
        '''
        model = self.get_model_by_name(model_name)#pylint: disable=unused-variable
        model = new_model
        return self

    @ds.model()
    def fft_inception():#pylint: disable=too-many-locals
        '''
        FFT inception model. Includes initial convolution layers, then FFT transform, then
        a series of inception blocks.
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
        diag_classes = ['A', 'NonA']

        return model, hist, diag_classes

    @ds.action()
    def replace_labels(self, model_name, new_labels):
        '''
        Replace original labels by new labels.

        Arguments
        model_name: name of the model where to replece labels.
        new_labels: new labels to replace previous.
        '''
        model_comp = list(self.get_model_by_name(model_name))
        model_comp[2] = list(new_labels.values())
        return self.replace_all_labels(new_labels)

    @ds.action()
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def replace_all_labels(self, new_labels):
        '''
        Replace original labels by new labels.

        Arguments
        new_labels: dict of previous and corresponding new labels.
        '''
        _ = new_labels
        return replace_labels_in_meta

    @ds.action()
    def get_categorical_labels(self, model_name):
        '''
        Returns a dummy matrix given an array of categorical labels.

        Arguments
        model_name: name of the model that will use dummy martix.
        '''
        classes = self.get_model_by_name(model_name)[2]
        labels = [self._meta[ind]['diag'] for ind in self.indices]
        return pd.get_dummies(classes + labels).as_matrix()[len(classes):]

    @ds.action()
    def train_on_batch(self, model_name):
        '''
        Train model
        '''
        model_comp = self.get_model_by_name(model_name)
        model, hist, _ = model_comp
        train_x = np.array([x for x in self._signal]).reshape((-1, 3000, 1))
        train_y = self.get_categorical_labels(model_name)
        res = model.train_on_batch(train_x, train_y)
        pred = model.predict(train_x)
        y_pred = get_pos_of_max(pred)
        hist['train_loss'].append(res)
        hist['train_metric'].append(f1_score(train_y, y_pred, average='macro'))
        return self

    @ds.action()
    def validate_on_batch(self, model_name):
        '''
        Validate model
        '''
        model_comp = self.get_model_by_name(model_name)
        model, hist, _ = model_comp
        test_x = np.array([x for x in self._signal]).reshape((-1, 3000, 1))
        test_y = self.get_categorical_labels(model_name)
        pred = model.predict(test_x)
        y_pred = get_pos_of_max(pred)
        hist['val_loss'].append(log_loss(test_y, pred))
        hist['val_metric'].append(f1_score(test_y, y_pred, average='macro'))
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

    @ds.action()
    def train_hmm(self, model_name):
        '''
        Train hmm model on the whole batch
        '''
        warnings.filterwarnings("ignore")
        model = self.get_model_by_name(model_name)
        train_x = np.concatenate(self._signal, axis=1).T
        lengths = [x.shape[1] for x in self._signal]
        model.fit(train_x, lengths)
        return self

    @ds.action()
    def predict_hmm(self, model_name):
        '''
        Get hmm predictited classes
        '''
        model = self.get_model_by_name(model_name)
        return self.predict_all_hmm(model)

    @ds.action()
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def predict_all_hmm(self, model):
        '''
        Get hmm predictited classes
        '''
        _ = model
        return predict_hmm_classes

    @ds.action()
    def save_hmm_model(self, model_name, fname):
        '''
        Save hmm model
        '''
        model = self.get_model_by_name(model_name)
        joblib.dump(model, fname + '.pkl')
        return self

    @ds.action()
    def load_hmm_model(self, model_name, fname):
        '''
        Load hmm model
        '''
        model = self.get_model_by_name(model_name)#pylint: disable=unused-variable
        model = joblib.load(fname + '.pkl')
        return self

    @ds.action()
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def gradient(self, order):
        """
        Compute derivative of given order and add it to annotation
        """
        _ = order
        return get_gradient

    @ds.action()
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def convolve(self, layer, kernel):
        """
        Convolve layer with kernel
        """
        _ = layer, kernel
        return convolve_layer

    @ds.action()
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def merge_layers(self, list_of_layers):
        """
        Merge layers from list of layers to signal
        """
        _ = list_of_layers
        return merge_list_of_layers
