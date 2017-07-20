""" contain Batch class for processing ECGs """

import os
import sys
import copy
import itertools
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, log_loss
from sklearn.externals import joblib

from keras.layers import Input, Conv1D, Lambda, \
                         MaxPooling1D, MaxPooling2D, \
                         Dense, GlobalMaxPooling2D
from keras.layers.core import Dropout
from keras.models import Model, model_from_yaml
from keras.optimizers import Adam
import keras.backend as K

from hmmlearn import hmm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import dataset as ds
from .ecg_batch_tools import * #pylint: disable=wildcard-import, unused-wildcard-import
from .keras_extra_layers import RFFT, Crop, Inception2D


class EcgBatch(ds.Batch):#pylint: disable=too-many-public-methods
    """
    Batch of ECG data
    """
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded)
        self._data = (None, None, None)
        self.signal = np.array([])
        self.annotation = {}
        self.meta = dict()
        self.history = []

    @property
    def components(self):
        return "signal", "annotation", "meta"

    @ds.action
    @ds.inbatch_parallel(init='indices', post="post_parallel", target='threads')
    def load_ecg(self, index, src, fmt):
        """
        Loads ecg data

        Arguments
        index: list or array of ecg indices.
        src: dict of type index: path to ecg.
        fmt: format of ecg files. Supported formats: 'wfdb', 'npz'.
        """
        if src:
            path = src[index]
        else:
            path = self.index.get_fullpath(index)
        if fmt == 'wfdb':
            return load_wfdb(index, path)
        elif fmt == 'npz':
            return load_npz(index, path)
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
        return (self.signal[pos],
                {k: v[pos] for k, v in self.annotation.items()},
                self.meta[index])

    def update(self, data=None, annot=None, meta=None):
        """
        Update content of ecg_batch
        """
        if data is not None:
            self.signal = np.array(data)
        if annot is not None:
            self.annotation = annot
        if meta is not None:
            self.meta = meta
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
        if len(valid_results) == 0:
            print('Error: all resulta are None')
            return self

        list_of_arrs = [x[0] for x in valid_results]
        list_of_lens = np.array([len(x[0]) for x in valid_results])
        list_of_annot = np.array([x[1] for x in valid_results]).ravel()
        list_of_meta = np.array([x[2] for x in valid_results]).ravel()
        list_of_origs = np.array([x[3] for x in valid_results]).ravel()

        if max(list_of_lens) <= 1:
            ind = ds.DatasetIndex(index=list_of_origs)
        else:
            ind = ds.DatasetIndex(index=np.arange(sum(list_of_lens), dtype=int))
        out_batch = EcgBatch(ind)

        if list_of_arrs[0].ndim > 3:
            raise ValueError('Signal is expected to have ndim = 1, 2 or 3, found ndim = {0}'
                             .format(list_of_arrs[0].ndim))
        if list_of_arrs[0].ndim in [1, 3]:
			#list_of_arrs[0] has shape (nb_signals, nb_channels, siglen)
			#ndim = 3 for signals with similar siglens and 1 for signals with differenr siglens
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
                for _ in range(rep):
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
                for _ in range(rep):
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
        list_of_distr: list of tuples (distr, params). See augment_fssignal for details.
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
    def drop_label(self, label):
        '''
        Drop signals labeled as label from the batch.

        Arguments
        label: label to be dropped from batch
        ''' 
        _ = label
        return drop_label

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
            self.meta[ecg]['diag'] = ref.ix[ecg]['diag']
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

    @ds.action
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
        x = Input((None, 1))

        conv_1 = Conv1D(4, 4, activation='relu')(x)
        mp_1 = MaxPooling1D()(conv_1)

        conv_2 = Conv1D(8, 4, activation='relu')(mp_1)
        mp_2 = MaxPooling1D()(conv_2)
        conv_3 = Conv1D(16, 4, activation='relu')(mp_2)
        mp_3 = MaxPooling1D()(conv_3)
        conv_4 = Conv1D(32, 4, activation='relu')(mp_3)

        fft_1 = RFFT()(conv_4)
        crop_1 = Crop(begin=0, size=128)(fft_1)
        to2d = Lambda(K.expand_dims)(crop_1)

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

    @ds.action
    def replace_labels(self, model_name, new_labels):
        '''
        Replace original labels by new labels.

        Arguments
        model_name: name of the model where to replace labels.
        new_labels: new labels to replace previous.
        '''
        model_comp = list(self.get_model_by_name(model_name))
        model_comp[2] = list(new_labels.values())
        return self.replace_all_labels(new_labels)

    @ds.action
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

    def get_categorical_labels(self, model_name):
        '''
        Returns a dummy matrix given an array of categorical labels.

        Arguments
        model_name: name of the model that will use dummy martix.
        '''
        classes = self.get_model_by_name(model_name)[2]
        labels = [self.meta[ind]['diag'] for ind in self.indices]
        return pd.get_dummies(classes + labels).as_matrix()[len(classes):]

    @ds.action
    def train_on_batch(self, model_name):
        '''
        Train model
        '''
        model_comp = self.get_model_by_name(model_name)
        model, hist, _ = model_comp
        train_x = np.array([x for x in self.signal]).reshape((-1, 3000, 1))
        train_y = self.get_categorical_labels(model_name)
        res = model.train_on_batch(train_x, train_y)
        pred = model.predict(train_x)
        y_pred = get_pos_of_max(pred)
        hist['train_loss'].append(res)
        hist['train_metric'].append(f1_score(train_y, y_pred, average='macro'))
        return self

    @ds.action
    def validate_on_batch(self, model_name):
        '''
        Validate model
        '''
        model_comp = self.get_model_by_name(model_name)
        model, hist, _ = model_comp
        test_x = np.array([x for x in self.signal]).reshape((-1, 3000, 1))
        test_y = self.get_categorical_labels(model_name)
        pred = model.predict(test_x)
        y_pred = get_pos_of_max(pred)
        hist['val_loss'].append(log_loss(test_y, pred))
        hist['val_metric'].append(f1_score(test_y, y_pred, average='macro'))
        return self

    @ds.action
    def model_summary(self, model_name):
        '''
        Print model layers
        '''
        model_comp = self.get_model_by_name(model_name)
        print(model_name)
        print(model_comp[0].summary())
        return self

    @ds.action
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

    @ds.action
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

    @ds.action
    def train_hmm(self, model_name):
        '''
        Train hmm model on the whole batch
        '''
        warnings.filterwarnings("ignore")
        model = self.get_model_by_name(model_name)
        train_x = np.concatenate(self.signal, axis=1).T
        lengths = [x.shape[1] for x in self.signal]
        model.fit(train_x, lengths)
        return self

    @ds.action
    def predict_hmm(self, model_name):
        '''
        Get hmm predictited classes
        '''
        model = self.get_model_by_name(model_name)
        return self.predict_all_hmm(model)

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel", target='mpc')
    def predict_all_hmm(self, model):
        '''
        Get hmm predictited classes
        '''
        _ = model
        return predict_hmm_classes

    @ds.action
    def save_hmm_model(self, model_name, fname):
        '''
        Save hmm model
        '''
        model = self.get_model_by_name(model_name)
        joblib.dump(model, fname + '.pkl')
        return self

    @ds.action
    def load_hmm_model(self, model_name, fname):
        '''
        Load hmm model
        '''
        model = self.get_model_by_name(model_name)#pylint: disable=unused-variable
        model = joblib.load(fname + '.pkl')
        return self

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel", target='mpc')
    def gradient(self, order):
        """
        Compute derivative of given order and add it to annotation
        """
        _ = order
        return get_gradient

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel", target='mpc')
    def convolve(self, layer, kernel):
        """
        Convolve layer with kernel
        """
        _ = layer, kernel
        return convolve_layer

    @ds.action
    @ds.inbatch_parallel(init="init_parallel", post="post_parallel",
                         target='mpc')
    def merge_layers(self, list_of_layers):
        """
        Merge layers from list of layers to signal
        """
        _ = list_of_layers
        return merge_list_of_layers

    def input_check_post(self, all_results, *args, **kwargs):
        """ Post function to gather and handle results of check-ups
        """
        _ = args, kwargs
        if ds.any_action_failed(all_results):
            all_errors = self.get_errors(all_results)
            print(all_errors)
            raise ValueError("Checkup failed")

    @ds.action
    @ds.inbatch_parallel(
        init='indices', post='input_check_post')
    def check_signal_length(self, index, operator=np.greater_equal, length=0):
        """Check if real length of the signal is appropriate.
        Args:
        operator - operator to use in check-up (np.greater_equal by default)
        length - value to compare with real signal length
        """
        pos = self.index.get_pos(index)
        if operator(self.signal[pos].shape[1], length):
            return True
        else:
            raise InputDataError('Signal length is wrong')

    @ds.action
    @ds.inbatch_parallel(
        init='indices', post='input_check_post')
    def check_signal_fs(self, index, desired_fs=None):
        """Check if sampling rate of the signal equals to desired
        sampling rate.
        """
        pos = self.index.get_pos(index)
        if np.equals(self.meta[pos]['fs'], desired_fs):
            return True
        else:
            raise InputDataError('Signal sampling rate is wrong')
