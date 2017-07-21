"""Module for testing ecg_batch methods
"""

import os
import sys
from copy import deepcopy
import random

import numpy as np
import pytest

sys.path.append(os.path.join("."))

from batch import EcgBatch
from batch import ds

random.seed(170720143422)

@pytest.fixture(scope="module")
def setup_module_load(request):
    '''
    Fixture to setup module. Performs check for presence of test files,
    creates initial batch object.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    files = ["A00001.hea", "A00001.mat", "A00002.hea", "A00002.mat",
             "A00004.hea", "A00004.mat", "A00005.hea", "A00005.mat",
             "A00008.hea", "A00008.mat", "A00013.hea", "A00013.mat", "REFERENCE.csv"]
    # TODO: make better test for presence of files .hea and
    # REFERENCE.csv

    if np.all([os.path.isfile(os.path.join(path, file)) for file in files]):
        ind = ds.FilesIndex(path=os.path.join(path, '*.hea'), no_ext=True, sort=True)
        batch_init = EcgBatch(ind)
    else:
        raise FileNotFoundError("Test files not found in 'tests/data/'!")

    def teardown_module_load():
        '''
        Teardown module
        '''
        inds = ["A00001", "A00002", "A00004", 
                "A00005", "A00008", "A00013"]

        for ind in inds:
            os.remove(os.path.join(path, ind+".npz"))

    request.addfinalizer(teardown_module_load)
    return batch_init, path

@pytest.fixture(scope="class")
def setup_class_methods(request):
    '''
    Fixture to setup class to test EcgBatch methods separately.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    ind = ds.FilesIndex(path=os.path.join(path, '*.hea'), no_ext=True, sort=True)
    batch_loaded = EcgBatch(ind).load_ecg(src=None, fmt="wfdb")

    def teardown_class_methods():
        '''
        Teardown class
        '''
    request.addfinalizer(teardown_class_methods)
    return batch_loaded

@pytest.fixture(scope="class")
def setup_class_pipeline(request):
    '''
    Fixture to setup class to test EcgBatch methods in pipeline.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    ind = ds.FilesIndex(path=os.path.join(path, '*.hea'), no_ext=True, sort=True)
    ecg_dataset = ds.Dataset(ind, batch_class=EcgBatch)
    ecg_pipeline = ecg_dataset.pipeline().load_ecg(src=None, fmt="wfdb")
    def teardown_class_pipeline():
        '''
        Teardown class
        '''
        pass
    request.addfinalizer(teardown_class_pipeline)
    return ecg_dataset, ecg_pipeline


class TestEcgBatchLoad():
    '''
    Class for test of load / dump methods
    '''

    def test_load_wfdb(self, setup_module_load): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Test of wfdb loader
        '''
        batch = deepcopy(setup_module_load[0])
        batch = batch.load_ecg(src=None, fmt='wfdb')
        assert isinstance(batch.signal, np.ndarray)
        assert isinstance(batch.meta, dict)
        assert isinstance(batch.annotation, dict)
        assert batch.signal.shape == (6, )
        assert len(batch.meta) == 6
        assert len(batch[batch.indices[0]]) == 3
        del batch

    def test_dump(self, setup_module_load): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Test of dump to npz
        '''
        batch = deepcopy(setup_module_load[0])
        path = setup_module_load[1]
        batch = batch.load_ecg(src=None, fmt='wfdb')
        batch = batch.dump_ecg(path=path, fmt='npz')
        files = os.listdir(path)
        assert "A00001.npz" in files
        assert "A00004.npz" in files
        del batch

    def test_load_npz(self, setup_module_load): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Test of npz loader
        '''
        path = setup_module_load[1]
        ind = ds.FilesIndex(path=os.path.join(path, '*.npz'), no_ext=True, sort=True)
        batch = EcgBatch(ind)
        batch = batch.load_ecg(src=None, fmt='npz')
        assert isinstance(batch.signal, np.ndarray)
        assert isinstance(batch.meta, dict)
        assert isinstance(batch.annotation, dict)
        assert batch.signal.shape == (6, )
        assert batch.signal[0].shape == (1, 9000)
        assert len(batch.meta) == 6
        assert len(batch[batch.indices[0]]) == 3
        del batch


@pytest.mark.usefixtures("setup_class_methods")
class TestEcgBatchSingleMethods:
    '''
    Class for testing other single methods of EcgBatch class
    '''

    def test_augment_fs(self, setup_class_methods): #pylint:disable=no-self-use,redefined-outer-name
        '''
        Testing augmentation of sampling rate
        '''
        batch = deepcopy(setup_class_methods)
        old_fs = batch[batch.indices[0]][2]['fs']
        new_fs = old_fs+50
        batch = batch.augment_fs([("delta", {'loc': new_fs})])
        assert batch.indices.shape == (6,)
        assert batch["A00001"][2]['fs'] == new_fs
        assert batch["A00004"][2]['fs'] == new_fs

    def test_load_labels(self, setup_module_load, setup_class_methods): #pylint:disable=no-self-use,redefined-outer-name
        '''
        Testing of labels loader
        '''
        batch = deepcopy(setup_class_methods)
        path = setup_module_load[1]
        batch = batch.load_labels(os.path.join(path, "REFERENCE.csv"))
        if "diag" in batch["A00001"][2].keys():
            assert batch["A00001"][2]['diag'] == 'N'
            assert batch["A00004"][2]['diag'] == 'A'
            assert batch["A00008"][2]['diag'] == 'O'
        else:
            raise ValueError("No key 'diag' in meta!")

    def test_split_to_segments(self, setup_class_methods): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Testing of segmentator
        '''
        batch = deepcopy(setup_class_methods)
        batch = batch.split_to_segments(
            4500, 4499, pad=True, return_copy=False)
        assert batch.indices.shape == (16, )
        assert batch.signal[0].shape == (1, 4500)

    def test_replace_all_labels(self, setup_class_methods, setup_module_load): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Testing of label replacer
        '''
        batch = deepcopy(setup_class_methods)
        path = setup_module_load[1]
        batch = batch.load_labels(os.path.join(path, "REFERENCE.csv"))
        batch = batch.replace_all_labels({"A":"A", "N":"NonA", "O":"NonA"})
        assert batch["A00001"][2]['diag'] == "NonA"
        assert batch["A00004"][2]['diag'] == "A"
        assert batch["A00008"][2]['diag'] == "NonA"

    def test_update(self): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Testing of updater
        '''
        new_inds = ["A00001", "A00004"]
        ind = ds.DatasetIndex(index=new_inds)
        batch = EcgBatch(ind)

        seq1 = np.array([1, 2, 3]).reshape(1, -1)
        seq2 = np.array([1, 2, 3, 4]).reshape(1, -1)
        data = np.array([seq1, seq2, []], dtype=object)[:-1]
        annotation = None
        meta = dict(zip(new_inds, [{"new_meta":True}, {"new_meta":True}]))
        batch.update(data, annotation, meta)
        assert batch["A00001"][0].shape == (1, 3)
        assert batch["A00004"][0].shape == (1, 4)
        assert batch["A00001"][2]["new_meta"]
        assert batch["A00004"][2]["new_meta"]


@pytest.mark.usefixtures("setup_class_pipeline")
class TestEcgBatchPipelineMethods:
    ''' Class to test EcgBatch methods in pipeline.
    '''

    def test_cv_split(self, setup_class_pipeline):
        ecg_dtst = setup_class_pipeline[0]
        ecg_dtst.cv_split(shares=0.5)
        assert ecg_dtst.train.indices.shape == (3,)
        assert ecg_dtst.test.indices.shape == (3,)
        assert isinstance(ecg_dtst.train, ds.Dataset)
        assert isinstance(ecg_dtst.test, ds.Dataset)

        ecg_dtst.cv_split(shares=[0.5,0.49])
        assert ecg_dtst.train.indices.shape == (3,)
        assert ecg_dtst.test.indices.shape == (2,)
        assert ecg_dtst.validation.indices.shape == (1,)
        assert isinstance(ecg_dtst.train, ds.Dataset)
        assert isinstance(ecg_dtst.test, ds.Dataset)
        assert isinstance(ecg_dtst.validation, ds.Dataset)


    def test_base_ppln(self, setup_class_pipeline, setup_module_load):
        path = setup_module_load[1]
        ecg_ppln = setup_class_pipeline[1]
        new_fs = 300 + 50
        ppln = (ecg_ppln.load_labels(os.path.join(path, "REFERENCE.csv"))
                        .augment_fs([("delta", {'loc': new_fs})])
                        .split_to_segments(4500, 4499, pad=True, return_copy=False)
                        .replace_all_labels({"A":"A", "N":"NonA", "O":"NonA"}))

        batch = ppln.next_batch(batch_size=2, shuffle=False)
        assert batch.signal[0].shape == (1, 4500)
        assert batch[batch.indices[0]][2]['diag'] in ["A", "NonA"]



# list_of_methods_to_implement = [
#  'init_parallel',
#  'post_parallel',
#  'check_signal_fs',
#  'check_signal_length'
# ]
