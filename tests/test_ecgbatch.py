"""Module for testing ecg_batch methods
"""

import os
import sys
from copy import deepcopy

import numpy as np
import pytest

sys.path.append(os.path.join("."))

from batch import EcgBatch
from batch import ds

@pytest.fixture(scope="module")
def setup_module_load(request):
    '''
    Fixture to setup module. Performs check for presence of test files,
    creates initial batch object.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    files = ["A00001.hea", "A00004.hea", "A00001.mat", "A00004.mat", "REFERENCE.csv"]
    # TODO: make better test for presence of files .hea and
    # REFERENCE.csv
    
    print(sys.path, file=sys.stderr)

    if np.all([os.path.isfile(os.path.join(path, file)) for file in files]):
        ind = ds.FilesIndex(path=os.path.join(path, '*.hea'), no_ext=True, sort=True)
        batch_init = EcgBatch(ind)
    else:
        raise FileNotFoundError("Test files not found in 'tests/data/'!")

    def teardown_module_load():
        '''
        Teardown module
        ''' 
        os.remove(os.path.join(path, "A00001.npz"))
        os.remove(os.path.join(path, "A00004.npz"))

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
    pass
    def teardown_class_pipeline():
        '''
        Teardown class
        '''
    request.addfinalizer(teardown_class_pipeline)
    return pipeline_loaded


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
        assert batch.signal.shape == (2, )
        assert len(batch.meta) == 2
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
        assert batch.signal.shape == (2, )
        assert batch.signal[0].shape == (1, 9000)
        assert len(batch.meta) == 2
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
        assert batch.indices.shape == (2,)
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
        else:
            raise ValueError("No key 'diag' in meta!")

    def test_split_to_segments(self, setup_class_methods): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Testing of segmentator
        '''
        batch = deepcopy(setup_class_methods)
        batch = batch.split_to_segments(
            4500, 4499, pad=True, return_copy=False)
        assert batch.indices.shape == (4, )
        assert batch.signal[0].shape == (1, 4500)

    def test_replace_all_labels(self, setup_class_methods, setup_module_load): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Testing of label replacer
        '''
        batch = deepcopy(setup_class_methods)
        path = setup_module_load[1]
        batch = batch.load_labels(os.path.join(path, "REFERENCE.csv"))
        batch = batch.replace_all_labels({"A":"A", "N":"NonA"})
        assert batch["A00001"][2]['diag'] == "NonA"
        assert batch["A00004"][2]['diag'] == "A"

    def test_update(self, setup_class_methods): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Testing of updater
        '''
        batch = deepcopy(setup_class_methods)
        seq1 = np.array([1, 2, 3]).reshape(1, -1)
        seq2 = np.array([1, 2, 3, 4]).reshape(1, -1)
        data = np.array([seq1, seq2, []], dtype=object)[:-1]
        annotation = None
        meta = dict(zip(["A00001", "A00004"], [{"new_meta":True}, {"new_meta":True}]))
        batch.update(data, annotation, meta)
        assert batch["A00001"][0].shape == (1, 3)
        assert batch["A00004"][0].shape == (1, 4)
        assert batch["A00001"][2]["new_meta"]
        assert batch["A00004"][2]["new_meta"]

# list_of_methods_to_implement = [
#  'init_parallel',
#  'post_parallel',
#  'check_signal_fs',
#  'check_signal_length'
# ]
