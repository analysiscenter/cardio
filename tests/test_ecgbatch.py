import pytest
import dataset as ds
import os
from ecg_batch2 import *
from copy import deepcopy


@pytest.fixture(scope="module")
def setup_module_load(request):
    print("\nModule setup")
    PATH = 'data/'
    files = os.listdir(PATH)
    # TODO: make better test for presence of files .hea and 
    # REFERENCE.csv
    if ("A00001.mat" in files) and ("A00004.mat" in files):
        PATH = 'data/'
        ind = ds.FilesIndex(path=PATH + '*.hea', no_ext=True, sort=True)
        batch_init = EcgBatch(ind)
    else:
        raise ValueError('Something wrong with test data!')

    def teardown_module_load():
        print("\nModule teardown")
        os.remove(PATH + "A00001.npz")
        os.remove(PATH + "A00004.npz")

    request.addfinalizer(teardown_module_load)
    return batch_init, PATH


@pytest.fixture(scope="class")
def setup_class_methods(request):
    print("\nClass setup")
    PATH = 'data/'
    ind = ds.FilesIndex(path=PATH + '*.hea', no_ext=True, sort=True)
    batch_loaded = EcgBatch(ind).load(src=None, fmt="wfdb")

    def teardown_class_methods():
        print("\nClass teardown")

    request.addfinalizer(teardown_class_methods)
    return batch_loaded


class TestEcgBatchLoad():
    '''
    '''

    def test_load_wfdb(self, setup_module_load):
        batch = deepcopy(setup_module_load[0])
        batch = batch.load(src=None, fmt='wfdb')
        assert isinstance(batch.signal, np.ndarray)
        assert isinstance(batch.meta, dict)
        assert isinstance(batch.annotation, dict)
        assert batch.signal.shape == (2, )
        assert len(batch.annotation) == 2
        assert len(batch.meta) == 2
        assert len(batch[batch.indices[0]]) == 3
        del batch

    def test_dump(self, setup_module_load):
        batch = deepcopy(setup_module_load[0])
        PATH = setup_module_load[1]
        batch = batch.load(src=None, fmt='wfdb')
        batch = batch.dump(dst=PATH, fmt='npz')
        files = os.listdir(PATH)
        assert ("A00001.npz" in files)
        assert ("A00004.npz" in files)
        del batch

    def test_load_npz(self, setup_module_load):
        PATH = setup_module_load[1]
        ind = ds.FilesIndex(path=PATH + '*.npz', no_ext=True, sort=True)
        batch = EcgBatch(ind)
        batch = batch.load(src=None, fmt='npz')
        assert isinstance(batch.signal, np.ndarray)
        assert isinstance(batch.meta, dict)
        assert isinstance(batch.annotation, dict)
        assert batch.signal.shape == (2, )
        assert batch.signal[0].shape == (1, 9000)
        assert len(batch.annotation) == 2
        assert len(batch.meta) == 2
        assert len(batch[batch.indices[0]]) == 3
        del batch


@pytest.mark.usefixtures("setup_class_methods")
class TestEcgBatch_single_methods:
    '''
    '''

    def test_load_labels(self, setup_module_load, setup_class_methods):
        batch = deepcopy(setup_class_methods)
        PATH = setup_module_load[1]
        batch = batch.load(
            fmt="wfdb", src=None).load_labels(PATH + "REFERENCE.csv")
        if "diag" in batch["A00001"][2].keys():
            assert batch["A00001"][2]['diag'] == 'N'
            assert batch["A00004"][2]['diag'] == 'A'
        else:
            raise ValueError("No key 'diag' in meta!")

    def test_split_to_segments(self, setup_class_methods):
        batch = deepcopy(setup_class_methods)
        batch = batch.split_to_segments(
            4500, 4499, pad=True, return_copy=False)
        assert batch.indices.shape == (4, )
        assert batch.signal[0].shape == (1, 4500)

    #def test_augment_fs(self, setup_class_methods):
    #    batch = setup_class_methods[0]
    #    batch = batch.augment_fs([("delta", {})])
