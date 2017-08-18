"""Module for testing ecg_batch methods
"""

import os
import sys
import random
from copy import copy
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
    else:
        raise FileNotFoundError("Test files not found in 'tests/data/'!")

    def teardown_module_load():
        '''
        Teardown module
        '''
        # Dump in .npz format is not implemented in EcgBatch.
        # Following section is commented but not deleted 
        # to maintain structure of the test suite.

        # inds = ["A00001", "A00002", "A00004", 
        #         "A00005", "A00008", "A00013"]

        # for ind in inds:
        #     os.remove(os.path.join(path, ind+".npz"))

    request.addfinalizer(teardown_module_load)
    return ind, path

@pytest.fixture()
def setup_class_methods(request):
    '''
    Fixture to setup class to test EcgBatch methods separately.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    ind = ds.FilesIndex(path=os.path.join(path, '*.hea'), no_ext=True, sort=True)
    target_path = os.path.join(path, "REFERENCE.csv")
    batch_loaded = (EcgBatch(ind, unique_labels=["A", "O", "N"])
                    .load(fmt="wfdb", components=["signal", "annotation", "meta"]))

    def teardown_class_methods():
        '''
        Teardown class
        '''

    request.addfinalizer(teardown_class_methods)
    return batch_loaded

@pytest.fixture(scope="class")
def setup_class_dataset(request):
    '''
    Fixture to setup class to test EcgBatch methods in pipeline.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    ind = ds.FilesIndex(path=os.path.join(path, '*.hea'), no_ext=True, sort=True)
    ecg_dataset = ds.Dataset(ind, batch_class=EcgBatch)
    
    def teardown_class_dataset():
        '''
        Teardown class
        '''

    request.addfinalizer(teardown_class_dataset)
    return ecg_dataset

@pytest.fixture(scope="class")
def setup_class_pipeline(request):
    '''
    Fixture to setup class to test EcgBatch methods in pipeline.
    '''
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    ind = ds.FilesIndex(path=os.path.join(path, '*.hea'), no_ext=True, sort=True)
    target_path = os.path.join(path, "REFERENCE.csv")
    ecg_pipeline = (ds.Dataset(ind, batch_class=EcgBatch)
                    .p
                    .load(src=None, fmt="wfdb", components=["signal", "annotation", "meta"])
                    .load(src=target_path, fmt="csv", components=["target"]))

    def teardown_class_pipeline():
        '''
        Teardown class
        '''

    request.addfinalizer(teardown_class_pipeline)
    return ecg_pipeline


class TestEcgBatchLoad():
    '''
    Class for test of load / dump methods
    '''

    def test_load_wfdb(self, setup_module_load): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Test of wfdb loader
        '''
        ind = setup_module_load[0]
        batch = EcgBatch(ind)
        batch = batch.load(fmt="wfdb", components=["signal", "annotation", "meta"])
        assert isinstance(batch.signal, np.ndarray)
        assert isinstance(batch.meta, np.ndarray)
        assert isinstance(batch.annotation, np.ndarray)
        assert batch.signal.shape == (6,)
        assert batch.annotation.shape == (6,)
        assert batch.meta.shape == (6,)
        assert isinstance(batch.signal[0], np.ndarray)
        assert isinstance(batch.annotation[0], dict)
        assert isinstance(batch.meta[0], dict)
        del batch


class TestEcgBatchSingleMethods:
    '''
    Class for testing other single methods of EcgBatch class
    '''
    @pytest.mark.usefixtures("setup_class_methods")
    def test_update(self): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Testing of updater
        '''
        new_inds = ["A00001", "A00004"]
        ind = ds.DatasetIndex(index=new_inds)
        batch = EcgBatch(ind)

        seq1 = np.array([1, 2, 3]).reshape(1, -1)
        seq2 = np.array([1, 2, 3, 4]).reshape(1, -1)
        signal = np.array([seq1, seq2, []], dtype=object)[:-1]
        annotation = np.array([{}] * len(new_inds))
        meta = np.array([{"new_meta":True}, {"new_meta":True}])
        target = np.array(["N","A"])
        batch.update(signal, annotation, meta, target)
        assert batch.signal[0].shape == (1, 3)
        assert batch.signal[1].shape == (1, 4)
        assert batch["A00001"].meta["new_meta"]
        assert batch["A00004"].meta["new_meta"]
        assert batch["A00001"].target == "N"
        assert batch["A00004"].target == "A"

    @pytest.mark.usefixtures("setup_class_methods")
    def tets_drop_short_signal(self, setup_class_methods):
        batch = setup_class_methods
        batch = batch.drop_short_signals(17000, axis=-1)

        assert batch.signal.shape == (2,)
        assert np.all([True if sig.shape[-1]>17000 else False for sig in batch.signal])

    @pytest.mark.usefixtures("setup_class_methods")
    def test_segment_signals(self, setup_class_methods): #pylint: disable=no-self-use,redefined-outer-name
        '''
        Testing of segmentator
        '''
        batch = setup_class_methods
        batch = batch.segment_signals(4500, 4499)
        assert batch.indices.shape == (6,)
        assert batch.signal[0].shape == (2, 1, 4500)

    @pytest.mark.usefixtures("setup_class_methods")
    def test_resample_signals(self, setup_class_methods):
        batch = setup_class_methods
        batch = batch.resample_signals(150)

        assert batch.meta[0]['fs'] == 150
        assert batch.signal[0].shape == (1, 4500)

    @pytest.mark.usefixtures("setup_class_methods")
    def test_band_pass_signals(self, setup_class_methods):
        batch = setup_class_methods
        batch = batch.band_pass_signals(0.5, 5)

        assert batch.signal.shape == (6,)
        assert batch.signal[0].shape == (1, 9000)

    @pytest.mark.usefixtures("setup_class_methods")
    def test_flip_signals(self, setup_class_methods): #pylint:disable=no-self-use,redefined-outer-name
        '''
        Testing function that flips signals if R-peaks are directed downwards
        '''
        batch = setup_class_methods

        batch = batch.flip_signals()
        assert batch.indices.shape == (6,)


@pytest.mark.usefixtures("setup_class_dataset")
class TestEcgBatchDataset:
    ''' Class to test EcgBathc load in pipeline
    '''
    @pytest.mark.xfail
    def test_cv_split(self, setup_class_dataset):
        ecg_dtst = setup_class_dataset
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

    def test_pipeline_load(self, setup_class_dataset, setup_module_load):
        ecg_dtst = setup_class_dataset
        path = setup_module_load[1]
        target_path = os.path.join(path, "REFERENCE.csv")
        
        ppln = (ecg_dtst.p
                .load(fmt="wfdb", components=["signal", "annotation", "meta"])
                .load(src=target_path, fmt="csv", components=["target"]))

        batch_size = 2
        epochs = ecg_dtst.indices.shape[0] // batch_size

        assert epochs == 3

        for i in range(epochs):
            ppln_batch = ppln.next_batch(batch_size)
            assert ppln_batch.indices.shape == (batch_size,)
            assert ppln_batch.signal.shape == (batch_size,)
            assert np.unique(ppln_batch.target).shape == (1,)
            first_indice = ppln_batch.indices[0]

            assert isinstance(ppln_batch[first_indice], ds.components.EcgBatchComponents)



@pytest.mark.usefixtures("setup_class_pipeline")
class TestEcgBatchPipelineMethods:
    ''' Class to test EcgBatch methods in pipeline.
    '''

    def test_pipeline_1(self, setup_class_pipeline):
        ecg_ppln = setup_class_pipeline
        ppln = (ecg_ppln.drop_labels(["A"])
                        .flip_signals()
                        .segment_signals(4500, 4499)
                        .replace_labels({"A":"A", "N":"NonA", "O":"NonA"}))

        batch = ppln.next_batch(len(ppln))

        assert len(batch) == 4
        assert batch.signal[0].shape == (2, 1, 4500)
        assert np.unique(batch.target)[0] == "NonA"

    def test_pipeline_2(self, setup_class_pipeline):
        ecg_ppln = setup_class_pipeline
        ppln = (ecg_ppln.drop_short_signals(17000, axis=-1)
                        .band_pass_signals(0.5,50)
                        .resample_signals(150)
                        .binarize_labels())

        batch = ppln.next_batch(len(ppln))

        assert len(batch) == 2
        assert batch.meta[0]["fs"] == 150
        assert np.all([True if sig.shape[-1]==9000 else False for sig in batch.signal])
        assert batch.target.shape == (2,3)

    def test_base_ppln(self, setup_class_pipeline):
        ecg_ppln = setup_class_pipeline
        new_fs = 300 + 50
        ppln = (ecg_ppln.flip_signals()
                        .segment_signals(4500, 4499)
                        .replace_labels({"A":"A", "N":"NonA", "O":"NonA"}))

        batch = ppln.next_batch(2)
        assert batch.signal[0].shape == (2, 1, 4500)
        assert batch.target[0] in ["A", "NonA"]
