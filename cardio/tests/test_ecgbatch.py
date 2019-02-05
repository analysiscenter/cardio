"""Module for testing ecg_batch methods"""

import os
import random
import numpy as np
import pytest

from cardio import EcgBatch, EcgDataset, batchflow as bf
from cardio.core import ecg_batch_tools as bt

random.seed(170720143422)

@pytest.fixture(scope="module")
def setup_module_load(request):
    """
    Fixture to setup module. Performs check for presence of test files,
    creates initial batch object.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    files = ["A00001.hea", "A00001.mat", "A00002.hea", "A00002.mat",
             "A00004.hea", "A00004.mat", "A00005.hea", "A00005.mat",
             "A00008.hea", "A00008.mat", "A00013.hea", "A00013.mat", "REFERENCE.csv"]
    # TODO: make better test for presence of files .hea and
    # REFERENCE.csv

    if np.all([os.path.isfile(os.path.join(path, file)) for file in files]):
        ind = bf.FilesIndex(path=os.path.join(path, 'A*.hea'), no_ext=True, sort=True)
    else:
        raise FileNotFoundError("Test files not found in 'tests/data/'!")

    def teardown_module_load():
        """
        Teardown module
        """

    request.addfinalizer(teardown_module_load)
    return ind, path

@pytest.fixture()
def setup_class_methods(request):
    """
    Fixture to setup class to test EcgBatch methods separately.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    ind = bf.FilesIndex(path=os.path.join(path, 'A*.hea'), no_ext=True, sort=True)
    batch_loaded = (EcgBatch(ind, unique_labels=["A", "O", "N"])
                    .load(fmt="wfdb", components=["signal", "annotation", "meta"]))

    def teardown_class_methods():
        """
        Teardown class
        """

    request.addfinalizer(teardown_class_methods)
    return batch_loaded

@pytest.fixture(scope="class")
def setup_class_dataset(request):
    """
    Fixture to setup class to test EcgBatch methods in pipeline.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    ecg_dataset = EcgDataset(path=os.path.join(path, 'A*.hea'), no_ext=True, sort=True)

    def teardown_class_dataset():
        """
        Teardown class
        """

    request.addfinalizer(teardown_class_dataset)
    return ecg_dataset

@pytest.fixture(scope="class")
def setup_class_pipeline(request):
    """
    Fixture to setup class to test EcgBatch methods in pipeline.
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')
    target_path = os.path.join(path, "REFERENCE.csv")
    ecg_pipeline = (EcgDataset(path=os.path.join(path, 'A*.hea'), no_ext=True, sort=True)
                    .p
                    .load(src=None, fmt="wfdb", components=["signal", "annotation", "meta"])
                    .load(src=target_path, fmt="csv", components=["target"]))

    def teardown_class_pipeline():
        """
        Teardown class
        """

    request.addfinalizer(teardown_class_pipeline)
    return ecg_pipeline


class TestEcgBatchLoad():
    """
    Class for test of load / dump methods.
    """

    def test_load_wfdb(self, setup_module_load): #pylint: disable=redefined-outer-name
        """
        Testing wfdb loader.
        """
        # Arrange
        ind = setup_module_load[0]
        batch = EcgBatch(ind)
        # Act
        batch = batch.load(fmt="wfdb", components=["signal", "annotation", "meta"])
        # Assert
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

    def test_load_wfdb_annotation(self, setup_module_load): #pylint: disable=redefined-outer-name
        """
        Testing wfdb loader for annotation.
        """
        # Arrange
        path = setup_module_load[1]
        ind = bf.FilesIndex(path=os.path.join(path, 'sel100.hea'), no_ext=True, sort=True)
        batch = EcgBatch(ind)
        # Act
        batch = batch.load(fmt="wfdb", components=["signal", "annotation", "meta"], ann_ext="pu1")
        # Assert
        assert isinstance(batch.signal, np.ndarray)
        assert isinstance(batch.meta, np.ndarray)
        assert isinstance(batch.annotation, np.ndarray)
        assert batch.signal.shape == (1,)
        assert batch.annotation.shape == (1,)
        assert batch.meta.shape == (1,)
        assert isinstance(batch.signal[0], np.ndarray)
        assert isinstance(batch.annotation[0], dict)
        assert isinstance(batch.meta[0], dict)
        assert 'annsamp' in batch.annotation[0]
        assert 'anntype' in batch.annotation[0]
        del batch

    def test_load_dicom(self, setup_module_load): #pylint: disable=redefined-outer-name
        """
        Testing DICOM loader.
        """
        # Arrange
        path = setup_module_load[1]
        ind = bf.FilesIndex(path=os.path.join(path, 'sample*.dcm'), no_ext=True, sort=True)
        batch = EcgBatch(ind)
        # Act
        batch = batch.load(fmt="dicom", components=["signal", "annotation", "meta"])
        # Assert
        assert isinstance(batch.signal, np.ndarray)
        assert isinstance(batch.meta, np.ndarray)
        assert isinstance(batch.annotation, np.ndarray)
        assert batch.signal.shape == (1,)
        assert batch.annotation.shape == (1,)
        assert batch.meta.shape == (1,)
        assert isinstance(batch.signal[0], np.ndarray)
        assert isinstance(batch.annotation[0], dict)
        assert isinstance(batch.meta[0], dict)
        del batch

    def test_load_edf(self, setup_module_load): #pylint: disable=redefined-outer-name
        """
        Testing EDF loader.
        """
        # Arrange
        path = setup_module_load[1]
        ind = bf.FilesIndex(path=os.path.join(path, 'sample*.edf'), no_ext=True, sort=True)
        batch = EcgBatch(ind)
        # Act
        batch = batch.load(fmt="edf", components=["signal", "annotation", "meta"])
        # Assert
        assert isinstance(batch.signal, np.ndarray)
        assert isinstance(batch.meta, np.ndarray)
        assert isinstance(batch.annotation, np.ndarray)
        assert batch.signal.shape == (1,)
        assert batch.annotation.shape == (1,)
        assert batch.meta.shape == (1,)
        assert isinstance(batch.signal[0], np.ndarray)
        assert isinstance(batch.annotation[0], dict)
        assert isinstance(batch.meta[0], dict)
        del batch

    def test_load_wav(self, setup_module_load): #pylint: disable=redefined-outer-name
        """
        Testing EDF loader.
        """
        # Arrange
        path = setup_module_load[1]
        ind = bf.FilesIndex(path=os.path.join(path, 'sample*.wav'), no_ext=True, sort=True)
        batch = EcgBatch(ind)
        # Act
        batch = batch.load(fmt="wav", components=["signal", "annotation", "meta"])
        # Assert
        assert isinstance(batch.signal, np.ndarray)
        assert isinstance(batch.meta, np.ndarray)
        assert isinstance(batch.annotation, np.ndarray)
        assert batch.signal.shape == (1,)
        assert batch.annotation.shape == (1,)
        assert batch.meta.shape == (1,)
        assert isinstance(batch.signal[0], np.ndarray)
        assert isinstance(batch.annotation[0], dict)
        assert isinstance(batch.meta[0], dict)
        del batch

class TestEcgBatchSingleMethods:
    """
    Class for testing other single methods of EcgBatch class
    """
    @pytest.mark.usefixtures("setup_class_methods")
    def tets_drop_short_signal(self, setup_class_methods): #pylint: disable=redefined-outer-name
        """
        Testing of drop_short_signals
        """

        # Arrange
        batch = setup_class_methods

        # Act
        batch = batch.drop_short_signals(17000, axis=-1)

        # Assert
        assert batch.signal.shape == (2,)
        assert np.all([(sig.shape[-1] > 17000) for sig in batch.signal])

    @pytest.mark.usefixtures("setup_class_methods")
    def test_split_signals(self, setup_class_methods): #pylint: disable=redefined-outer-name
        """
        Testing split_signals.
        """
        batch = setup_class_methods
        batch = batch.split_signals(4500, 4499)
        assert batch.indices.shape == (6,)
        assert batch.signal[0].shape == (2, 1, 4500)

    @pytest.mark.usefixtures("setup_class_methods")
    def test_resample_signals(self, setup_class_methods):#pylint: disable=redefined-outer-name
        """
        Testing resample_signal.
        """
        batch = setup_class_methods
        batch = batch.resample_signals(150)

        assert batch.meta[0]['fs'] == 150
        assert batch.signal[0].shape == (1, 4500)

    @pytest.mark.usefixtures("setup_class_methods")
    def test_band_pass_signals(self, setup_class_methods): #pylint: disable=redefined-outer-name
        """
        Testing band_pass_signals.
        """
        batch = setup_class_methods
        batch = batch.band_pass_signals(0.5, 5)

        assert batch.signal.shape == (6,)
        assert batch.signal[0].shape == (1, 9000)

    @pytest.mark.usefixtures("setup_class_methods")
    def test_flip_signals(self, setup_class_methods): #pylint: disable=redefined-outer-name
        """
        Testing flip_signals.
        """
        batch = setup_class_methods

        batch = batch.flip_signals()
        assert batch.indices.shape == (6,)


@pytest.mark.usefixtures("setup_class_dataset")
class TestEcgBatchDataset:
    """
    Class to test EcgBathc load in pipeline
    """
    @pytest.mark.xfail
    def test_cv_split(self, setup_class_dataset): #pylint: disable=redefined-outer-name
        """
        Testing cv_split.
        """
        ecg_dtst = setup_class_dataset
        ecg_dtst.cv_split(shares=0.5)
        assert ecg_dtst.train.indices.shape == (3,)
        assert ecg_dtst.test.indices.shape == (3,)
        assert isinstance(ecg_dtst.train, bf.Dataset)
        assert isinstance(ecg_dtst.test, bf.Dataset)

        ecg_dtst.cv_split(shares=[0.5, 0.49])
        assert ecg_dtst.train.indices.shape == (3,)
        assert ecg_dtst.test.indices.shape == (2,)
        assert ecg_dtst.validation.indices.shape == (1,)
        assert isinstance(ecg_dtst.train, bf.Dataset)
        assert isinstance(ecg_dtst.test, bf.Dataset)
        assert isinstance(ecg_dtst.validation, bf.Dataset)

    def test_pipeline_load(self, setup_class_dataset, setup_module_load): #pylint: disable=redefined-outer-name
        """
        Testing load in pipeline.
        """
        ecg_dtst = setup_class_dataset
        path = setup_module_load[1]
        target_path = os.path.join(path, "REFERENCE.csv")

        ppln = (ecg_dtst.p
                .load(fmt="wfdb", components=["signal", "annotation", "meta"])
                .load(src=target_path, fmt="csv", components=["target"]))

        batch_size = 2
        epochs = ecg_dtst.indices.shape[0] // batch_size

        assert epochs == 3

        for _ in range(epochs):
            ppln_batch = ppln.next_batch(batch_size, shuffle=False)
            assert ppln_batch.indices.shape == (batch_size,)
            assert ppln_batch.signal.shape == (batch_size,)
            assert np.unique(ppln_batch.target).shape == (1,)

            first_indice = ppln_batch.indices[0]
            assert isinstance(ppln_batch[first_indice], bf.components.EcgBatchComponents) #pylint: disable=no-member



@pytest.mark.usefixtures("setup_class_pipeline")
class TestEcgBatchPipelineMethods:
    """
    Class to test EcgBatch methods in pipeline.
    """

    def test_pipeline_1(self, setup_class_pipeline): #pylint: disable=redefined-outer-name
        """
        Testing typical pipeline.
        """
        ecg_ppln = setup_class_pipeline
        ppln = (ecg_ppln
                .drop_labels(["A"])
                .flip_signals()
                .split_signals(4500, 4499)
                .rename_labels({"A":"A", "N":"NonA", "O":"NonA"}))

        batch = ppln.next_batch(len(ppln), shuffle=False)
        assert len(batch) == 4
        assert batch.signal[0].shape == (2, 1, 4500)
        assert np.unique(batch.target)[0] == "NonA"

    def test_pipeline_2(self, setup_class_pipeline): #pylint: disable=redefined-outer-name
        """
        Testing typical pipeline.
        """
        ecg_ppln = setup_class_pipeline
        ppln = (ecg_ppln
                .drop_short_signals(17000, axis=-1)
                .band_pass_signals(0.5, 50)
                .resample_signals(150)
                .binarize_labels())

        batch = ppln.next_batch(len(ppln), shuffle=False)

        assert len(batch) == 2
        assert batch.meta[0]["fs"] == 150
        assert np.all([(sig.shape[-1] == 9000) for sig in batch.signal])
        assert batch.target.shape == (2, 3)

    def test_get_signal_with_meta(self, setup_module_load): #pylint: disable=redefined-outer-name
        """
        Testing get_signal_meta.
        """
        # Arrange
        ppln = (bf.Pipeline()
                .init_variable(name="signal", init_on_each_run=list)
                .load(fmt='wfdb', components=["signal", "meta"])
                .flip_signals()
                .update_variable("signal", bf.B("signal"), mode='a')
                .run(batch_size=2, shuffle=False,
                     drop_last=False, n_epochs=1, lazy=True))

        dtst = EcgDataset(setup_module_load[0])

        # Act
        ppln_run = (dtst >> ppln).run()
        signal_var = ppln_run.get_variable("signal")

        # Assert
        assert len(signal_var) == 3
        assert len(signal_var[0]) == 2
        assert signal_var[0][0].shape == (1, 9000)

class TestIntervalBatchTools:
    """
    Class for testing batch tools that involved in intervals calculation
    """
    def test_find_interval_borders(self): #pylint: disable=redefined-outer-name
        """
        Testing find_interval_borders.
        """
        # Arrange
        hmm_annotation = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 0,
                                   0, 0, 4, 4, 4, 0, 0, 0, 1, 1,
                                   1, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                   4, 4, 4, 4, 0, 0, 0, 1, 1, 1]*2,
                                  dtype=np.int64)

        # Act
        starts_3, ends_3 = bt.find_intervals_borders(hmm_annotation,
                                                     np.array([3], dtype=np.int64))

        starts_12, ends_12 = bt.find_intervals_borders(hmm_annotation,
                                                       np.array([1, 2], dtype=np.int64))

        # Assert
        assert np.all(np.equal(starts_3, np.array([6, 25, 46, 65])))
        assert np.all(np.equal(ends_3, np.array([9, 30, 49, 70])))

        assert np.all(np.equal(starts_12, np.array([18, 37, 58])))
        assert np.all(np.equal(ends_12, np.array([25, 46, 65])))

    def test_find_maxes(self): #pylint: disable=redefined-outer-name
        """
        Testing find_maxes.
        """
        # Arrange
        hmm_annotation = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 0,
                                   0, 0, 4, 4, 4, 0, 0, 0, 1, 1,
                                   1, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                   4, 4, 4, 4, 0, 0, 0, 1, 1, 1]*2,
                                  dtype=np.int64)

        # 9 is the max for interval of 3's
        # 8 is the max for interval of 1's and 2's
        signal = np.array([1, 1, 1, 2, 2, 8, 3, 9, 3, 0,
                           0, 0, 4, 4, 4, 0, 0, 0, 1, 8,
                           1, 2, 2, 2, 2, 9, 3, 3, 3, 3,
                           4, 4, 4, 4, 0, 0, 0, 1, 1, 1]*2,
                          dtype=np.float64).reshape(1, -1)

        # Act
        starts_3, ends_3 = bt.find_intervals_borders(hmm_annotation,
                                                     np.array([3], dtype=np.int64))

        starts_12, ends_12 = bt.find_intervals_borders(hmm_annotation,
                                                       np.array([1, 2], dtype=np.int64))

        maxes_3 = bt.find_maxes(signal, starts_3, ends_3)
        maxes_12 = bt.find_maxes(signal, starts_12, ends_12)

        # Assert
        assert np.all(np.equal(maxes_3, np.array([7, 25, 47, 65])))
        assert np.all(np.equal(maxes_12, np.array([19, 45, 59])))

    def test_calc_hr(self):
        """
        Testing calc_hr.
        """
        # Arrange
        fs = 18.0

        R_STATE = np.array([3], dtype=np.int64) # pylint: disable=invalid-name

        hmm_annotation = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 0,
                                   0, 0, 4, 4, 4, 0, 0, 0, 1, 1,
                                   1, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                   4, 4, 4, 4, 0, 0, 0, 1, 1, 1]*2,
                                  dtype=np.int64)

        # 9 is the max for interval of 3's
        # 8 is the max for interval of 1's and 2's
        signal = np.array([1, 1, 1, 2, 2, 8, 3, 9, 3, 0,
                           0, 0, 4, 4, 4, 0, 0, 0, 1, 8,
                           1, 2, 2, 2, 2, 9, 3, 3, 3, 3,
                           4, 4, 4, 4, 0, 0, 0, 1, 1, 1]*2,
                          dtype=np.float64).reshape(1, -1)

        # Act
        hr = bt.calc_hr(signal, hmm_annotation, fs, R_STATE) # pylint: disable=invalid-name

        # Arrange
        assert hr == 60
