""" Dataset module implements Dataset, DatasetIndex, Preprocess and Batch classes"""
import sys

from .base import Baseset
from .batch import Batch, ArrayBatch, DataFrameBatch
from .batch_image import ImagesBatch
from .dataset import Dataset
from .jointdataset import JointDataset, FullDataset
from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, parallel, any_action_failed, model


if sys.version_info < (3, 5):
    raise ImportError("Dataset module requires Python 3.5 or higher")
