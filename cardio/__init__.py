""" ECG package """
import sys

from .batch import * # pylint: disable=wildcard-import
from . import dataset as ds
from .models import * # pylint: disable=wildcard-import


__version__ = '0.1.0'


if sys.version_info < (3, 5):
    raise ImportError("Dataset module requires Python 3.5 or higher")