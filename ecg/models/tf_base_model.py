"""Contains tensorflow base model class."""

import tensorflow as tf

from .base_model import BaseModel


class TFBaseModel(BaseModel):  # pylint: disable=abstract-method
    """Base tensorflow model class."""

    def __init__(self):
        self._graph = None
        self._session = None

    @property
    def graph(self):
        """Get tensorflow graph."""
        return self._graph

    @property
    def session(self):
        """Get tensorflow session."""
        return self._session
