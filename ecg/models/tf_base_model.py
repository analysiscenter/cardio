"""Contains tensorflow base model class."""

import tensorflow as tf

from .base_model import BaseModel


class TFBaseModel(BaseModel):  # pylint: disable=abstract-method
    """Base tensorflow model class."""

    def __init__(self):
        self._graph = None
        self._session = None
        self._initialize_variables = True

    @property
    def graph(self):
        """Get tensorflow graph."""
        return self._graph

    @property
    def session(self):
        """Get tensorflow session."""
        return self._session

    def _create_session(self):
        if self._session is None:
            if self._graph is None:
                raise ValueError("Model graph cannot be empty")
            with self.graph.as_default():
                self._session = tf.Session()
                if self._initialize_variables:
                    self._session.run(tf.global_variables_initializer())
