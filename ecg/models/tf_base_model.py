"""Contains tensorflow base model class."""

from .base_model import BaseModel


class TFBaseModel(BaseModel):
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
        return self._sess

    def _create_session(self):
        if self._session is None:
            if self._graph is None:
                raise ValueError("Model graph cannot be empty")
            self._session = tf.Session(graph=self._graph)
            if self._initialize_variables:
                self._session.run(tf.global_variables_initializer())
