"""Miscellaneous ecg batch utils."""

import numpy as np
import sklearn


class LabelBinarizer(sklearn.preprocessing.LabelBinarizer):
    """Encode categorical features using a one-hot scheme."""
    # pylint: disable=invalid-name

    def transform(self, y):
        """Transform y using one-hot encoding.

        Parameters
        ----------
        y : 1-D ndarray of shape [n_samples,]
            Class labels.

        Returns
        -------
        Y : 2-D array of shape [n_samples, n_classes]
            One-hot encoded labels.
        """
        Y = super().transform(y)
        if len(self.classes_) == 1:
            Y = 1 - Y  # pylint: disable=redefined-variable-type
        if len(self.classes_) == 2:
            Y = np.hstack((1 - Y, Y))
        return Y

    def inverse_transform(self, Y, threshold=None):
        """Transform one-hot encoded labels back to class labels.

        Parameters
        ----------
        Y : 2-D array of shape [n_samples, n_classes]
            One-hot encoded labels.
        threshold : float or None
            Threshold used in the binary and multi-label cases.

        Returns
        -------
        y : 1-D ndarray of shape [n_samples,]
            Class labels.
        """
        if len(self.classes_) == 1:
            y = super().inverse_transform(1 - Y, threshold)
        elif len(self.classes_) == 2:
            y = super().inverse_transform(Y[:, 1], threshold)
        else:
            y = super().inverse_transform(Y, threshold)
        return y
