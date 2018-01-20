"""Contains kernel generation functions."""

import numpy as np


def _check_kernel_size(size):
    """Check if kernel size is a positive integer."""
    if not isinstance(size, int) or size < 1:
        raise ValueError("Kernel size must be a positive integer")


def gaussian(size, sigma=None):
    """Create a 1-D Gaussian kernel.

    Parameters
    ----------
    size : positive int
        Kernel size.
    sigma : positive float, optional
        Standard deviation of Gaussian distribution. Controls the degree of
        smoothing. If ``None``, it is set to ``(size + 1) / 6``.

    Returns
    -------
    kernel : 1-D ndarray
        Gaussian kernel.

    Raises
    ------
    ValueError
        If ``size`` or ``sigma`` is negative or non-numeric.
    """
    _check_kernel_size(size)
    if sigma is None:
        sigma = (size + 1) / 6
    elif not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError("Sigma must be a positive integer or float")
    i = np.arange(size) - (size - 1) / 2
    kernel = np.exp(-i**2 / (2 * sigma**2))
    return kernel / sum(kernel)
