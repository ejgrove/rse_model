"""Connectivity kernels used by the RSE model.
"""

import numpy as np

## Gaussian Kernel
def gaussian_kernel_2d(x, y, sigma):
    """Produces values for a 2D Gaussian-like kernel centered at (0,0).

    Notes
    -----
    This implementation follows the formula used in the
    referenced model code. It is not the exact square of a Gaussian (the
    exponent is not multiplied by 1/2 as would be typical for a squared
    Gaussian).

    Parameters
    ----------
    x : array
        X-coordinates.
    y : array
        Y-coordinates.
    sigma : float
        Standard deviation parameter.

    Returns
    -------
    numpy.ndarray
        Kernel evaluated at (x, y).
    """

    norm_factor = 1 / (np.power(sigma,2) * np.pi)
    exponent = - (np.power(np.abs(x), 2) + np.power(np.abs(y), 2)) / (np.power(sigma, 2))

    return norm_factor * np.exp(exponent)

def generate_gaussian_kernel(sigma, N):
    """Generate a 2D Gaussian-like kernel over a square grid.

    Parameters
    ----------
    sigma : float
        Standard deviation parameter.
    N : int
        Kernel size. The output has shape (N, N).

    Returns
    -------
    numpy.ndarray
        The generated kernel.
    """
    # N values must be odd integers to center the kernel
    x = np.arange(-(N//2), (N//2) + 1, 1)
    y = np.arange(-(N//2), (N//2) + 1, 1)
    X, Y = np.meshgrid(x, y)
    kernel = gaussian_kernel_2d(X, Y, sigma)
    
    # Apply fft
    np.fft.fft2(np.array(kernel), axes=(0,1))

    return kernel