import numpy as np

## Gaussian Kernel
def K(x, y, S):
    ''' This function is not exactly the square of the gaussian (exponent should be
    multiplied by 1/2 in real square) but Ermentrout said to use this...
    Rule doesn't know why'''

    norm_factor = 1 / (np.power(S,2) * np.pi)
    exponent = - (np.power(np.abs(x), 2) + np.power(np.abs(y), 2)) / (np.power(S, 2))

    return norm_factor * np.exp(exponent)

def generate_gaussian_kernel(sigma, size):
    x = np.arange(-size//2, size//2, 1)
    y = np.arange(size//2, -size//2, -1)
    X, Y = np.meshgrid(x, y)
    kernel = K(X, Y, sigma)
    
    # Apply fft
    np.fft.fft2(np.array(kernel), axes=(0,1))

    return kernel