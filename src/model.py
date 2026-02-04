"""Core simulation routines for the RSE neural field model.

This module implements the model dynamics (activation function, convolution,
stimulus, Euler stepping) and the top-level :func:`run_simulation` helper that
produces time-indexed outputs for visualization.
"""

import numpy as np
import scipy
import timeit
from tqdm import tqdm
from scipy.special import expit

from .params import ModelParams
from .kernels import generate_gaussian_kernel

def firing_rate(x):
    """Compute the sigmoidal firing-rate nonlinearity.

    Parameters
    ----------
    x : array_like
        Input activity.

    Returns
    -------
    array_like
        Firing rate with the same shape as ``x``.
    """
    return expit(x)

def fft_convolution(U, k_fft):
    """Convolve neural field activity with a kernel using FFTs.

    Parameters
    ----------
    U : numpy.ndarray
        Neural field activity matrix.
    k_fft : numpy.ndarray
        Precomputed FFT of the connectivity kernel (same shape as ``U``).

    Returns
    -------
    numpy.ndarray
        Convolved activity matrix.
    """
    # Perform FFT on the inputs
    u_fft = scipy.fft.rfft2(U, axes=(0,1), s=(U.shape))
    u_fft *= k_fft

    return scipy.fft.irfft2(u_fft, axes=(0,1), s=(U.shape)) # Perform element-wise multiplication in Fourier space (frequency domain)

def step_function(x):
    """Compute a Heaviside-like step function.

    Parameters
    ----------
    x : array_like
        Input value(s).

    Returns
    -------
    array_like
        ``1`` where ``x > 0`` and ``0`` otherwise (with broadcasting rules
        matching NumPy).
    """
    return np.maximum(np.sign(x), 0)

def strobe_stimulus(t, A, T, p: ModelParams):
    """Compute a stroboscopic light stimulus.

    Parameters
    ----------
    t : float
        Time in milliseconds.
    A : float
        Stimulus amplitude.
    T : float
        Stimulus period in milliseconds.
    p : ModelParams
        Model parameters. Uses ``p.V`` as the threshold on the sinusoid
        controlling the effective duty cycle (Rule et al. 2011).

    Returns
    -------
    float
        Stimulus value at time ``t``.
    """
    return A*step_function(np.sin((2*np.pi*t)/T)-p.V)

def step(A, T, t, N, Ue, Ui, Ke, Ki, rng: np.random.Generator, noise_field, p: ModelParams):
    """Advance the neural field state by one Euler time step.

    Parameters
    ----------
    A : float
        Stimulus amplitude.
    T : float
        Stimulus period in milliseconds.
    t : float
        Current time in milliseconds.
    N : int
        Neural field side length. ``Ue`` and ``Ui`` are ``(N, N)`` arrays.
    Ue : numpy.ndarray
        Excitatory neural field activity (updated in-place).
    Ui : numpy.ndarray
        Inhibitory neural field activity (updated in-place).
    Ke : numpy.ndarray
        FFT of the excitatory connectivity kernel.
    Ki : numpy.ndarray
        FFT of the inhibitory connectivity kernel.
    rng : numpy.random.Generator
        Random number generator used to sample noise fields.
    noise_field : numpy.ndarray
        Preallocated noise field array.
    p : ModelParams
        Model parameters (e.g., ``dt``, time constants, gains, thresholds, and
        connection strengths).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Updated ``(Ue, Ui)`` arrays.
    """
    
    dtype = Ue.dtype
    
    t3 = timeit.default_timer()
    
    if noise_field is None:
        # one allocation, generates both E/I noise at once
        noise = rng.standard_normal((2, N, N), dtype=dtype)
    else:
        rng.standard_normal(size=noise_field.shape, dtype=noise_field.dtype, out=noise_field)
        noise = noise_field
    
    # Create new noise matrices for each neural field
    noise_E = noise[0]
    noise_I = noise[1]
    
    t4 = timeit.default_timer()
    elapsed_noise_time = t4 - t3
    #print(f"Noise generation took {elapsed_noise_time:.6f} seconds")

    ## Convolve matrices
    Uec = fft_convolution(Ue, Ke)
    Uic = fft_convolution(Ui, Ki)
    
    t5 = timeit.default_timer()
    elapsed_conv_time = t5 - t4
    #print(f"Convolution took {elapsed_conv_time:.6f} seconds")

    ## Strobe Stimulus
    stim = strobe_stimulus(t, A, T, p)

    # Euler's Method of Finding Activity Rate of Change
    dUe = (p.dt/p.Te)*(-Ue + firing_rate(p.Aee*Uec-p.Aie*Uic-p.He+p.Ge*stim+p.Ne*noise_E))
    dUi = (p.dt/p.Ti)*(-Ui + firing_rate(p.Aei*Uec-p.Aii*Uic-p.Hi+p.Gi*stim+p.Ni*noise_I))
    
    # Updating Neural Field Activities
    Ue += dUe
    Ui += dUi
    
    t6 = timeit.default_timer()
    elapsed_euler_time = t6 - t5
    #print(f"Euler step took {elapsed_euler_time:.6f} seconds")
    
    #print(f"Total step time: {t6 - t3:.6f} seconds")
    
    return Ue, Ui

def run_simulation(N, A, T, Se, Si, start_time, end_time, seed, plot, gif, interval, p: ModelParams, fps=50, dtype=np.float32):
    """Run the neural field simulation.

    Parameters
    ----------
    N : int
        Neural field side length. The simulation state has shape ``(N, N)``.
    A : float
        Stimulus amplitude.
    T : float
        Stimulus period in milliseconds.
    Se : float
        Standard deviation of the excitatory connectivity kernel.
    Si : float
        Standard deviation of the inhibitory connectivity kernel.
    start_time : int
        Start time in milliseconds.
    end_time : int
        End time in milliseconds.
    seed : int or None
        Random seed for reproducibility.
    plot : bool
        Whether to generate plot.
    gif : bool
        Whether to record frames for GIF output.
    interval : int
        Duration in milliseconds between recorded snapshots in ``plots['images']``.
    p : ModelParams
        Model parameters, including ``dt``.
    fps : int, default=50
        Frames per second for GIF sampling.

    Returns
    -------
    dict
        Dictionary with keys ``'gif'`` and ``'images'``. Each maps simulation
        times (float, in ms) to a dict of arrays/series used for visualization.
    """
    
    # Initialize Random Number Generator
    rng = np.random.default_rng(seed)

    # Initializing Random Activity Rates (Uniform Distribution between 0 and 1)
    Ue = rng.random((N, N)).astype(dtype, copy=False)
    Ui = rng.random((N, N)).astype(dtype, copy=False)
    
    # Connectivity Kernels
    Ke = generate_gaussian_kernel(Se, N).astype(dtype, copy=False)
    Ki = generate_gaussian_kernel(Si, N).astype(dtype, copy=False)
    
    # Precompute FFT of kernels
    Ke = scipy.fft.rfft2(Ke, s=(Ue.shape))
    Ki = scipy.fft.rfft2(Ki, s=(Ui.shape))

    # Time interval to record activity for plots
    plotting_range = range(start_time, end_time + 1, 1)
    
    # Preallocate noise
    noise_field = np.empty((2, N, N), dtype=dtype)

    # Initialize list for saving activity at a point (for plotting)
    pointE = []
    pointI = []

    # Initialize list for saving time (for plotting)
    time = []

    # Initialize list for saving strobe stimulation (for plotting)
    StimE = []
    StimI = []
    
    # Dictionary to hold plots
    plots = {"gif": {}, "images": {}}
    
    # Time step from model parameters
    dt = p.dt
    
    # Determine number of steps for plotting and gif saving
    plot_every_steps = int(round(interval / dt))
    gif_every_steps = int(round((1000 / fps) / dt))

    # Total number of simulation steps
    steps = int(round(end_time / dt))

    # Main Simulation Loop
    for step_idx in tqdm(range(steps + 1)):
        
        # Current time (ms)
        t = step_idx * dt
        
        t1 = timeit.default_timer()

        # Update Neural Field Activities
        Ue, Ui = step(A, T, t, N, Ue, Ui, Ke, Ki, rng, noise_field, p)
        
        t2 = timeit.default_timer()
        elapsed_time = t2 - t1
        #print(f"Step took {elapsed_time:.6f} seconds")
        
        if plot:
            # Saving Point Activities (2,2 was chosen arbitrarily)
            pointE.append(float(Ue[2, 2]))
            pointI.append(float(Ui[2, 2]))

            # Saving Time
            time.append(t)

            # Saving Strobe Stimulation
            StimE.append(p.Ge * strobe_stimulus(t, A, T, p))
            StimI.append(p.Gi * strobe_stimulus(t, A, T, p))
        
        # Check if time step is in plotting range and at specified interval 
        # to save plots
        if step_idx != 0 and \
            step_idx % plot_every_steps == 0 and \
            (start_time <= t <= end_time):
                
            # Calculate Cortical Activity
            cortical_activity = np.abs(Ue - Ui)
            
            plots["images"][t] = {
                "t": t,
                "cortical_activity": cortical_activity.copy(), # NumPy array copy
                "time": list(time),  
                "pointE": list(pointE),
                "pointI": list(pointI),
                "StimE": list(StimE),
                "StimI": list(StimI),
            }
        
        # Save frames for GIF at specified fps
        if gif and \
            step_idx % gif_every_steps == 0 and \
            int(t) in plotting_range:
                
                    # Calculate Cortical Activity
            cortical_activity = np.abs(Ue - Ui)

            plots["gif"][t] = {
                "t": t,
                "cortical_activity": cortical_activity.copy(), # NumPy array copy
                "time": list(time)
            }
    
    return plots