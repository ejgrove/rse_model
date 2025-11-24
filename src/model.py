import numpy as np
from tqdm import tqdm

from .params import ModelParams
from .kernels import generate_gaussian_kernel

### Nonlinear (Sigmoidal)Firing Rate

def F(x):
  return 1 / (1 + np.exp(-x))

## FFT Convolution
def fft_convolution(U, k_fft):
    
    # Perform FFT on the inputs
    u_fft = np.fft.fft2(np.array(U), axes=(0,1)) 
    u_fft *= k_fft

    return np.real(np.fft.ifft2(u_fft, axes=(0,1))) # Perform element-wise multiplication in Fourier space (frequency domain)

## Step Function
def H(x):
    return np.maximum(np.sign(x), 0)

## Flicker Stimulus Function
def S(t, A, T, p: ModelParams):
    return A*H(np.sin((2*np.pi*t)/T)-p.V)

def step(A, T, t, N, Ue, Ui, Ke, Ki, rng: np.random.Generator, p: ModelParams):
    # Gaussian Noise
    noise_E = rng.normal(loc=0.0, scale=1.0, size=(N, N))
    noise_I = rng.normal(loc=0.0, scale=1.0, size=(N, N))

    ## Convolve matrix
    Uec = fft_convolution(Ue, Ke)
    Uic = fft_convolution(Ui, Ki)

    # Euler's Method of Finding Activity Rate of Change
    dUe = (p.dt/p.Te)*(-Ue + F(p.Aee*Uec-p.Aie*Uic-p.He+p.Ge*S(t*p.dt, A, T, p)+p.Ne*noise_E))
    dUi = (p.dt/p.Ti)*(-Ui + F(p.Aei*Uec-p.Aii*Uic-p.Hi+p.Gi*S(t*p.dt, A, T, p)+p.Ni*noise_I))

    # Updating Neural Field Activities
    Ue += dUe
    Ui += dUi
    
    return Ue, Ui

def run_simulation(N, A, T, Se, Si, start_time, end_time, seed, gif, interval, p: ModelParams, fps=50):
    
    rng = np.random.default_rng(seed)

    # Initializing Random Activity Rates (Uniform)
    Ue = rng.random((N, N))
    Ui = rng.random((N, N))

    # Connectivity Kernels
    K_sidelength_E = N
    K_sidelength_I = N

    Ke = generate_gaussian_kernel(Se, K_sidelength_E)
    Ki = generate_gaussian_kernel(Si, K_sidelength_I)
    
    Ke = np.fft.fft2(np.array(Ke), axes=(0,1)) # Precompute FFT of excitatory kernel
    Ki = np.fft.fft2(np.array(Ki), axes=(0,1))

    # Time interval to record activity (1 - every ms)
    plotting_range = range(start_time, end_time + 1, 1)

    # Activity at a random point (for plotting)
    pointE = []
    pointI = []

    # Time (for plotting)
    time = []

    # Stimulation (for plotting)
    StimE = []
    StimI = []
    
    plots = {"gif": {}, "images": {}}

    # Time Steps
    steps = int((end_time + p.dt)/p.dt)
    
    for t in tqdm(range(steps)):

        Ue, Ui = step(A, T, t, N, Ue, Ui, Ke, Ki, rng, p)
        
        cortical_activity = np.abs(Ue - Ui)

        # Appending Point Activities
        pointE_int = Ue[2,2] # choosing random point at 2,2 to view activity
        pointI_int = Ui[2,2]
        pointE = np.append(pointE, [pointE_int])
        pointI = np.append(pointI, [pointI_int])

        # Appending Time
        time.append(t * p.dt)

        # Appending Stimulation
        StimE.append(p.Ge * S(t * p.dt, A, T, p))
        StimI.append(p.Gi * S(t * p.dt, A, T, p))

        # Show plot at intervals
        if t*p.dt != 0 and \
            t*p.dt % interval == 0 and \
            t*p.dt in plotting_range:
            
            plots["images"][t*p.dt] = {
                                        "t": t,
                                        "cortical_activity": cortical_activity.copy(), # NumPy array copy
                                        "time": list(time),     # or time.copy()
                                        "pointE": list(pointE),
                                        "pointI": list(pointI),
                                        "StimE": list(StimE),
                                        "StimI": list(StimI),
            }
            
        if gif and t*p.dt % (1000/fps) == 0 and \
            t*p.dt in plotting_range:
        
            plots["gif"][t*p.dt] = {
                                    "t": t,
                                    "cortical_activity": cortical_activity.copy(), # NumPy array copy
                                    "time": list(time)
            }
            
    return plots