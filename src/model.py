import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm
from src import kernels
from .visualization import plot_simulation
from .params import ModelParams

### Nonlinear (Sigmoidal)Firing Rate

def F(x):
  return 1 / (1 + np.exp(-x))

## FFT Convolution
def fft_convolution(U, K):
    k_fft = np.fft.fft2(np.array(K), axes=(0,1)) # Perform FFT on the kernels
    u_fft = np.fft.fft2(np.array(U), axes=(0,1)) # Perform FFT on the inputs
    convolution_u = np.fft.ifft2(u_fft * k_fft, axes=(0,1)) # Perform element-wise multiplication in Fourier space (frequency domain)
    Uc = np.real(convolution_u) # Take the real part of the result (to remove any residual imaginary parts due to numerical errors)

    return Uc

### Stimulus

## Step Function
def H(x):
    return np.maximum(np.sign(x), 0)

## Flicker Stimulus Function
def S(t, A, T, p: ModelParams):
    return A*H(np.sin((2*np.pi*t)/T)-p.V)

def step(A, T, t, N, Ue, Ui, Ke, Ki, p: ModelParams):
    Noise = np.random.normal(-1, 1, (N, N)) # Gaussian Noise

    ## Convolve matrix
    Uec = fft_convolution(Ue, Ke)
    Uic = fft_convolution(Ui, Ki)

    # Euler's Method of Finding Activity Rate of Change
    dUe = (p.dt/p.Te)*(-Ue + F(p.Aee*Uec-p.Aie*Uic-p.He+p.Ge*S(t*p.dt, A, T, p)+p.Ne*Noise))
    dUi = (p.dt/p.Ti)*(-Ui + F(p.Aei*Uec-p.Aii*Uic-p.Hi+p.Gi*S(t*p.dt, A, T, p)+p.Ni*Noise))

    # Updating Neural Field Activities
    Ue += dUe
    Ui += dUi
    
    return Ue, Ui

def run_simulation(N, A, T, Se, Si, TimeDuration, seed, p: ModelParams):
    # Initializing Random Activity Rates
    Ue = np.random.rand(N, N)
    Ui = np.random.rand(N, N)

    # Connectivity Kernels
    K_sidelength_E = N
    K_sidelength_I = N

    Ke = kernels.generate_gaussian_kernel(Se, K_sidelength_E)
    Ki = kernels.generate_gaussian_kernel(Si, K_sidelength_I)
    
    # Plotting Start Time
    plot_time_start = 0

    # Time interval to record activity (1 - every ms)
    plotting_range = range(plot_time_start, TimeDuration + 1, 1)

    # Activity at a random point (for plotting)
    pointE = []
    pointI = []

    # Time (for plotting)
    time = []

    # Stimulation (for plotting)
    StimE = []
    StimI = []
    
    plots = {}

    # Time Steps
    steps = int(TimeDuration/p.dt)
    
    for t in tqdm(range(steps)):
        np.random.seed(seed) # For reproducibility

        Ue, Ui = step(A, T, t, N, Ue, Ui, Ke, Ki, p)

        if t*p.dt in plotting_range:

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
        if t*p.dt != 0 and t*p.dt % 100 == 0:
            
            plots[t*p.dt] = {
                            "t": t,
                            "Ue": Ue.copy(),        # NumPy array copy
                            "Ui": Ui.copy(),
                            "time": list(time),     # or time.copy()
                            "pointE": list(pointE),
                            "pointI": list(pointI),
                            "StimE": list(StimE),
                            "StimI": list(StimI),
            }
        
    return plots