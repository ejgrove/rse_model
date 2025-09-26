from dataclasses import dataclass

@dataclass(frozen=True)
class ModelParams:
    """Parameters for the RSE model."""
    
    ''' Increasing Number of nodes (neural populations) increases spatial frequency
    however beyond ~250 the forms won't fully develop'''

    """
    
    # Number node on side of neural field (must be odd for fft transform)

    # Spatial Connectivity Constants - Rule et al. 2011
    Se: float = 2
    Si: float = 2.5 * Se
    """

    # Timestep (ms)
    dt: float = 0.2

    # Time constant - Rule et al. 2011
    Te: float = 10.0
    Ti: float = 20.0

    # Connectivity Constants - Rule et al. 2011
    Aee: float = 10
    Aei: float = 12
    Aie: float = 8.5
    Aii: float = 3.0

    # Threshold/Bias (theta) - Rule et al. 2011
    He: float = 2.0
    Hi: float = 3.5

    # Gain on stimulation / Input drive - Rule et al. 2011
    Ge: float = 1
    Gi: float = 0

    # Noise = Rule et al. 2011
    Ne: float = 0.05
    Ni: float = 0.05
    
    # Constant determining proportion of light to dark from Rule et al (2011))
    V: float = 0.8