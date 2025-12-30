"""Model parameter definitions.

Defines `~rse_model.params.ModelParams`, a frozen
dataclass containing default constants used throughout the simulation.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelParams:
    """Parameters for the RSE model.

    Notes
    -----
    Increasing the number of nodes (neural populations) increases spatial
    frequency; however beyond roughly ``N ~ 250`` the forms may not fully develop.

    Attributes
    ----------
    dt : float
        Time step in milliseconds.
    Te : float
        Excitatory time constant (Rule et al. 2011).
    Ti : float
        Inhibitory time constant (Rule et al. 2011).
    Aee, Aei, Aie, Aii : float
        Connection strengths between neural populations (Rule et al. 2011).
    He : float
        Excitatory firing threshold/bias (Rule et al. 2011).
    Hi : float
        Inhibitory firing threshold/bias (Rule et al. 2011).
    Ge : float
        Gain on stimulation / input drive for excitatory population.
    Gi : float
        Gain on stimulation / input drive for inhibitory population.
    Ne : float
        Excitatory noise strength (Rule et al. 2011).
    Ni : float
        Inhibitory noise strength (Rule et al. 2011).
    V : float
        Threshold on the sinusoid controlling the effective duty cycle of the
        stimulus (Rule et al. 2011).
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