"""Visualization and plotting functions for the RSE model.

Includes utilities for converting simulated neural-field activity
into plots, images, and GIFs, a simple retinotopic
transform that maps a Cartesian field into log-polar coordinates, and
a helper to ensure unique output paths.
"""

import os
import matplotlib.gridspec as gridspec
import numpy as np
import imageio.v2 as imageio
import re
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .params import ModelParams
from .kernels import K

### Retinoptic Mapping
def retinal_transform(input_img):
    """Apply a retinotopic (log-polar) mapping to an image.

    Parameters
    ----------
    input_img : numpy.ndarray
        2D array representing an image or activity field.

    Returns
    -------
    numpy.ndarray
        Transformed image with the same shape as ``input_img``.

    Notes
    -----
    To produce a mapping of the same size as the input, the function uses
    interpolation with grid wrapping. 
    """
    
    # Get the shape of the input image
    height, width = input_img.shape

    # Create a meshgrid for the output image
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(x, y)

    # Convert Cartesian to polar coordinates
    r = np.hypot(x, y)
    theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)

    # Scale the output grid coordinates
    r_scaled = np.log(r + 1e-26) / (2 * np.pi)
    theta_scaled = theta / (2 * np.pi)
    
    # Create a new grid for the input image
    x_in = r_scaled * (width)
    y_in = theta_scaled * (height)

    # Map the input image onto the output grid
    transformed_img = map_coordinates(input_img, [y_in, x_in],
                                      order=2, mode='grid-wrap')

    return transformed_img

def ensure_unique_path(
    p,
    width: int = 3,
    start: int = 1,
    suffix_re = re.compile(r"_(\d+)$")   # match trailing _<digits> at end of *stem*
) -> Path:
    """Return a unique path by incrementing a numeric suffix to filename.

    If ``p`` already exists, this function appends or increments a ``_NNN`` suffix
    until a non-existing path is found.

    Parameters
    ----------
    p : str or pathlib.Path
        Candidate path.
    width : int, default=3
        Zero-pad width for the numeric suffix.
    start : int, default=1
        Starting number when no suffix is present.
    suffix_re : re.Pattern, optional
        Pattern used to detect an existing numeric suffix in the stem.

    Returns
    -------
    pathlib.Path
        A path guaranteed not to exist at the time of checking.

    Examples
    --------
    ``plot.png`` -> ``plot_001.png`` -> ``plot_002.png``
    """
    
    p = Path(p)
    parent, stem, suffix = p.parent, p.stem, p.suffix

    m = suffix_re.search(stem)
    if m:
        base = stem[:m.start()]
        n = max(int(m.group(1)), start - 1)
    else:
        base = stem
        n = start - 1

    candidate = p
    while candidate.exists():
        n += 1
        candidate = parent / f"{base}_{n:0{width}d}{suffix}"
    return candidate


def make_plot(t, cortical_activity, time, pointE, pointI, StimE, 
              StimI, Se, Si, A, T, N, contours, cmap,
              p: ModelParams,
              ):
    """Create a summary figure for a simulation at time t. 
    Includes:
        - Point activity over time (E and I)
        - Cortical view (contour plot)
        - Retinal view (contour plot after :func:`retinal_transform`)
        - Stimulation time series (E and I)
        - Plot of excitatory/inhibitory kernels

    Parameters
    ----------
    t : float
        Current time as stored by the simulation.
    cortical_activity : numpy.ndarray
        2D cortical activity field.
    time : sequence of float
        Time samples for the time-series plots.
    pointE : sequence of float
        Excitatory point activity over time.
    pointI : sequence of float
        Inhibitory point activity over time.
    StimE : sequence of float
        Excitatory stimulation over time.
    StimI : sequence of float
        Inhibitory stimulation over time.
    Se : float
        Excitatory kernel scale parameter.
    Si : float
        Inhibitory kernel scale parameter.
    A : float
        Stimulus amplitude (displayed in the title).
    T : float
        Stimulus period in milliseconds (displayed in the title).
    N : int
        Neural field side length (displayed in the title).
    contours : int
        Number of contour levels.
    cmap : str or matplotlib.colors.Colormap
        Colormap used for contour plots.
    p : ModelParams
        Model parameters used for metadata (e.g., ``dt`` and ``V``).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    
    ret_cortical_activity = retinal_transform(cortical_activity)

    # Plotting
    fig = plt.figure(tight_layout=True, figsize=(10,10))
    gs = gridspec.GridSpec(5, 2)

    ax1 = fig.add_subplot(gs[2, :])
    ax2 = fig.add_subplot(gs[:2, 0])
    ax3 = fig.add_subplot(gs[:2, 1])
    ax4 = fig.add_subplot(gs[3, :])
    ax5 = fig.add_subplot(gs[4, :])

    pointE_plot, = ax1.plot(time, pointE, label='E')
    pointI_plot, = ax1.plot(time, pointI, label='I')
    ax1.set_ylabel('Point Activity')
    ax1.set_xlabel('Time (ms)')
    ax1.legend(handles=[pointE_plot, pointI_plot], loc='upper right')

    cortical_plot = ax2.contourf(cortical_activity, contours, cmap=cmap)
    ax2.set_title('Cortical View')
    ax2.axis('equal')
    ax2.axis('off')

    ax3.contourf(ret_cortical_activity, contours, cmap=cmap)
    ax3.set_title('Retinal View')
    ax3.axis('equal')
    ax3.axis('off')

    stimE_plot, = ax4.plot(time, StimE, label='E')
    stimI_plot, = ax4.plot(time, StimI, label='I')
    ax4.set_ylabel('Stimulation (Amplitude)')
    ax4.set_xlabel('Time (ms)')
    ax4.legend(handles=[stimE_plot, stimI_plot])

    x = np.arange(-20, 21, 1)
    K_plotE, = ax5.plot(x, K(x, 0, Se), label='E')
    K_plotI, = ax5.plot(x, K(x, 0, Si), label='I')
    ax5.set_ylabel('Kernel \n Connectivity Factor')
    ax5.set_xlabel('x')
    ax5.legend(handles=[K_plotE, K_plotI])

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.25)

    fig.colorbar(cortical_plot, cax=cax)

    fig.suptitle(
        "{} ms - ".format(round(t * p.dt))
        + "A:{} ".format(round(A, 2))
        + "T:{} ".format(T)
        + "Se:{} ".format(round(Se, 2))
        + "Si:{} ".format(round(Si, 2))
        + "dt:{} ".format(p.dt)
        + "N:{} ".format(N)
        + "V: {} ".format(p.V)
    )

    return fig
    
def make_images(t, cortical_activity, time, Se, images,
                Si, A, T, N, contours, out_path, dpi,
                label, cmap, p: ModelParams, **_ignore
                ):
    """Save one or two contour images for a simulation snapshot.

    Depending on ``images``, this writes a cortical view, a retinal view, or both
    into ``out_path``.

    Parameters
    ----------
    t : float
        Snapshot time as stored by the simulation.
    cortical_activity : numpy.ndarray
        2D cortical activity field.
    time : sequence of float
        Unused here, passed through using same parameters.
    Se : float
        Excitatory kernel scale parameter (used for labeling).
    images : {'cortical', 'retinal', 'both'}
        Which images to make.
    Si : float
        Inhibitory kernel scale parameter (used for labeling).
    A : float
        Stimulus amplitude (used for labeling).
    T : float
        Stimulus period in milliseconds (used for labeling).
    N : int
        Neural field side length (used for labeling).
    contours : int
        Number of contour levels.
    out_path : str or os.PathLike
        Directory where images will be written.
    dpi : int
        Output resolution passed to Matplotlib.
    label : bool
        If True, add a title string with simulation metadata.
    cmap : str or matplotlib.colors.Colormap
        Colormap used for contour plots.
    p : ModelParams
        Model parameters used for metadata (e.g., ``dt`` and ``V``).
    **_ignore
        Extra keyword arguments are accepted and ignored for convenience.

    Returns
    -------
    None
    """

    label_text = (
        "{} ms - ".format(round(t * p.dt))
        + "A:{} ".format(round(A, 2))
        + "T:{} ".format(T)
        + "Se:{} ".format(round(Se, 2))
        + "Si:{} ".format(round(Si, 2))
        + "dt:{} ".format(p.dt)
        + "N:{} ".format(N)
        + "V: {} ".format(p.V)
    )

    ret_cortical_activity = retinal_transform(cortical_activity)

    if images in ("cortical", "both"):
        plt.figure(tight_layout=True, figsize=(10,10))

        plt.contourf(cortical_activity, contours, cmap=cmap)
        plt.axis('equal')
        plt.axis('off')

        if label:
            plt.suptitle(label_text)

        filename = os.path.join(out_path, f"cortical_{round(t * p.dt)}ms.png")
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)

    if images in ("retinal", "both"):
        plt.figure(tight_layout=True, figsize=(10,10))

        plt.contourf(ret_cortical_activity, contours, cmap=cmap)
        plt.axis('equal')
        plt.axis('off')

        if label:
            plt.suptitle(label_text)

        filename = os.path.join(out_path, f"retinal_{round(t * p.dt)}ms.png")
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)

def make_gif(data, Se, Si, A, T, N, contours, cmap, out_path, label, dpi,
             p: ModelParams, fps=50,
             ):
    """Render a GIF from recorded simulation frames.

    This function iterates through ``data`` (typically ``plots['gif']`` from the
    simulation), renders each frame as a contour plot of the retinal transform,
    and writes ``simulation.gif`` into ``out_path``.

    Parameters
    ----------
    data : dict
        Mapping of times to per-frame values. Each value is expected to contain
        a ``'cortical_activity'`` 2D array.
    Se : float
        Excitatory kernel scale parameter (used for labeling).
    Si : float
        Inhibitory kernel scale parameter (used for labeling).
    A : float
        Stimulus amplitude (used for labeling).
    T : float
        Stimulus period in milliseconds (used for labeling).
    N : int
        Neural field side length (used for labeling).
    contours : int
        Number of contour levels.
    cmap : str or matplotlib.colors.Colormap
        Colormap used for contour plots.
    out_path : str or os.PathLike
        Output directory where ``simulation.gif`` is written.
    label : bool
        If True, add a title string to each frame.
    dpi : int
        Output resolution for intermediate frame images.
    p : ModelParams
        Model parameters used for metadata (e.g., ``dt`` and ``V``).
    fps : int, default=50
        Frames per second for the GIF.

    Returns
    -------
    None

    Notes
    -----
    Frames are rendered to a temporary directory (``frames_temp``) and cleaned up
    afterwards.
    """

    frames = []
    temp_dir = "frames_temp"
    os.makedirs(temp_dir, exist_ok=True)

    for i, (t, values) in tqdm(enumerate(data.items()), total=len(data)):
        # Cortical Activity
        cortical_activity = values['cortical_activity']
        ret_cortical_activity = retinal_transform(cortical_activity)

        # Plotting
        fig = plt.figure(tight_layout=True, figsize=(10,10))
        plt.contourf(ret_cortical_activity, contours, cmap=cmap)
        # vmin=global_min, vmax=global_max)
        plt.axis('equal')
        plt.axis('off')

        if label is True:
            plt.title("{} s".format(round(t / 1000, 1)))
            plt.suptitle(
                "A:{} ".format(round(A, 2))
                + "T:{} ".format(T)
                + "Se:{} ".format(round(Se, 2))
                + "Si:{} ".format(round(Si, 2))
                + "dt:{} ".format(p.dt)
                + "N:{} ".format(N)
                + "V: {} ".format(p.V)
            )

        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, bbox_inches='tight', dpi=dpi)
        frames.append(imageio.imread(frame_path))
        plt.close(fig)

    print("Saving GIF...")

    # Combine frames into a GIF with simulation-accurate timing
    imageio.mimsave(
        os.path.join(out_path, "simulation.gif"),
        frames,
        duration=1000 / fps,
        loop=0,
    )  # duration in seconds per frame

    print("GIF saved")

    # Clean up
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)