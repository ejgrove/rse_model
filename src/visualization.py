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

from pathlib import Path
import re

def ensure_unique_path(
    p,
    width: int = 3,
    start: int = 1,
    suffix_re = re.compile(r"_(\d+)$")   # match trailing _<digits> at end of *stem*
) -> Path:
    """
    If 'p' exists, append or increment a _NNN suffix until a free path is found.
    Example sequence: plot.png -> plot_001.png -> plot_002.png -> ...
    - width: zero-pad width for the numeric suffix.
    - start: starting number when no suffix is present.
    - suffix_re: pattern used to detect an existing numeric suffix in the stem.
    """
    p = Path(p)
    parent, stem, suffix = p.parent, p.stem, p.suffix

    m = suffix_re.search(stem)
    if m:
        base = stem[:m.start()]
        n = max(int(m.group(1)), start - 1)  # don't go backwards
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

        retinal_plot = ax3.contourf(ret_cortical_activity, contours,
                      cmap=cmap)
        ax3.set_title('Retinal View')
        ax3.axis('equal')
        ax3.axis('off')

        stimE_plot, = ax4.plot(time, StimE, label = 'E')
        stimI_plot, = ax4.plot(time, StimI, label = 'I')
        ax4.set_ylabel('Stimulation (Amplitude)')
        ax4.set_xlabel('Time (ms)')
        ax4.legend(handles=[stimE_plot,stimI_plot])

        x = np.arange(-20, 21, 1)
        K_plotE, = ax5.plot(x, K(x, 0, Se),label='E')
        K_plotI, = ax5.plot(x, K(x, 0, Si), label='I')
        ax5.set_ylabel('Kernel \n Connectivity Factor')
        ax5.set_xlabel('x')
        ax5.legend(handles=[K_plotE,K_plotI])

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.25)

        fig.colorbar(cortical_plot, cax=cax)

        fig.suptitle("{} ms - ".format(round(t*p.dt))+
                    "A:{} ".format(round(A,2))+
                    "T:{} ".format(T)+
                    "Se:{} ".format(round(Se,2))+
                    "Si:{} ".format(round(Si,2))+
                    "dt:{} ".format(p.dt)+
                    "N:{} ".format(N)+
                    "V: {} ".format(p.V))

        return fig
    
def make_images(t, cortical_activity, time, Se, images,
                Si, A, T, N, contours, out_path, dpi,
                label, cmap, p: ModelParams, **_ignore
                ):
    
        label_text = "{} ms - ".format(round(t*p.dt)) \
                        + "A:{} ".format(round(A,2)) \
                        + "T:{} ".format(T) \
                        + "Se:{} ".format(round(Se,2)) \
                        + "Si:{} ".format(round(Si,2)) \
                        + "dt:{} ".format(p.dt) \
                        + "N:{} ".format(N) \
                        + "V: {} ".format(p.V)

        ret_cortical_activity = retinal_transform(cortical_activity)

        if images in ("cortical", "both"):
            fig = plt.figure(tight_layout=True, figsize=(10,10))

            plt.contourf(cortical_activity, contours, cmap=cmap)
            plt.axis('equal')
            plt.axis('off')
            
            if label:
                plt.suptitle(label_text)
                
            filename = os.path.join(out_path, f"cortical_{round(t*p.dt)}ms.png")
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
            
        if images in ("retinal", "both"):
            fig = plt.figure(tight_layout=True, figsize=(10,10))
            
            plt.contourf(ret_cortical_activity, contours, cmap=cmap)
            plt.axis('equal')
            plt.axis('off')

            if label:
                plt.suptitle(label_text)

            filename = os.path.join(out_path, f"retinal_{round(t*p.dt)}ms.png")
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)

def make_gif(data, Se, Si, A, T, N, contours, cmap, out_path, label, dpi,
             p: ModelParams, fps=50,
             ):
        
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

            if label == True:
                plt.title("{} s".format(round(t/1000, 1))) 
                plt.suptitle("A:{} ".format(round(A,2))+
                            "T:{} ".format(T)+
                            "Se:{} ".format(round(Se,2))+
                            "Si:{} ".format(round(Si,2))+
                            "dt:{} ".format(p.dt)+
                            "N:{} ".format(N)+
                            "V: {} ".format(p.V))
                
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            plt.savefig(frame_path, bbox_inches='tight', dpi=dpi)
            frames.append(imageio.imread(frame_path))
            plt.close(fig)
        
        print("Saving GIF...")
            
        # Combine frames into a GIF with simulation-accurate timing
        imageio.mimsave(os.path.join(out_path, "simulation.gif"), frames, 
                        duration=1000/fps, loop=0) # duration in seconds per frame
        
        print(f"GIF saved")
        
        # Clean up
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)