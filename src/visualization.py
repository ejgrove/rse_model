from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
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
    transformed_img = map_coordinates(input_img, [y_in, x_in], order=2, mode='grid-wrap')

    return transformed_img


def plot_simulation(t, Ue, Ui, time, pointE, pointI, StimE, StimI, Se, Si, A, T, N, contours,
                    p: ModelParams):
    
        # Cortical Activity
        cortical_activity = np.abs(Ue - Ui)
    
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

        cortical_plot = ax2.contourf(cortical_activity, contours, cmap='plasma')
        ax2.set_title('Cortical View')
        ax2.axis('equal')
        ax2.axis('off')

        retinal_plot = ax3.contourf(ret_cortical_activity, contours,
                      cmap='plasma')
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
        ax5.set_ylabel('Connectivity Factor')
        ax5.set_xlabel('x')
        ax5.legend(handles=[K_plotE,K_plotI])

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.25)

        fig.colorbar(cortical_plot, cax=cax)

        fig.suptitle("2d FFT Conv - {} ms - ".format(round(t*p.dt))+
                    "A:{} ".format(round(A,2))+
                    "T:{} ".format(T)+
                    "Se:{} ".format(round(Se,2))+
                    "Si:{} ".format(round(Si,2))+
                    "dt:{} ".format(p.dt)+
                    "N:{} ".format(N)+
                    "Ne:{} ".format(p.Ne)+
                    "Ni:{} ".format(p.Ni)+
                    "V: {} ".format(p.V))

        return fig  # let caller decide whether to show() or savefig()
    
