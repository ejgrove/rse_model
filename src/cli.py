#!/usr/bin/env python

"""Command-line interface for running RSE model simulations.

This module defines the CLI used to configure model parameters, execute a
simulation, and generate plots, images, and GIF outputs.
"""

import argparse
import os
import re
from pathlib import Path

from .model import run_simulation
from .visualization import make_plot, make_gif, make_images, \
    ensure_unique_path
from .params import ModelParams


def odd_positive_int(value: str) -> int:
    """Parse an integer and coerce it to an odd positive value. 
    Necessary for the model to center kernels correctly.

    Parameters
    ----------
    value : str
        CLI argument value.

    Returns
    -------
    int
        An odd integer computed as ``(n // 2) * 2 + 1``.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``value`` cannot be parsed as an integer or is not positive.
    """
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value!r}") from exc

    if n <= 0:
        raise argparse.ArgumentTypeError("N must be a positive integer")

    return (n // 2) * 2 + 1

def main():
    ap = argparse.ArgumentParser()
    
    ### Model Parameters
    ap.add_argument("--N", type=odd_positive_int, default=101) # Neural field size (forced odd)
    ap.add_argument("--A", type=float, default=0.7) # Amplitude
    ap.add_argument("--T", type=float, default=115) # Period (ms)
    ap.add_argument("--Se", type=float, default=2.0) # Excitatory Kernel Std Dev
    ap.add_argument("--Si", type=float, default=5.0) # Inhibitory Kernel Std Dev

    ### Simulation Parameters
    ap.add_argument("--seed", type=int, default=None) # Random seed
    ap.add_argument("--start", type=int, default=0) # Time (ms) to start saving outputs
    ap.add_argument("--end", type=int, default=2000) # Time (ms) to end saving outputs
    ap.add_argument("--interval", type=int, default=1000) # Time interval (ms) for saving outputs

    ### Visualization Parameters
    ap.add_argument("--plot", action='store_true') # Save plot of simulation images, stimulation, activity, and connectivity
    ap.add_argument("--images", type=str, default=None) # Save images of simulation ("retinal", "cortical", or "both")
    ap.add_argument("--contours", type=int, default=50) # Number of contours for visualization
    ap.add_argument("--cmap", type=str, default='plasma') # Matplotlib colormap
    ap.add_argument("--dpi", type=int, default=100) # Matplotlib dpi (image resolution)
    ap.add_argument("--out-path", type=str, default='outputs') # Output directory
    
    ### GIF Parameters
    ap.add_argument("--gif", action='store_true') # Whether to save gif of simulation
    ap.add_argument("--fps", type=int, default=50) # Frames per second for gif
    ap.add_argument("--label", action='store_true') # Whether to add label to gif

    args = ap.parse_args()
    
    # Create output path
    if args.gif or (args.images is not None) or args.plot:
        file_suffix = (
            f"simulation_A{str(args.A).replace('.', '_')}"
            f"_T{str(round(args.T)).replace('.', '_')}"
            f"_Se{str(round(args.Se)).replace('.', '_')}"
            f"_Si{str(round(args.Si)).replace('.', '_')}"
            f"_N{str(args.N)}"
        )

        out_path = os.path.join(args.out_path, file_suffix)
        out_path = ensure_unique_path(out_path)
        os.makedirs(out_path, exist_ok=False)
        
        print(f"Outputs will be saved to: {out_path}")

    # Simulation parameter dict
    params = {"N": args.N,
              "A": args.A,
              "T": args.T,
              "Se": args.Se,
              "Si": args.Si,
              "p": ModelParams()
              }
    
    # Visualization params dict
    vis_params = {"contours": args.contours,
                    "cmap": args.cmap,
                }

    # Run simulation
    data = run_simulation(**params,
                        seed=args.seed,
                        gif= args.gif,
                        interval=args.interval,
                        start_time= args.start,
                        end_time= args.end,
                        )
    # Generate GIF
    if args.gif:
        make_gif(data["gif"], 
                 label=args.label,
                 out_path=out_path,
                 fps=args.fps,
                 dpi=args.dpi,
                 **params,
                 **vis_params
                 )
    
    # Generate images and/or plots
    if args.images != None or args.plot:
        print("Generating images and/or plots...")
        for t, values in data["images"].items():
            if args.plot:
                plot_dir = os.path.join(out_path, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                
                fig = make_plot(**values,
                                **params,
                                **vis_params
                                )

                fig.savefig(os.path.join(plot_dir,
                    f"plot_{round(t)}ms.png"), dpi=100)

            if args.images != None:
                plot_dir = os.path.join(out_path, "images")
                os.makedirs(plot_dir, exist_ok=True)
                
                make_images(**values,
                            images=args.images,
                            label=args.label,
                            out_path=plot_dir,
                            dpi=args.dpi,
                            **params,
                            **vis_params
                            )

if __name__ == "__main__":
    main()