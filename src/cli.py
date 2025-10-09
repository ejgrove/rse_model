import argparse
import time
from .model import run_simulation
from .visualization import plot_simulation
from .params import ModelParams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=101) # Neural Field Size
    ap.add_argument("--Time", type=int, default=1000) # Time Duration of Simulation (ms)
    ap.add_argument("--A", type=float, default=0.7) # Flicker Amplittude
    ap.add_argument("--T", type=float, default=115) # Flicker Period (ms)
    ap.add_argument("--Se", type=float, default=2.0) # Excitatory Spatial Spread
    ap.add_argument("--Si", type=float, default=5.0) # Inhibitory Spatial Spread
    ap.add_argument("--seed", type=int, default=42) # Random Seed
    ap.add_argument("--out", type=str, default='outputs/simulation.png') # Output file prefix
    args = ap.parse_args()
    

    data = run_simulation(N=args.N, 
                            A=args.A, 
                            T=args.T, 
                            Se=args.Se, 
                            Si=args.Si, 
                            TimeDuration=args.Time,
                            seed=args.seed,
                            p=ModelParams())
    
    for t, values in data.items():
        
        print("t:", t)

        print(len(values["time"]))
        print(len(values["pointE"]))
        
        fig = plot_simulation(**values,
                            contours=50,
                            Se=args.Se,
                            Si=args.Si,
                            A=args.A,
                            T=args.T,
                            N=args.N,
                            p=ModelParams())

        fig.savefig(f"{args.out.split('.')[0]}_{t}.png", dpi=200)

if __name__ == "__main__":
    main()