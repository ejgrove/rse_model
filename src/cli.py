import argparse
from .model import run_simulation
from .viz import plot_simulation
from .params import ModelParams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=115)
    ap.add_argument("--Time", type=int, default=1000)
    ap.add_argument("--A", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--Se", type=float, default=2.0)
    ap.add_argument("--Si", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default='outputs/simulation.png')
    args = ap.parse_args()
    
    data = run_simulation(N=args.N, 
                            A=args.A, 
                            T=args.T, 
                            Se=args.Se, 
                            Si=args.Si, 
                            TimeDuration=args.Time,
                            seed=args.seed,
                            p=ModelParams())

    fig = plot_simulation(**data,
                            contours=50,
                            Se=args.Se,
                            Si=args.Si,
                            A=args.A,
                            T=args.T,
                            N=args.N,
                            p=ModelParams())
    
    fig.savefig(args.out, dpi=200)

if __name__ == "__main__":
    main()