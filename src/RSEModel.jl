module RSEModel

include("Params.jl")
include("Kernels.jl")
include("Model.jl")
include("Visualization.jl")
include("CommandLine.jl")

export ModelParams,
    Snapshot,
    SimulationOutput,
    gaussian_kernel_2d,
    generate_gaussian_kernel,
    firing_rate,
    fft_convolution!,
    run_simulation,
    retinal_transform,
    ensure_unique_path,
    make_images,
    make_plot,
    make_gif,
    odd_positive_int,
    main

end
