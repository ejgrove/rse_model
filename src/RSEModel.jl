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
    generate_gaussian_kernel_1d,
    firing_rate,
    DEFAULT_FFT_FLAGS,
    fft_convolution!,
    run_simulation,
    run_simulation_gpu,
    retinal_transform,
    ensure_unique_path,
    make_images,
    make_plot,
    make_gif,
    is_fast_fft_size,
    next_fast_odd_size,
    odd_positive_int,
    main

end
