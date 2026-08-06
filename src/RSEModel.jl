module RSEModel

include("Params.jl")
include("Kernels.jl")
include("Model.jl")
include("Visualization.jl")
include("ParameterSearch.jl")
include("CommandLine.jl")
include("Applet.jl")

export ModelParams,
    Snapshot,
    SimulationOutput,
    gaussian_kernel_2d,
    generate_gaussian_kernel,
    generate_gaussian_kernel_1d,
    firing_rate,
    DEFAULT_FFT_FLAGS,
    fft_convolution!,
    duty_cycle_percent_from_threshold,
    stimulus_threshold_from_duty_cycle_percent,
    run_simulation,
    run_simulation_gpu,
    retinal_transform,
    ensure_unique_path,
    make_images,
    make_plot,
    make_gif,
    ParameterSearchConfig,
    run_parameter_search,
    parameter_search_main,
    is_fast_fft_size,
    next_fast_odd_size,
    odd_positive_int,
    LiveConfig,
    LiveFrame,
    normalize_live_config,
    live_config_from_query,
    stream_live_frames,
    serve_applet,
    serve_applet_async,
    applet_url,
    main

end
