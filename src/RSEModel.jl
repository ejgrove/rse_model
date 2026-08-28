"""Core model, geometry, streaming runtime, and HTTP server for the RSE web app."""
module RSEModel

include("Params.jl")
include("Grid.jl")
include("Kernels.jl")
include("Geometry.jl")
include("Model.jl")
include("RetinalMapping.jl")
include("Applet.jl")

export ModelParams,
    generate_gaussian_kernel,
    generate_gaussian_kernel_1d,
    FieldGeometry,
    field_geometry,
    double_sech_shear,
    dipole_double_sech_map,
    DoubleSechRetinalPlan,
    double_sech_retinal_plan,
    double_sech_retinal_transform!,
    double_sech_retinal_transform,
    has_field_mask,
    apply_field_mask!,
    firing_rate,
    DEFAULT_FFT_FLAGS,
    fft_convolution!,
    duty_cycle_percent_from_threshold,
    stimulus_threshold_from_duty_cycle_percent,
    RetinalMapPlan,
    retinal_map_plan,
    retinal_transform!,
    retinal_transform,
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
    applet_url

end
