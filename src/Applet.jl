using Base64
using HTTP

Base.@kwdef struct LiveConfig
    N::Int = 81
    fast_n::Bool = true
    backend::Symbol = :metal
    convolution::Symbol = :separable
    A::Float32 = 0.7f0
    period::Float32 = 115.0f0
    duty_cycle_percent::Union{Nothing,Float32} = Float32(duty_cycle_percent_from_threshold(ModelParams{Float32}().V))
    Se::Float32 = 2.0f0
    Si::Float32 = 5.0f0
    dt::Float32 = 0.2f0
    seed::Union{Nothing,Int} = nothing
    target_fps::Int = 30
    speed::Float64 = 1.0
    gpu_threads::Int = 256
    kernel_cutoff::Float64 = 3.0
    boundary::Union{Nothing,Symbol} = nothing
    boundary_x::Symbol = :periodic
    boundary_y::Symbol = :periodic
    partial_reflect_strength::Float32 = 0.5f0
    coupling::Symbol = :none
    coupling_strength::Float32 = 0.02f0
    overlap_rows::Int = 6
    field_geometry::Symbol = :square
    field_density::Float64 = 1.0
    activity_scale::Symbol = :frame
    max_frames::Int = 0
end

Base.@kwdef struct LiveFrame
    frame::Int
    N::Int
    rows::Int
    cols::Int
    retinal_n::Int
    retinal_rows::Int
    retinal_cols::Int
    t::Float64
    lo::Float32
    hi::Float32
    step_ms::Float64
    frame_ms::Float64
    ms_per_step::Float64
    realtime_x::Float64
    steps_per_frame::Int
    data::Vector{UInt8}
    retinal_data::Vector{UInt8}
    phase_count::Int
    phase_e_data::Vector{UInt8}
    phase_i_data::Vector{UInt8}
end

Base.@kwdef mutable struct LiveRuntime
    target_fps::Int = 30
    speed::Float64 = 1.0
    activity_scale::Symbol = :frame
    scale_lo::Float32 = Float32(Inf)
    scale_hi::Float32 = -Float32(Inf)
    throttle_deadline_ns::UInt64 = 0
    control_version::Int = 0
end

function normalize_live_config(config::LiveConfig)
    backend = config.backend == :gpu ? :metal : config.backend
    backend in (:cpu, :metal) || throw(ArgumentError("backend must be :cpu or :metal."))
    boundary_x, boundary_y = _resolve_boundaries(config.boundary, config.boundary_x, config.boundary_y)
    geometry_kind = _normalize_field_geometry(config.field_geometry)
    config.field_density > 0 || throw(ArgumentError("field_density must be positive."))
    if geometry_kind == :double_sech
        boundary_x == :periodic && (boundary_x = :edge)
        boundary_y == :periodic && (boundary_y = :edge)
    end

    convolution = if config.convolution == :auto
        geometry_kind == :double_sech && backend == :metal ? :separable : _default_convolution(backend, boundary_x, boundary_y)
    else
        config.convolution
    end
    convolution in (:fft, :separable) || throw(ArgumentError("convolution must be :auto, :fft, or :separable."))
    backend == :metal || convolution == :fft ||
        throw(ArgumentError("The CPU live backend currently supports FFT convolution only."))
    if geometry_kind == :double_sech && (backend != :metal || convolution != :separable)
        throw(ArgumentError("The double-sech live field currently requires the Metal separable backend."))
    end
    _validate_boundaries(boundary_x, boundary_y, convolution, backend)
    coupling = _normalize_live_coupling(config.coupling)
    activity_scale = _normalize_activity_scale(config.activity_scale)

    target_fps = max(1, config.target_fps)
    N = if geometry_kind == :double_sech
        field_geometry(:double_sech; density=config.field_density).rows
    else
        config.fast_n ? next_fast_odd_size(config.N) : odd_positive_int(config.N)
    end
    config.speed >= 0 || throw(ArgumentError("speed must be non-negative."))
    config.dt > 0 || throw(ArgumentError("dt must be positive."))
    config.gpu_threads > 0 || throw(ArgumentError("gpu_threads must be positive."))
    config.kernel_cutoff > 0 || throw(ArgumentError("kernel_cutoff must be positive."))
    config.max_frames >= 0 || throw(ArgumentError("max_frames must be non-negative."))
    0 <= config.partial_reflect_strength <= 1 ||
        throw(ArgumentError("partial_reflect_strength must be between 0 and 1."))
    config.coupling_strength >= 0 || throw(ArgumentError("coupling_strength must be non-negative."))
    overlap_rows = max(2, 2 * cld(config.overlap_rows, 2))
    overlap_rows = min(overlap_rows, max(2, 2 * div(max(N - 1, 2), 4)))
    if config.duty_cycle_percent !== nothing
        0 <= config.duty_cycle_percent <= 100 ||
            throw(ArgumentError("duty_cycle_percent must be between 0 and 100."))
    end

    return LiveConfig(
        N=N,
        fast_n=config.fast_n,
        backend=backend,
        convolution=convolution,
        A=config.A,
        period=config.period,
        duty_cycle_percent=config.duty_cycle_percent,
        Se=config.Se,
        Si=config.Si,
        dt=config.dt,
        seed=config.seed,
        target_fps=target_fps,
        speed=config.speed,
        gpu_threads=config.gpu_threads,
        kernel_cutoff=config.kernel_cutoff,
        boundary=nothing,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        partial_reflect_strength=config.partial_reflect_strength,
        coupling=coupling,
        coupling_strength=config.coupling_strength,
        overlap_rows=overlap_rows,
        field_geometry=geometry_kind,
        field_density=config.field_density,
        activity_scale=activity_scale,
        max_frames=config.max_frames,
    )
end

function _normalize_live_coupling(coupling::Symbol)
    coupling == :none && return :off
    coupling == :midline && return :overlap
    coupling in (:off, :no_connection, :overlap) && return coupling
    throw(ArgumentError("coupling must be :off, :no_connection, or :overlap."))
end

function _uses_two_hemispheres(config::LiveConfig)
    return config.field_geometry == :double_sech || config.coupling in (:no_connection, :overlap)
end

function _uses_overlap_coupling(config::LiveConfig)
    return config.coupling == :overlap && config.coupling_strength > 0
end

function _normalize_activity_scale(scale::Symbol)
    scale in (:frame, :local) && return :frame
    scale in (:simulation, :sim, :global) && return :simulation
    throw(ArgumentError("activity_scale must be :frame or :simulation."))
end

function _parse_bool(value::AbstractString, default::Bool)
    isempty(value) && return default
    key = lowercase(value)
    key in ("1", "true", "yes", "on") && return true
    key in ("0", "false", "no", "off") && return false
    return default
end

function _get(params, key, default)
    return get(params, key, default)
end

function _parse_int(params, key, default)
    value = _get(params, key, nothing)
    value === nothing && return default
    return parse(Int, value)
end

function _parse_float(params, key, default)
    value = _get(params, key, nothing)
    value === nothing && return default
    return parse(Float64, value)
end

function _parse_float32(params, key, default)
    return Float32(_parse_float(params, key, Float64(default)))
end

function _parse_optional_float32(params, key)
    value = _get(params, key, "")
    isempty(value) && return nothing
    return Float32(parse(Float64, value))
end

function _parse_symbol(params, key, default)
    value = _get(params, key, nothing)
    value === nothing && return default
    return Symbol(lowercase(value))
end

function _parse_optional_symbol(params, key)
    value = _get(params, key, "")
    isempty(value) && return nothing
    return Symbol(lowercase(value))
end

function live_config_from_query(params::AbstractDict{String,String})
    seed_value = _get(params, "seed", "")
    seed = isempty(seed_value) ? nothing : parse(Int, seed_value)
    backend = _parse_symbol(params, "backend", :metal)
    convolution_default = backend == :cpu ? :fft : :separable

    return normalize_live_config(LiveConfig(
        N=_parse_int(params, "N", 81),
        fast_n=_parse_bool(_get(params, "fast_n", "true"), true),
        backend=backend,
        convolution=_parse_symbol(params, "conv", convolution_default),
        A=_parse_float32(params, "A", 0.7f0),
        period=_parse_float32(params, "T", 115.0f0),
        duty_cycle_percent=_parse_optional_float32(params, "duty_cycle"),
        Se=_parse_float32(params, "Se", 2.0f0),
        Si=_parse_float32(params, "Si", 5.0f0),
        dt=_parse_float32(params, "dt", 0.2f0),
        seed=seed,
        target_fps=_parse_int(params, "fps", 30),
        speed=_parse_float(params, "speed", 1.0),
        gpu_threads=_parse_int(params, "gpu_threads", 256),
        kernel_cutoff=_parse_float(params, "kernel_cutoff", 3.0),
        boundary=_parse_optional_symbol(params, "boundary"),
        boundary_x=_parse_symbol(params, "boundary_x", _parse_symbol(params, "boundary", :periodic)),
        boundary_y=_parse_symbol(params, "boundary_y", _parse_symbol(params, "boundary", :periodic)),
        partial_reflect_strength=_parse_float32(params, "partial_reflect_strength", 0.5f0),
        coupling=_parse_symbol(params, "coupling", :none),
        coupling_strength=_parse_float32(params, "coupling_strength", 0.02f0),
        overlap_rows=_parse_int(params, "overlap_rows", 6),
        field_geometry=_parse_symbol(params, "field_geometry", _parse_symbol(params, "geometry", :square)),
        field_density=_parse_float(params, "field_density", 1.0),
        activity_scale=_parse_symbol(params, "activity_scale", :frame),
        max_frames=_parse_int(params, "max_frames", 0),
    ))
end

function _live_model_params(config::LiveConfig)
    return ModelParams{Float32}(dt=config.dt)
end

function _live_runtime(config::LiveConfig)
    return LiveRuntime(
        target_fps=config.target_fps,
        speed=config.speed,
        activity_scale=config.activity_scale,
    )
end

function _live_field_geometry(config::LiveConfig)
    if config.field_geometry == :double_sech
        return field_geometry(:double_sech; density=config.field_density)
    end
    return field_geometry(:square, config.N)
end

function _steps_per_frame(target_fps::Integer, p::ModelParams)
    target_frame_ms = 1000 / max(1, target_fps)
    return max(1, round(Int, target_frame_ms / p.dt))
end

function _steps_per_frame(config::LiveConfig, p::ModelParams)
    return _steps_per_frame(config.target_fps, p)
end

function _activity_bytes(activity::AbstractMatrix; lo=nothing, hi=nothing)
    rows, cols = size(activity)
    lo_value = lo === nothing ? Float32(minimum(activity)) : Float32(lo)
    hi_value = hi === nothing ? Float32(maximum(activity)) : Float32(hi)
    scale = hi_value == lo_value ? 0.0f0 : 255.0f0 / (hi_value - lo_value)
    bytes = Vector{UInt8}(undef, rows * cols)

    @inbounds for row in 1:rows, col in 1:cols
        value = hi_value == lo_value ? 0.0f0 : (Float32(activity[row, col]) - lo_value) * scale
        bytes[(row - 1) * cols + col] = UInt8(round(Int, clamp(value, 0.0f0, 255.0f0)))
    end

    return bytes, lo_value, hi_value
end

function _phase_rate_byte(value)
    return UInt8(round(Int, 255 * clamp(Float32(value), 0.0f0, 1.0f0)))
end

function _phase_bytes(
    Ue::AbstractMatrix,
    Ui::AbstractMatrix;
    mask::Union{Nothing,AbstractMatrix{Bool}}=nothing,
)
    rows, cols = size(Ue)
    size(Ui) == (rows, cols) || throw(ArgumentError("Ue and Ui phase matrices must have matching sizes."))
    if mask !== nothing && size(mask) != (rows, cols)
        throw(ArgumentError("phase mask must match the firing-rate matrices."))
    end

    count_valid = mask === nothing ? length(Ue) : count(mask)
    e_bytes = Vector{UInt8}(undef, count_valid)
    i_bytes = Vector{UInt8}(undef, count_valid)
    idx = 1

    @inbounds for col in 1:cols, row in 1:rows
        if mask === nothing || mask[row, col]
            e_bytes[idx] = _phase_rate_byte(Ue[row, col])
            i_bytes[idx] = _phase_rate_byte(Ui[row, col])
            idx += 1
        end
    end

    return e_bytes, i_bytes
end

function _phase_bytes(
    Ue_left::AbstractMatrix,
    Ui_left::AbstractMatrix,
    Ue_right::AbstractMatrix,
    Ui_right::AbstractMatrix;
    mask::Union{Nothing,AbstractMatrix{Bool}}=nothing,
)
    left_e, left_i = _phase_bytes(Ue_left, Ui_left; mask=mask)
    right_e, right_i = _phase_bytes(Ue_right, Ui_right; mask=mask)
    e_bytes = Vector{UInt8}(undef, length(left_e) + length(right_e))
    i_bytes = Vector{UInt8}(undef, length(left_i) + length(right_i))
    copyto!(e_bytes, 1, left_e, 1, length(left_e))
    copyto!(e_bytes, length(left_e) + 1, right_e, 1, length(right_e))
    copyto!(i_bytes, 1, left_i, 1, length(left_i))
    copyto!(i_bytes, length(left_i) + 1, right_i, 1, length(right_i))
    return e_bytes, i_bytes
end

function _activity_scale_bounds!(runtime::Union{Nothing,LiveRuntime}, activity::AbstractMatrix)
    frame_lo = Float32(minimum(activity))
    frame_hi = Float32(maximum(activity))
    runtime === nothing && return frame_lo, frame_hi

    runtime.scale_lo = min(runtime.scale_lo, frame_lo)
    runtime.scale_hi = max(runtime.scale_hi, frame_hi)
    if runtime.activity_scale == :simulation
        return runtime.scale_lo, runtime.scale_hi
    end
    return frame_lo, frame_hi
end

function _make_live_frame(
    activity::AbstractMatrix,
    frame_idx::Integer,
    t,
    step_ms::Float64,
    frame_start_ns::UInt64,
    steps_per_frame::Integer,
    p::ModelParams,
    retinal_activity::AbstractMatrix=activity,
    ;
    runtime::Union{Nothing,LiveRuntime}=nothing,
    phase_e_data::Vector{UInt8}=UInt8[],
    phase_i_data::Vector{UInt8}=UInt8[],
)
    scale_lo, scale_hi = _activity_scale_bounds!(runtime, activity)
    bytes, lo, hi = _activity_bytes(activity; lo=scale_lo, hi=scale_hi)
    retinal_bytes, _, _ = retinal_activity === activity ? (bytes, lo, hi) : _activity_bytes(retinal_activity; lo=lo, hi=hi)
    frame_ms = (time_ns() - frame_start_ns) / 1e6
    sim_ms = steps_per_frame * Float64(p.dt)
    ms_per_step = step_ms / steps_per_frame
    realtime_x = step_ms == 0 ? Inf : sim_ms / step_ms

    return LiveFrame(
        frame=Int(frame_idx),
        N=size(retinal_activity, 1),
        rows=size(activity, 1),
        cols=size(activity, 2),
        retinal_n=size(retinal_activity, 1),
        retinal_rows=size(retinal_activity, 1),
        retinal_cols=size(retinal_activity, 2),
        t=Float64(t),
        lo=lo,
        hi=hi,
        step_ms=step_ms,
        frame_ms=frame_ms,
        ms_per_step=ms_per_step,
        realtime_x=realtime_x,
        steps_per_frame=steps_per_frame,
        data=bytes,
        retinal_data=retinal_bytes,
        phase_count=min(length(phase_e_data), length(phase_i_data)),
        phase_e_data=phase_e_data,
        phase_i_data=phase_i_data,
    )
end

function _fill_coupled_views!(display, left_activity, right_activity)
    rows, cols = size(left_activity)
    @inbounds for col in 1:cols, row in 1:rows
        display[row, col] = left_activity[row, col]
        display[row, cols + col] = right_activity[row, col]
    end
    return display
end

function _fill_coupled_retinal_source!(retinal_source, left_activity, right_activity)
    rows, cols = size(left_activity)
    @inbounds for col in 1:cols, row in 1:rows
        retinal_source[row, col] = left_activity[row, col]
        retinal_source[rows + row, col] = right_activity[row, col]
    end
    return retinal_source
end

function _coupled_retinal_activity(left_activity, right_activity, retinal_source, geometry::FieldGeometry, config::LiveConfig)
    if geometry.kind == :double_sech
        return double_sech_retinal_transform(
            left_activity,
            right_activity,
            geometry;
            output_size=(config.N, config.N),
        )
    end

    _fill_coupled_retinal_source!(retinal_source, left_activity, right_activity)
    return retinal_transform(retinal_source; output_size=(config.N, config.N), angle_origin=Float32(pi / 2))
end

function _reset_throttle!(runtime::LiveRuntime)
    runtime.throttle_deadline_ns = 0
    return
end

function _throttle!(runtime::LiveRuntime, frame_start_ns::UInt64, frame_sim_ms::Float64)
    yield()
    speed = runtime.speed
    if speed <= 0
        _reset_throttle!(runtime)
        return
    end

    interval_ns = UInt64(max(0, round(Int, frame_sim_ms * 1e6 / speed)))
    now_ns = time_ns()
    previous_deadline = runtime.throttle_deadline_ns
    deadline_ns = if previous_deadline == 0 || now_ns > previous_deadline + max(interval_ns, UInt64(50_000_000))
        frame_start_ns + interval_ns
    else
        previous_deadline + interval_ns
    end

    if deadline_ns <= now_ns
        runtime.throttle_deadline_ns = now_ns
        return
    end

    version = runtime.control_version
    while true
        now_ns = time_ns()
        remaining_ns = deadline_ns > now_ns ? deadline_ns - now_ns : UInt64(0)
        remaining_ns <= 1_000_000 && break
        runtime.control_version == version || begin
            runtime.throttle_deadline_ns = time_ns()
            return
        end
        sleep(min(Float64(remaining_ns) / 1e9, 0.05))
    end

    runtime.throttle_deadline_ns = deadline_ns
    return
end

function _stream_cpu_frames(callback::Function, config::LiveConfig, runtime::LiveRuntime)
    _uses_two_hemispheres(config) && return _stream_cpu_coupled_frames(callback, config, runtime)
    p = _live_model_params(config)
    geometry = _live_field_geometry(config)
    rng = _rng(config.seed)
    Ue = rand(rng, Float32, geometry.rows, geometry.cols)
    Ui = rand(rng, Float32, geometry.rows, geometry.cols)
    apply_field_mask!(Ue, Ui, geometry)
    Ke = generate_gaussian_kernel(config.Se, geometry.rows; dtype=Float32)
    Ki = generate_gaussian_kernel(config.Si, geometry.rows; dtype=Float32)
    excitatory_convolver = FFTConvolver(Ke, Ue)
    inhibitory_convolver = FFTConvolver(Ki, Ui)
    Uec = similar(Ue)
    Uic = similar(Ui)
    noise = Array{Float32}(undef, 2, geometry.rows, geometry.cols)

    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        steps_per_frame = _steps_per_frame(runtime.target_fps, p)
        frame_idx += 1
        frame_start = time_ns()
        step_start = time_ns()
        for _ in 1:steps_per_frame
            t_step = Float32(step_idx) * p.dt
            randn!(rng, noise)
            _step!(
                Ue,
                Ui,
                Uec,
                Uic,
                excitatory_convolver,
                inhibitory_convolver,
                noise,
                config.A,
                config.period,
                t_step,
                p,
                config.duty_cycle_percent,
            )
            apply_field_mask!(Ue, Ui, geometry)
            step_idx += 1
        end
        step_ms = (time_ns() - step_start) / 1e6
        activity = abs.(Ue .- Ui)
        retinal_activity = retinal_transform(activity; output_size=(config.N, config.N))
        phase_e_data, phase_i_data = _phase_bytes(Ue, Ui; mask=has_field_mask(geometry) ? geometry.mask : nothing)
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(
            activity,
            frame_idx,
            t,
            step_ms,
            frame_start,
            steps_per_frame,
            p,
            retinal_activity;
            runtime=runtime,
            phase_e_data=phase_e_data,
            phase_i_data=phase_i_data,
        )
        callback(frame) === false && break
        _throttle!(runtime, frame_start, Float64(steps_per_frame) * Float64(p.dt))
    end

    return frame_idx
end

function _stream_cpu_coupled_frames(callback::Function, config::LiveConfig, runtime::LiveRuntime)
    p = _live_model_params(config)
    geometry = _live_field_geometry(config)
    rng = _rng(config.seed)
    rows = geometry.rows
    cols = geometry.cols
    Ue_left = rand(rng, Float32, rows, cols)
    Ui_left = rand(rng, Float32, rows, cols)
    Ue_right = rand(rng, Float32, rows, cols)
    Ui_right = rand(rng, Float32, rows, cols)
    apply_field_mask!(Ue_left, Ui_left, geometry)
    apply_field_mask!(Ue_right, Ui_right, geometry)
    Ke = generate_gaussian_kernel(config.Se, rows; dtype=Float32)
    Ki = generate_gaussian_kernel(config.Si, rows; dtype=Float32)

    convolver_e_left = FFTConvolver(Ke, Ue_left)
    convolver_i_left = FFTConvolver(Ki, Ui_left)
    convolver_e_right = FFTConvolver(Ke, Ue_right)
    convolver_i_right = FFTConvolver(Ki, Ui_right)
    Uec_left = similar(Ue_left)
    Uic_left = similar(Ui_left)
    Uec_right = similar(Ue_right)
    Uic_right = similar(Ui_right)
    noise_left = Array{Float32}(undef, 2, rows, cols)
    noise_right = Array{Float32}(undef, 2, rows, cols)
    activity_left = similar(Ue_left)
    activity_right = similar(Ue_right)
    display_activity = Matrix{Float32}(undef, rows, 2cols)
    retinal_source = Matrix{Float32}(undef, 2rows, cols)
    border_coupling_mask = has_field_mask(geometry) ?
        field_border_mask(geometry.mask, max(1, div(config.overlap_rows, 2))) : nothing

    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        steps_per_frame = _steps_per_frame(runtime.target_fps, p)
        frame_idx += 1
        frame_start = time_ns()
        step_start = time_ns()
        for _ in 1:steps_per_frame
            t_step = Float32(step_idx) * p.dt
            randn!(rng, noise_left)
            randn!(rng, noise_right)
            _step!(
                Ue_left,
                Ui_left,
                Uec_left,
                Uic_left,
                convolver_e_left,
                convolver_i_left,
                noise_left,
                config.A,
                config.period,
                t_step,
                p,
                config.duty_cycle_percent,
            )
            _step!(
                Ue_right,
                Ui_right,
                Uec_right,
                Uic_right,
                convolver_e_right,
                convolver_i_right,
                noise_right,
                config.A,
                config.period,
                t_step,
                p,
                config.duty_cycle_percent,
            )
            apply_field_mask!(Ue_left, Ui_left, geometry)
            apply_field_mask!(Ue_right, Ui_right, geometry)
            if _uses_overlap_coupling(config)
                if has_field_mask(geometry)
                    _apply_border_coupling!(
                        Ue_left,
                        Ui_left,
                        Ue_right,
                        Ui_right,
                        border_coupling_mask,
                        config.coupling_strength,
                    )
                else
                    _apply_midline_coupling!(
                        Ue_left,
                        Ui_left,
                        Ue_right,
                        Ui_right,
                        config.coupling_strength,
                        config.overlap_rows,
                    )
                end
                apply_field_mask!(Ue_left, Ui_left, geometry)
                apply_field_mask!(Ue_right, Ui_right, geometry)
            end
            step_idx += 1
        end
        step_ms = (time_ns() - step_start) / 1e6
        @. activity_left = abs(Ue_left - Ui_left)
        @. activity_right = abs(Ue_right - Ui_right)
        _fill_coupled_views!(display_activity, activity_left, activity_right)
        retinal_activity = _coupled_retinal_activity(activity_left, activity_right, retinal_source, geometry, config)
        phase_e_data, phase_i_data = _phase_bytes(
            Ue_left,
            Ui_left,
            Ue_right,
            Ui_right;
            mask=has_field_mask(geometry) ? geometry.mask : nothing,
        )
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(
            display_activity,
            frame_idx,
            t,
            step_ms,
            frame_start,
            steps_per_frame,
            p,
            retinal_activity;
            runtime=runtime,
            phase_e_data=phase_e_data,
            phase_i_data=phase_i_data,
        )
        callback(frame) === false && break
        _throttle!(runtime, frame_start, Float64(steps_per_frame) * Float64(p.dt))
    end

    return frame_idx
end

function _stream_metal_frames(callback::Function, config::LiveConfig, runtime::LiveRuntime)
    _uses_two_hemispheres(config) && return _stream_metal_coupled_frames(callback, config, runtime)
    Metal.functional() || throw(ErrorException("Metal.jl is not functional on this machine."))
    p = _live_model_params(config)
    geometry = _live_field_geometry(config)
    config.seed === nothing || Metal.seed!(config.seed)

    Ue = Metal.rand(Float32, geometry.rows, geometry.cols)
    Ui = Metal.rand(Float32, geometry.rows, geometry.cols)
    mask_gpu = has_field_mask(geometry) ? Metal.MtlArray(geometry.mask_float32) : nothing
    mask_gpu === nothing || apply_field_mask!(Ue, Ui, mask_gpu, config.gpu_threads)
    excitatory_convolver = if config.convolution == :fft
        Ke = generate_gaussian_kernel(config.Se, geometry.rows; dtype=Float32)
        MetalFFTConvolver(Ke, Ue)
    else
        MetalSeparableConvolver(config.Se, Ue; cutoff=config.kernel_cutoff)
    end
    inhibitory_convolver = if config.convolution == :fft
        Ki = generate_gaussian_kernel(config.Si, geometry.rows; dtype=Float32)
        MetalFFTConvolver(Ki, Ui)
    else
        MetalSeparableConvolver(config.Si, Ui; cutoff=config.kernel_cutoff)
    end

    Uec = similar(Ue)
    Uic = similar(Ui)
    noise_E = similar(Ue)
    noise_I = similar(Ui)
    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        steps_per_frame = _steps_per_frame(runtime.target_fps, p)
        frame_idx += 1
        frame_start = time_ns()
        step_start = time_ns()
        for _ in 1:steps_per_frame
            t_step = Float32(step_idx) * p.dt
            Metal.randn!(noise_E)
            Metal.randn!(noise_I)
            if config.convolution == :fft
                _metal_step!(
                    Ue,
                    Ui,
                    Uec,
                    Uic,
                    excitatory_convolver,
                    inhibitory_convolver,
                    noise_E,
                    noise_I,
                    config.A,
                    config.period,
                    t_step,
                    p,
                    config.gpu_threads,
                    config.duty_cycle_percent,
                )
            else
                _metal_step_separable!(
                    Ue,
                    Ui,
                    Uec,
                    Uic,
                    excitatory_convolver,
                    inhibitory_convolver,
                    noise_E,
                    noise_I,
                    config.A,
                    config.period,
                    t_step,
                    p,
                    config.gpu_threads,
                    config.boundary_x,
                    config.boundary_y,
                    config.partial_reflect_strength,
                    config.duty_cycle_percent,
                )
            end
            mask_gpu === nothing || apply_field_mask!(Ue, Ui, mask_gpu, config.gpu_threads)
            step_idx += 1
        end
        Metal.synchronize()
        step_ms = (time_ns() - step_start) / 1e6

        Metal.synchronize()
        phase_e = Array(Ue)
        phase_i = Array(Ui)
        activity = abs.(phase_e .- phase_i)
        retinal_activity = retinal_transform(activity; output_size=(config.N, config.N))
        phase_e_data, phase_i_data = _phase_bytes(phase_e, phase_i; mask=has_field_mask(geometry) ? geometry.mask : nothing)
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(
            activity,
            frame_idx,
            t,
            step_ms,
            frame_start,
            steps_per_frame,
            p,
            retinal_activity;
            runtime=runtime,
            phase_e_data=phase_e_data,
            phase_i_data=phase_i_data,
        )
        callback(frame) === false && break
        _throttle!(runtime, frame_start, Float64(steps_per_frame) * Float64(p.dt))
    end

    return frame_idx
end

function _stream_metal_coupled_frames(callback::Function, config::LiveConfig, runtime::LiveRuntime)
    Metal.functional() || throw(ErrorException("Metal.jl is not functional on this machine."))
    p = _live_model_params(config)
    geometry = _live_field_geometry(config)
    config.seed === nothing || Metal.seed!(config.seed)
    rows = geometry.rows
    cols = geometry.cols

    Ue_left = Metal.rand(Float32, rows, cols)
    Ui_left = Metal.rand(Float32, rows, cols)
    Ue_right = Metal.rand(Float32, rows, cols)
    Ui_right = Metal.rand(Float32, rows, cols)
    mask_gpu = has_field_mask(geometry) ? Metal.MtlArray(geometry.mask_float32) : nothing
    border_mask_gpu = has_field_mask(geometry) ?
        Metal.MtlArray(Float32.(field_border_mask(geometry.mask, max(1, div(config.overlap_rows, 2))))) : nothing
    if mask_gpu !== nothing
        apply_field_mask!(Ue_left, Ui_left, mask_gpu, config.gpu_threads)
        apply_field_mask!(Ue_right, Ui_right, mask_gpu, config.gpu_threads)
    end

    if config.convolution == :fft
        Ke = generate_gaussian_kernel(config.Se, rows; dtype=Float32)
        Ki = generate_gaussian_kernel(config.Si, rows; dtype=Float32)
        convolver_e_left = MetalFFTConvolver(Ke, Ue_left)
        convolver_i_left = MetalFFTConvolver(Ki, Ui_left)
        convolver_e_right = MetalFFTConvolver(Ke, Ue_right)
        convolver_i_right = MetalFFTConvolver(Ki, Ui_right)
    else
        convolver_e_left = MetalSeparableConvolver(config.Se, Ue_left; cutoff=config.kernel_cutoff)
        convolver_i_left = MetalSeparableConvolver(config.Si, Ui_left; cutoff=config.kernel_cutoff)
        convolver_e_right = MetalSeparableConvolver(config.Se, Ue_right; cutoff=config.kernel_cutoff)
        convolver_i_right = MetalSeparableConvolver(config.Si, Ui_right; cutoff=config.kernel_cutoff)
    end

    Uec_left = similar(Ue_left)
    Uic_left = similar(Ui_left)
    Uec_right = similar(Ue_right)
    Uic_right = similar(Ui_right)
    noise_E_left = similar(Ue_left)
    noise_I_left = similar(Ui_left)
    noise_E_right = similar(Ue_right)
    noise_I_right = similar(Ui_right)
    activity_left = Matrix{Float32}(undef, rows, cols)
    activity_right = Matrix{Float32}(undef, rows, cols)
    display_activity = Matrix{Float32}(undef, rows, 2cols)
    retinal_source = Matrix{Float32}(undef, 2rows, cols)

    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        steps_per_frame = _steps_per_frame(runtime.target_fps, p)
        frame_idx += 1
        frame_start = time_ns()
        step_start = time_ns()
        for _ in 1:steps_per_frame
            t_step = Float32(step_idx) * p.dt
            Metal.randn!(noise_E_left)
            Metal.randn!(noise_I_left)
            Metal.randn!(noise_E_right)
            Metal.randn!(noise_I_right)

            if config.convolution == :fft
                _metal_step!(
                    Ue_left,
                    Ui_left,
                    Uec_left,
                    Uic_left,
                    convolver_e_left,
                    convolver_i_left,
                    noise_E_left,
                    noise_I_left,
                    config.A,
                    config.period,
                    t_step,
                    p,
                    config.gpu_threads,
                    config.duty_cycle_percent,
                )
                _metal_step!(
                    Ue_right,
                    Ui_right,
                    Uec_right,
                    Uic_right,
                    convolver_e_right,
                    convolver_i_right,
                    noise_E_right,
                    noise_I_right,
                    config.A,
                    config.period,
                    t_step,
                    p,
                    config.gpu_threads,
                    config.duty_cycle_percent,
                )
            else
                _metal_step_separable!(
                    Ue_left,
                    Ui_left,
                    Uec_left,
                    Uic_left,
                    convolver_e_left,
                    convolver_i_left,
                    noise_E_left,
                    noise_I_left,
                    config.A,
                    config.period,
                    t_step,
                    p,
                    config.gpu_threads,
                    config.boundary_x,
                    config.boundary_y,
                    config.partial_reflect_strength,
                    config.duty_cycle_percent,
                )
                _metal_step_separable!(
                    Ue_right,
                    Ui_right,
                    Uec_right,
                    Uic_right,
                    convolver_e_right,
                    convolver_i_right,
                    noise_E_right,
                    noise_I_right,
                    config.A,
                    config.period,
                    t_step,
                    p,
                    config.gpu_threads,
                    config.boundary_x,
                    config.boundary_y,
                    config.partial_reflect_strength,
                    config.duty_cycle_percent,
                )
            end
            if mask_gpu !== nothing
                apply_field_mask!(Ue_left, Ui_left, mask_gpu, config.gpu_threads)
                apply_field_mask!(Ue_right, Ui_right, mask_gpu, config.gpu_threads)
            end

            if _uses_overlap_coupling(config)
                if border_mask_gpu !== nothing
                    apply_border_coupling!(
                        Ue_left,
                        Ui_left,
                        Ue_right,
                        Ui_right,
                        border_mask_gpu;
                        strength=config.coupling_strength,
                        gpu_threads=config.gpu_threads,
                    )
                else
                    apply_midline_coupling!(
                        Ue_left,
                        Ui_left,
                        Ue_right,
                        Ui_right;
                        strength=config.coupling_strength,
                        overlap_rows=config.overlap_rows,
                        gpu_threads=config.gpu_threads,
                    )
                end
                if mask_gpu !== nothing
                    apply_field_mask!(Ue_left, Ui_left, mask_gpu, config.gpu_threads)
                    apply_field_mask!(Ue_right, Ui_right, mask_gpu, config.gpu_threads)
                end
            end
            step_idx += 1
        end
        Metal.synchronize()
        step_ms = (time_ns() - step_start) / 1e6

        Metal.synchronize()
        phase_e_left = Array(Ue_left)
        phase_i_left = Array(Ui_left)
        phase_e_right = Array(Ue_right)
        phase_i_right = Array(Ui_right)
        @. activity_left = abs(phase_e_left - phase_i_left)
        @. activity_right = abs(phase_e_right - phase_i_right)
        _fill_coupled_views!(display_activity, activity_left, activity_right)
        retinal_activity = _coupled_retinal_activity(activity_left, activity_right, retinal_source, geometry, config)
        phase_e_data, phase_i_data = _phase_bytes(
            phase_e_left,
            phase_i_left,
            phase_e_right,
            phase_i_right;
            mask=has_field_mask(geometry) ? geometry.mask : nothing,
        )
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(
            display_activity,
            frame_idx,
            t,
            step_ms,
            frame_start,
            steps_per_frame,
            p,
            retinal_activity;
            runtime=runtime,
            phase_e_data=phase_e_data,
            phase_i_data=phase_i_data,
        )
        callback(frame) === false && break
        _throttle!(runtime, frame_start, Float64(steps_per_frame) * Float64(p.dt))
    end

    return frame_idx
end

function stream_live_frames(callback::Function, config::LiveConfig=LiveConfig(), runtime::Union{Nothing,LiveRuntime}=nothing)
    normalized = normalize_live_config(config)
    live_runtime = runtime === nothing ? _live_runtime(normalized) : runtime
    if normalized.backend == :metal
        return _stream_metal_frames(callback, normalized, live_runtime)
    else
        return _stream_cpu_frames(callback, normalized, live_runtime)
    end
end

function _json_string(value)
    escaped = replace(
        string(value),
        "\\" => "\\\\",
        "\"" => "\\\"",
        "\n" => "\\n",
        "\r" => "\\r",
    )
    return "\"$(escaped)\""
end

function _json_number(value; digits=4)
    value === nothing && return "null"
    if value isa AbstractFloat
        isfinite(value) || return "null"
        return string(round(value; digits=digits))
    end
    return string(value)
end

function _hello_json(config::LiveConfig)
    return string(
        "{\"type\":\"hello\"",
        ",\"N\":", config.N,
        ",\"backend\":", _json_string(config.backend),
        ",\"conv\":", _json_string(config.convolution),
        ",\"fps\":", config.target_fps,
        ",\"speed\":", _json_number(config.speed),
        ",\"A\":", _json_number(config.A),
        ",\"T\":", _json_number(config.period),
        ",\"dutyCycle\":", _json_number(config.duty_cycle_percent),
        ",\"Se\":", _json_number(config.Se),
        ",\"Si\":", _json_number(config.Si),
        ",\"dt\":", _json_number(config.dt),
        ",\"kernelCutoff\":", _json_number(config.kernel_cutoff),
        ",\"boundaryX\":", _json_string(config.boundary_x),
        ",\"boundaryY\":", _json_string(config.boundary_y),
        ",\"partialReflectStrength\":", _json_number(config.partial_reflect_strength; digits=3),
        ",\"coupling\":", _json_string(config.coupling),
        ",\"couplingStrength\":", _json_number(config.coupling_strength; digits=5),
        ",\"overlapRows\":", config.overlap_rows,
        ",\"fieldGeometry\":", _json_string(config.field_geometry),
        ",\"fieldDensity\":", _json_number(config.field_density; digits=3),
        ",\"activityScale\":", _json_string(config.activity_scale),
        "}",
    )
end

function _frame_json(frame::LiveFrame)
    return string(
        "{\"type\":\"frame\"",
        ",\"frame\":", frame.frame,
        ",\"N\":", frame.N,
        ",\"rows\":", frame.rows,
        ",\"cols\":", frame.cols,
        ",\"retinalN\":", frame.retinal_n,
        ",\"retinalRows\":", frame.retinal_rows,
        ",\"retinalCols\":", frame.retinal_cols,
        ",\"t\":", _json_number(frame.t),
        ",\"min\":", _json_number(frame.lo; digits=6),
        ",\"max\":", _json_number(frame.hi; digits=6),
        ",\"stepMs\":", _json_number(frame.step_ms),
        ",\"frameMs\":", _json_number(frame.frame_ms),
        ",\"msPerStep\":", _json_number(frame.ms_per_step; digits=5),
        ",\"realtimeX\":", _json_number(frame.realtime_x),
        ",\"stepsPerFrame\":", frame.steps_per_frame,
        ",\"data\":", _json_string(base64encode(frame.data)),
        ",\"retinalData\":", _json_string(base64encode(frame.retinal_data)),
        ",\"phaseCount\":", frame.phase_count,
        ",\"phaseEData\":", _json_string(base64encode(frame.phase_e_data)),
        ",\"phaseIData\":", _json_string(base64encode(frame.phase_i_data)),
        "}",
    )
end

function _done_json(frames)
    return "{\"type\":\"done\",\"frames\":$(frames)}"
end

function _error_json(err)
    return string("{\"type\":\"error\",\"message\":", _json_string(sprint(showerror, err)), "}")
end

function _safe_ws_send(ws, message)
    try
        HTTP.WebSockets.send(ws, message)
        return true
    catch err
        if err isa HTTP.WebSockets.WebSocketError || err isa EOFError || err isa IOError
            return false
        end
        rethrow()
    end
end

function _apply_visual_control!(runtime::LiveRuntime, message::AbstractString)
    prefix = "visual:"
    startswith(message, prefix) || return false
    payload = ncodeunits(message) == ncodeunits(prefix) ? "" :
        message[nextind(message, firstindex(message), ncodeunits(prefix)):lastindex(message)]
    isempty(payload) && return true

    for part in split(payload, '&')
        isempty(part) && continue
        pieces = split(part, '='; limit=2)
        length(pieces) == 2 || continue
        key, value = pieces
        try
            if key == "fps"
                next_fps = max(1, parse(Int, value))
                if next_fps != runtime.target_fps
                    runtime.target_fps = next_fps
                    runtime.control_version += 1
                end
            elseif key == "speed"
                speed = parse(Float64, value)
                if speed >= 0 && speed != runtime.speed
                    runtime.speed = speed
                    runtime.control_version += 1
                end
            elseif key == "activity_scale"
                scale = _normalize_activity_scale(Symbol(lowercase(value)))
                if scale != runtime.activity_scale
                    runtime.activity_scale = scale
                    runtime.control_version += 1
                end
            end
        catch
            continue
        end
    end
    return true
end

function _stream_websocket(ws, params)
    paused = Ref(false)
    closed = Ref(false)
    config = live_config_from_query(params)
    runtime = _live_runtime(config)
    control_task = @async try
        while !closed[]
            message = String(HTTP.WebSockets.receive(ws))
            if message == "pause"
                paused[] = true
                runtime.control_version += 1
                _reset_throttle!(runtime)
            elseif message == "play"
                paused[] = false
                runtime.control_version += 1
                _reset_throttle!(runtime)
            elseif message == "close"
                closed[] = true
                break
            else
                _apply_visual_control!(runtime, message)
            end
        end
    catch err
        if !(err isa HTTP.WebSockets.WebSocketError || err isa EOFError || err isa IOError)
            @warn "Applet control channel failed." exception=(err, catch_backtrace())
        end
        closed[] = true
    end

    try
        _safe_ws_send(ws, _hello_json(config)) || return
        frames = stream_live_frames(config, runtime) do frame
            while paused[] && !closed[]
                sleep(0.03)
            end
            closed[] && return false
            _safe_ws_send(ws, _frame_json(frame))
        end
        closed[] || _safe_ws_send(ws, _done_json(frames))
    catch err
        _safe_ws_send(ws, _error_json(err))
    finally
        closed[] = true
    end
    return
end

function _request_uri(stream)
    return HTTP.URI(String(stream.message.target))
end

function _write_response(stream, status::Integer, content_type::AbstractString, body::AbstractString)
    HTTP.setstatus(stream, status)
    HTTP.setheader(stream, "Content-Type", content_type)
    HTTP.setheader(stream, "Cache-Control", "no-store")
    HTTP.startwrite(stream)
    write(stream, body)
    return
end

function _handle_applet_stream(stream)
    uri = _request_uri(stream)
    if HTTP.WebSockets.isupgrade(stream.message) && uri.path == "/stream"
        HTTP.WebSockets.upgrade(stream; check_origin=nothing) do ws
            _stream_websocket(ws, HTTP.queryparams(uri))
        end
    elseif uri.path == "/" || uri.path == "/index.html"
        _write_response(stream, 200, "text/html; charset=utf-8", APPLET_HTML)
    elseif uri.path == "/health"
        _write_response(stream, 200, "text/plain; charset=utf-8", "ok")
    else
        _write_response(stream, 404, "text/plain; charset=utf-8", "not found")
    end
    return
end

function _display_host(host)
    return host in ("0.0.0.0", "::") ? "127.0.0.1" : host
end

function applet_url(server, host)
    return "http://$(_display_host(host)):$(HTTP.port(server))/"
end

function serve_applet_async(;
    host::AbstractString="127.0.0.1",
    port::Integer=8088,
    verbose::Bool=true,
)
    server = HTTP.listen!(host, port; listenany=true) do stream
        _handle_applet_stream(stream)
    end
    if verbose
        println("RSE real-time applet: ", applet_url(server, host))
        println("Press Ctrl-C to stop the server.")
    end
    return server
end

function serve_applet(;
    host::AbstractString="127.0.0.1",
    port::Integer=8088,
    verbose::Bool=true,
)
    server = serve_applet_async(host=host, port=port, verbose=verbose)
    wait(server)
    return server
end

const APPLET_HTML = raw"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Real-time Strobe Hallucination Simulator</title>
  <style>
    :root {
      --bg: #edf5f7;
      --panel: rgba(255, 255, 255, 0.82);
      --panel-strong: #ffffff;
      --ink: #0d2638;
      --muted: #607284;
      --accent: #009eaa;
      --accent-2: #f3b33d;
      --danger: #db4d54;
      --line: #dbe7ef;
      --line-strong: #c6d8e2;
      --shadow: 0 24px 70px rgba(25, 56, 82, 0.13);
      --soft-shadow: 0 14px 38px rgba(25, 56, 82, 0.08);
      --legend-gradient: linear-gradient(90deg, #0d0887, #5403a0, #8b0aa5, #b93289, #db5c68, #f48849, #feba2c, #f0f921);
      font-family: "IBM Plex Sans", "Aptos", "Helvetica Neue", sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      min-height: 100vh;
      background:
        linear-gradient(rgba(13, 38, 56, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(13, 38, 56, 0.035) 1px, transparent 1px),
        radial-gradient(circle at 9% 10%, rgba(0, 158, 170, 0.19), transparent 32rem),
        radial-gradient(circle at 84% 4%, rgba(243, 179, 61, 0.22), transparent 27rem),
        radial-gradient(circle at 72% 88%, rgba(13, 38, 56, 0.08), transparent 38rem),
        linear-gradient(135deg, #f8fbfd 0%, #eef7f8 45%, #f8f3e4 100%);
      background-size: 30px 30px, 30px 30px, auto, auto, auto, auto;
    }

    main {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 28px;
      display: grid;
      grid-template-columns: 350px 1fr;
      gap: 22px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 3.4vw, 46px);
      letter-spacing: -0.06em;
      line-height: 0.95;
      color: #092337;
    }

    h2 {
      margin: 0;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--accent);
    }

    .panel, .stage {
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(20px);
      border-radius: 26px;
      box-shadow: var(--shadow);
    }

    .panel {
      padding: 20px;
      align-self: start;
      position: sticky;
      top: 18px;
    }

    .subtitle {
      color: var(--muted);
      margin: 0 0 18px;
      line-height: 1.45;
      font-size: 14px;
    }

    .key-hints {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .key {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 3px 8px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      box-shadow: var(--soft-shadow);
    }

    .control-section {
      margin-top: 14px;
      padding: 12px;
      border: 1px solid rgba(198, 216, 226, 0.78);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.55);
    }

    .section-title {
      margin-bottom: 10px;
      color: #0b3146;
      font-size: 11px;
      font-weight: 900;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }

    .control-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .preset-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .preset-button {
      min-height: 64px;
      padding: 9px 10px;
      color: #0b3146;
      background:
        radial-gradient(circle at 95% 5%, rgba(0, 158, 170, 0.13), transparent 4rem),
        #ffffff;
      border: 1px solid var(--line);
      border-radius: 14px;
      text-align: left;
      box-shadow: none;
    }

    .preset-button strong {
      display: block;
      margin-bottom: 3px;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .preset-button span {
      display: block;
      color: var(--muted);
      font-size: 10px;
      line-height: 1.25;
    }

    .param-output {
      margin: 10px 0 0;
      max-height: 190px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid rgba(13, 38, 56, 0.18);
      border-radius: 14px;
      background: #071824;
      color: #dff8f8;
      padding: 10px;
      font: 10.5px/1.45 "IBM Plex Mono", "SFMono-Regular", monospace;
    }

    .wide {
      grid-column: 1 / -1;
    }

    .hidden-control {
      display: none !important;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.04em;
    }

    label sub {
      font-size: 0.72em;
      line-height: 0;
      vertical-align: sub;
    }

    input, select, button {
      width: 100%;
      border: 1px solid var(--line-strong);
      border-radius: 14px;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.74);
      padding: 10px 11px;
      font: inherit;
      outline: none;
    }

    input[type="checkbox"] {
      width: auto;
      accent-color: var(--accent);
    }

    .check-row {
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 8px 0 0;
      color: var(--ink);
    }

    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 18px;
    }

    button {
      cursor: pointer;
      background: linear-gradient(135deg, #0b3146, #006f7c);
      color: white;
      border: 0;
      font-weight: 800;
      letter-spacing: 0.02em;
    }

    button.secondary {
      color: #0b3146;
      background: #ffffff;
      border: 1px solid var(--line);
    }

    button.pause {
      background: linear-gradient(135deg, var(--accent), #32c5b8);
    }

    button.paused {
      background: linear-gradient(135deg, var(--accent-2), #ffcf70);
      color: #2d2108;
    }

    .section-actions {
      margin-top: 10px;
    }

    .status {
      margin-top: 0;
      padding: 12px;
      border-radius: 18px;
      background: #f7fbfc;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      min-height: 44px;
    }

    .stage {
      padding: 18px;
      display: grid;
      gap: 14px;
      min-width: 0;
      align-content: start;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      align-items: start;
      grid-auto-rows: 38px;
    }

    .metric {
      border: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(246, 251, 252, 0.92));
      border-radius: 12px;
      padding: 5px 8px;
      min-width: 0;
      height: 38px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      box-shadow: var(--soft-shadow);
    }

    .metric span {
      color: var(--muted);
      font-size: 8.5px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .metric strong {
      font-size: clamp(12px, 1vw, 15px);
      letter-spacing: -0.045em;
      white-space: nowrap;
      text-align: right;
    }

    .views {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      min-width: 0;
      align-items: start;
    }

    .view {
      border: 1px solid var(--line);
      border-radius: 24px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(247, 251, 252, 0.96));
      padding: 12px;
      min-width: 0;
      box-shadow: var(--soft-shadow);
    }

    .view-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 8px;
    }

    .view-title {
      font-size: 14px;
      font-weight: 800;
      letter-spacing: 0.02em;
    }

    .view-note {
      color: var(--muted);
      font-size: 12px;
    }

    canvas {
      display: block;
      width: 100%;
      border-radius: 0;
      image-rendering: pixelated;
      background: #071018;
      box-shadow: inset 0 0 0 1px rgba(13, 38, 56, 0.12);
    }

    .view canvas {
      aspect-ratio: 1 / 1;
    }

    .canvas-frame {
      position: relative;
      padding: 30px 36px;
      border-radius: 14px;
      background:
        radial-gradient(circle at 50% 18%, rgba(0, 158, 170, 0.07), transparent 16rem),
        linear-gradient(180deg, #fbfdfe, #f3f8fa);
      border: 1px solid rgba(198, 216, 226, 0.72);
    }

    .canvas-frame canvas {
      width: 100%;
    }

    .axis-label,
    .hemi-label,
    .ecc-label {
      position: absolute;
      z-index: 2;
      color: #33495c;
      font-size: 8px;
      font-weight: 800;
      letter-spacing: 0.04em;
      line-height: 1;
      pointer-events: none;
      text-transform: uppercase;
    }

    .axis-label {
      color: #486174;
      font-variant-numeric: tabular-nums;
      background: transparent;
      border: 0;
      padding: 0;
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.85);
    }

    .hemi-label {
      color: #0b3146;
      font-size: 9px;
      letter-spacing: 0.07em;
    }

    .ecc-label {
      color: #607284;
      font-size: 8px;
      text-transform: none;
      letter-spacing: 0.01em;
    }

    .cortical-frame {
      --left-center: 50%;
      --right-center: 50%;
      --left-label-y: 10px;
      --right-label-y: 10px;
      --left-top-y: 30px;
      --right-top-y: 30px;
      --left-bottom-y: calc(100% - 30px);
      --right-bottom-y: calc(100% - 30px);
      --left-ecc-y: 50%;
      --right-ecc-y: 50%;
      padding: 30px 56px 28px;
    }

    .hemi-left {
      left: var(--left-center);
      transform: translateX(-50%);
    }

    .hemi-right {
      left: var(--right-center);
      transform: translateX(-50%);
    }

    .hemi-left {
      top: var(--left-label-y);
    }

    .hemi-right {
      top: var(--right-label-y);
    }

    .axis-top-left {
      top: var(--left-top-y);
    }

    .axis-top-right {
      top: var(--right-top-y);
    }

    .axis-bottom-left {
      top: var(--left-bottom-y);
    }

    .axis-bottom-right {
      top: var(--right-bottom-y);
    }

    .axis-top-left,
    .axis-bottom-left,
    .axis-top-right,
    .axis-bottom-right {
      left: 18px;
      width: 24px;
      text-align: right;
      transform: translateY(-50%);
    }

    .axis-top-left::after,
    .axis-bottom-left::after,
    .axis-top-right::after,
    .axis-bottom-right::after {
      content: "";
      position: absolute;
      left: calc(100% + 5px);
      top: 50%;
      width: 9px;
      border-top: 1px solid rgba(72, 97, 116, 0.55);
    }

    .cortical-ecc {
      top: var(--left-ecc-y);
      transform: translateY(-50%);
    }

    .cortical-fovea-left,
    .cortical-fovea-right {
      left: 8px;
    }

    .cortical-periphery-left,
    .cortical-periphery-right {
      right: 7px;
    }

    .cortical-fovea-right,
    .cortical-periphery-right {
      top: var(--right-ecc-y);
    }

    .cortical-frame:not(.coupled) .cortical-fovea-right,
    .cortical-frame:not(.coupled) .cortical-periphery-right {
      display: none;
    }

    .cortical-frame.stacked {
      padding: 40px 56px 30px;
    }

    .cortical-frame:not(.coupled) .hemi-left,
    .cortical-frame:not(.coupled) .hemi-right,
    .cortical-frame:not(.coupled) .axis-top-right,
    .cortical-frame:not(.coupled) .axis-bottom-right {
      display: none;
    }

    .retinal-angle-90 {
      top: 12px;
      left: 50%;
      transform: translateX(-50%);
    }

    .retinal-angle-270 {
      bottom: 12px;
      left: 50%;
      transform: translateX(-50%);
    }

    .retinal-angle-0 {
      right: 10px;
      top: 50%;
      transform: translateY(-50%);
    }

    .retinal-angle-180 {
      left: 10px;
      top: 50%;
      transform: translateY(-50%);
    }

    .frame-card {
      padding: 13px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: #ffffff;
      box-shadow: var(--soft-shadow);
    }

    .frame-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 9px;
    }

    .frame-toolbar .view-title {
      flex: 1;
    }

    .frame-toolbar select {
      width: min(220px, 52%);
      padding: 8px 10px;
    }

    .frame-panel.hidden-control {
      display: none !important;
    }

    .kernel-canvas {
      aspect-ratio: 2.15 / 1;
      background: #f8fbfd;
      image-rendering: auto;
    }

    .stimulus-canvas {
      height: 82px;
      background: #f8fbfd;
      image-rendering: auto;
    }

    .field-canvas {
      aspect-ratio: 2.2 / 1;
      background: #f8fbfd;
      image-rendering: auto;
    }

    .phase-canvas {
      aspect-ratio: 1 / 1;
      background: #f8fbfd;
      image-rendering: auto;
    }

    .phase-options {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      margin: 0 0 8px;
    }

    .phase-options label {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.03em;
    }

    .tiny-note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }

    .legend {
      height: 9px;
      border-radius: 0;
      background: var(--legend-gradient);
      border: 1px solid var(--line);
      box-shadow: var(--soft-shadow);
    }

    .legend-wrap {
      width: min(460px, 100%);
      margin: 0 auto;
      display: grid;
      grid-template-columns: auto minmax(140px, 1fr) auto;
      grid-template-areas:
        "label label label"
        "low bar high";
      gap: 5px 8px;
      align-items: center;
      color: var(--muted);
      font-size: 10px;
      font-variant-numeric: tabular-nums;
    }

    .legend-label {
      grid-area: label;
      justify-self: center;
      color: #0b3146;
      font-size: 10px;
      font-weight: 900;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .legend-low { grid-area: low; }
    .legend-high { grid-area: high; }
    .legend-wrap .legend { grid-area: bar; }

    @media (max-width: 940px) {
      main { grid-template-columns: 1fr; padding: 18px; }
      .panel { position: static; }
      .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .views { grid-template-columns: 1fr; }
    }

    @media (max-width: 1180px) {
      .views { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h2>Rule-Ermentrout-Stroffegen</h2>
      <h1>Real-time Strobe Hallucination Simulator</h1>
      <p class="subtitle key-hints"><span class="key">Space</span> pause/play <span class="key">Enter</span> reset/apply</p>

      <div class="control-section">
        <div class="section-title">Visualization</div>
        <div class="control-grid">
          <label>FPS<input id="fps" type="number" min="1" max="60" step="1" value="30"></label>
          <label>Speed x<input id="speed" type="number" min="0.1" max="10" step="0.1" value="1"></label>
          <label id="maxSpeedControl" class="check-row"><input id="maxSpeed" type="checkbox"> Max speed</label>
          <label class="wide">Colormap<select id="colorMap"><option value="plasma">plasma</option><option value="viridis">viridis</option><option value="magma">magma</option><option value="inferno">inferno</option><option value="cividis">cividis</option><option value="turbo">turbo</option><option value="nipy_spectral">nipy_spectral</option><option value="gray">gray</option></select></label>
          <label class="wide">Activity scale<select id="activityScale"><option value="frame">frame min/max</option><option value="simulation">simulation min/max</option></select></label>
        </div>
      </div>

      <div class="control-section">
        <div class="section-title">Backend Implementation</div>
        <div class="control-grid">
          <label>Backend<select id="backend"><option value="metal">GPU (Metal)</option><option value="cpu">CPU</option></select></label>
          <label>Convolution<select id="conv"><option value="separable">separable</option><option value="fft">FFT</option></select></label>
          <label>Kernel cutoff<input id="kernelCutoff" type="number" min="0.5" max="6" step="0.25" value="3"></label>
          <label>Seed<input id="seed" type="number" step="1" placeholder="optional"></label>
        </div>
        <label id="fastNControl" class="check-row"><input id="fastN" type="checkbox" checked> Snap to FFT-friendly odd N</label>
      </div>

      <div class="control-section">
        <div class="section-title">Boundary</div>
        <div class="control-grid">
          <label id="boundaryControl" class="hidden-control">Boundary<select id="boundary"><option value="edge">edge</option><option value="zero">zero</option><option value="partial_reflect">partial reflect</option></select></label>
          <label id="boundaryXControl">Boundary X<select id="boundaryX"><option value="periodic">periodic</option><option value="edge">edge</option><option value="zero">zero</option><option value="partial_reflect">partial reflect</option></select></label>
          <label id="boundaryYControl">Boundary Y<select id="boundaryY"><option value="periodic">periodic</option><option value="edge">edge</option><option value="zero">zero</option><option value="partial_reflect">partial reflect</option></select></label>
          <label id="partialReflectControl" class="hidden-control">Reflect gain<input id="partialReflectStrength" type="number" min="0" max="1" step="0.05" value="0.5"></label>
        </div>
      </div>

      <div class="control-section">
        <div class="section-title">Coupling</div>
        <div class="control-grid">
          <label>Coupling<select id="coupling"><option value="off">none</option><option value="no_connection">no connection</option><option value="overlap">overlap</option></select></label>
          <label>Overlap rows<input id="overlapRows" type="number" min="2" step="2" value="6"></label>
          <label>Coupling g<input id="couplingStrength" type="number" min="0" max="0.5" step="0.005" value="0.02"></label>
        </div>
      </div>

      <div class="control-section">
        <div class="section-title">Strobe Parameters</div>
        <div class="control-grid">
          <label>Amplitude<input id="amp" type="number" min="0" step="0.05" value="0.7"></label>
          <label>Period (ms)<input id="period" type="number" min="1" step="1" value="115"></label>
          <label>Duty cycle (%)<input id="duty" type="number" min="1" max="99" step="0.5" value="20.5"></label>
        </div>
      </div>

      <div class="control-section">
        <div class="section-title">Neural Field Parameters</div>
        <div class="control-grid">
          <label class="wide">Field geometry<select id="fieldGeometry"><option value="square">square</option><option value="double_sech">double-sech V1</option></select></label>
          <label id="fieldDensityControl" class="hidden-control">Field density<input id="fieldDensity" type="number" min="0.25" max="3" step="0.25" value="1"></label>
          <label id="nControl">N<input id="n" type="number" min="5" step="2" value="81"></label>
          <label><span>&sigma;<sub>e</sub></span><input id="se" type="number" min="0.1" step="0.05" value="2"></label>
          <label><span>&sigma;<sub>i</sub></span><input id="si" type="number" min="0.1" step="0.05" value="5"></label>
          <label>dt (ms)<input id="dt" type="number" min="0.01" step="0.05" value="0.2"></label>
        </div>
      </div>

      <div class="control-section">
        <div class="section-title">Selected Parameters</div>
        <div class="preset-grid">
          <button class="preset-button" data-preset="default"><strong>Default</strong><span>Boundary periodic, coupling none, kernel 3, dt 0.2</span></button>
          <button class="preset-button" data-preset="p1"><strong>1</strong><span>Zig-zag square grid<br>N64 A0.2 T55</span></button>
          <button class="preset-button" data-preset="p2"><strong>2</strong><span>Square grid<br>N81 A0.7 T120</span></button>
          <button class="preset-button" data-preset="p3"><strong>3</strong><span>Lines and dots<br>N81 A0.5 T125</span></button>
          <button class="preset-button" data-preset="p4"><strong>4</strong><span>Hex grid rings<br>N81 A0.5 T115</span></button>
        </div>
        <div class="section-actions">
          <button id="printParams" class="secondary">Print parameters</button>
        </div>
        <pre id="paramOutput" class="param-output">Click Print parameters to write the current settings here.</pre>
      </div>
      <div class="button-row">
        <button id="pausePlay" class="pause">Pause</button>
        <button id="reset" class="secondary">Reset</button>
      </div>
    </section>

    <section class="stage">
      <div class="metrics">
        <div class="metric"><span>Sim time</span><strong id="simTime">0 ms</strong></div>
        <div class="metric"><span>Stream FPS</span><strong id="streamFps">0</strong></div>
        <div class="metric"><span>ms / step</span><strong id="msStep">0</strong></div>
        <div class="metric"><span>Real-time x</span><strong id="rtx">0</strong></div>
      </div>
      <div class="views">
        <div class="view">
          <div class="view-head"><div class="view-title">Cortical sheet</div></div>
          <div id="corticalFrame" class="canvas-frame cortical-frame">
            <canvas id="cortical"></canvas>
            <span id="hemiLeft" class="hemi-label hemi-left">Cortical sheet</span>
            <span id="hemiRight" class="hemi-label hemi-right">Right hemisphere</span>
            <span class="axis-label axis-top-left">0&deg;</span>
            <span class="axis-label axis-bottom-left">0&deg;</span>
            <span class="axis-label axis-top-right">0&deg;</span>
            <span class="axis-label axis-bottom-right">0&deg;</span>
            <span class="ecc-label cortical-ecc cortical-fovea-left">fovea</span>
            <span class="ecc-label cortical-ecc cortical-periphery-left">periphery</span>
            <span class="ecc-label cortical-ecc cortical-fovea-right">fovea</span>
            <span class="ecc-label cortical-ecc cortical-periphery-right">periphery</span>
          </div>
        </div>
        <div class="view">
          <div class="view-head"><div class="view-title">Visual field</div></div>
          <div class="canvas-frame retinal-frame">
            <canvas id="retinal"></canvas>
            <span class="axis-label retinal-angle-90">90&deg;</span>
            <span class="axis-label retinal-angle-0">0&deg;</span>
            <span class="axis-label retinal-angle-180">180&deg;</span>
            <span class="axis-label retinal-angle-270">270&deg;</span>
          </div>
        </div>
      </div>
      <div class="legend-wrap">
        <span class="legend-label">Activity</span>
        <span id="legendLow" class="legend-low">low</span>
        <div id="legend" class="legend"></div>
        <span id="legendHigh" class="legend-high">high</span>
      </div>
      <div class="frame-card">
        <div class="frame-toolbar">
          <div class="view-title">Frames</div>
          <select id="frameSelect">
            <option value="stimulus">Stimulus</option>
            <option value="kernel">Kernel</option>
            <option value="field">Neural field</option>
            <option value="phase">Phase plane</option>
          </select>
        </div>
        <div id="stimulusPanel" class="frame-panel" data-frame="stimulus">
          <div class="view-head"><div class="view-title">Stimulus</div><div class="view-note" id="stimulusInfo">moving 0.5 s window</div></div>
          <canvas id="stimulusGraph" class="stimulus-canvas"></canvas>
        </div>
        <div id="kernelPanel" class="frame-panel hidden-control" data-frame="kernel">
          <div class="view-head"><div class="view-title">Kernel window</div><div class="view-note" id="kernelInfo">-</div></div>
          <canvas id="kernelGraph" class="kernel-canvas"></canvas>
          <div class="tiny-note">Cutoff is measured in sigma units: r<sub>e</sub> = ceil(cutoff x &sigma;<sub>e</sub>) and r<sub>i</sub> = ceil(cutoff x &sigma;<sub>i</sub>). The applet uses the separable product of the 1D kernels in x and y.</div>
        </div>
        <div id="fieldPanel" class="frame-panel hidden-control" data-frame="field">
          <div class="view-head"><div class="view-title">Neural field</div><div class="view-note" id="fieldInfo">node lattice and retinal projection</div></div>
          <canvas id="fieldGraph" class="field-canvas"></canvas>
        </div>
        <div id="phasePanel" class="frame-panel hidden-control" data-frame="phase">
          <div class="view-head"><div class="view-title">Phase plane</div><div class="view-note" id="phaseInfo">E/I firing-rate state cloud</div></div>
          <div class="phase-options">
            <label><input id="phaseIncludeAverage" type="checkbox" checked> Include average</label>
          </div>
          <canvas id="phaseGraph" class="phase-canvas"></canvas>
        </div>
      </div>
      <div id="status" class="status">Streaming starts automatically. Use Pause or Reset while tuning parameters.</div>
    </section>
  </main>

  <script>
    const colorMaps = {
      plasma: [[13, 8, 135], [84, 3, 160], [139, 10, 165], [185, 50, 137], [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]],
      viridis: [[68, 1, 84], [70, 50, 126], [54, 92, 141], [39, 127, 142], [31, 161, 136], [74, 193, 109], [160, 218, 57], [253, 231, 37]],
      magma: [[0, 0, 4], [32, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [252, 137, 97], [254, 194, 135], [252, 253, 191]],
      inferno: [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]],
      cividis: [[0, 34, 78], [31, 59, 110], [61, 82, 128], [91, 105, 135], [121, 128, 137], [153, 153, 134], [188, 180, 120], [225, 210, 92], [255, 233, 69]],
      turbo: [[48, 18, 59], [50, 101, 192], [34, 170, 224], [52, 221, 164], [172, 244, 68], [251, 221, 59], [252, 132, 34], [180, 34, 15], [122, 4, 3]],
      nipy_spectral: [[0, 0, 0], [102, 0, 153], [0, 0, 205], [0, 148, 255], [0, 180, 0], [255, 238, 0], [255, 128, 0], [210, 0, 0], [255, 255, 255]],
      gray: [[0, 0, 0], [36, 36, 36], [73, 73, 73], [109, 109, 109], [146, 146, 146], [182, 182, 182], [219, 219, 219], [255, 255, 255]]
    };

    const els = {
      n: document.getElementById("n"),
      fps: document.getElementById("fps"),
      backend: document.getElementById("backend"),
      conv: document.getElementById("conv"),
      boundary: document.getElementById("boundary"),
      boundaryControl: document.getElementById("boundaryControl"),
      boundaryX: document.getElementById("boundaryX"),
      boundaryXControl: document.getElementById("boundaryXControl"),
      boundaryY: document.getElementById("boundaryY"),
      boundaryYControl: document.getElementById("boundaryYControl"),
      partialReflectControl: document.getElementById("partialReflectControl"),
      partialReflectStrength: document.getElementById("partialReflectStrength"),
      coupling: document.getElementById("coupling"),
      speed: document.getElementById("speed"),
      maxSpeed: document.getElementById("maxSpeed"),
      maxSpeedControl: document.getElementById("maxSpeedControl"),
      kernelCutoff: document.getElementById("kernelCutoff"),
      amp: document.getElementById("amp"),
      period: document.getElementById("period"),
      duty: document.getElementById("duty"),
      couplingStrength: document.getElementById("couplingStrength"),
      overlapRows: document.getElementById("overlapRows"),
      colorMap: document.getElementById("colorMap"),
      activityScale: document.getElementById("activityScale"),
      se: document.getElementById("se"),
      si: document.getElementById("si"),
      dt: document.getElementById("dt"),
      fieldGeometry: document.getElementById("fieldGeometry"),
      fieldDensity: document.getElementById("fieldDensity"),
      fieldDensityControl: document.getElementById("fieldDensityControl"),
      nControl: document.getElementById("nControl"),
      fastN: document.getElementById("fastN"),
      fastNControl: document.getElementById("fastNControl"),
      seed: document.getElementById("seed"),
      pausePlay: document.getElementById("pausePlay"),
      reset: document.getElementById("reset"),
      printParams: document.getElementById("printParams"),
      paramOutput: document.getElementById("paramOutput"),
      status: document.getElementById("status"),
      simTime: document.getElementById("simTime"),
      streamFps: document.getElementById("streamFps"),
      msStep: document.getElementById("msStep"),
      rtx: document.getElementById("rtx"),
      stimulusGraph: document.getElementById("stimulusGraph"),
      stimulusInfo: document.getElementById("stimulusInfo"),
      corticalFrame: document.getElementById("corticalFrame"),
      hemiLeft: document.getElementById("hemiLeft"),
      hemiRight: document.getElementById("hemiRight"),
      cortical: document.getElementById("cortical"),
      retinal: document.getElementById("retinal"),
      kernelGraph: document.getElementById("kernelGraph"),
      kernelInfo: document.getElementById("kernelInfo"),
      frameSelect: document.getElementById("frameSelect"),
      framePanels: Array.from(document.querySelectorAll(".frame-panel")),
      fieldGraph: document.getElementById("fieldGraph"),
      fieldInfo: document.getElementById("fieldInfo"),
      phaseGraph: document.getElementById("phaseGraph"),
      phaseInfo: document.getElementById("phaseInfo"),
      phaseIncludeAverage: document.getElementById("phaseIncludeAverage"),
      legend: document.getElementById("legend"),
      legendLow: document.getElementById("legendLow"),
      legendHigh: document.getElementById("legendHigh")
    };

    const presets = {
      default: {
        label: "Default",
        values: {
          boundaryX: "periodic", boundaryY: "periodic", boundary: "edge",
          coupling: "off", kernelCutoff: 3, dt: 0.2,
          couplingStrength: 0.02, overlapRows: 6
        }
      },
      p1: {
        label: "1. Zig-zag square grid",
        values: {
          fieldGeometry: "square", n: 64, amp: 0.2, period: 55, duty: 50,
          se: 2, si: 5, boundaryX: "periodic", boundaryY: "periodic",
          coupling: "off", kernelCutoff: 3, dt: 0.2,
          couplingStrength: 0.02, overlapRows: 6
        }
      },
      p2: {
        label: "2. Square grid",
        values: {
          fieldGeometry: "square", n: 81, amp: 0.7, period: 120, duty: 50,
          se: 2, si: 5, boundaryX: "periodic", boundaryY: "periodic",
          coupling: "off", kernelCutoff: 3, dt: 0.2,
          couplingStrength: 0.02, overlapRows: 6
        }
      },
      p3: {
        label: "3. Lines and dots",
        values: {
          fieldGeometry: "square", n: 81, amp: 0.5, period: 125, duty: 50,
          se: 2.5, si: 6.875, boundaryX: "periodic", boundaryY: "periodic",
          coupling: "off", kernelCutoff: 3, dt: 0.2,
          couplingStrength: 0.02, overlapRows: 6
        }
      },
      p4: {
        label: "4. Hex grid rings",
        values: {
          fieldGeometry: "square", n: 81, amp: 0.5, period: 115, duty: 50,
          se: 2.5, si: 6.875, boundaryX: "periodic", boundaryY: "periodic",
          coupling: "off", kernelCutoff: 3, dt: 0.2,
          couplingStrength: 0.02, overlapRows: 6
        }
      }
    };

    let socket = null;
    let paused = false;
    let resetting = false;
    let streamToken = 0;
    let lastFrameAt = performance.now();
    let lastRateFrameAt = null;
    let lastRateSimTime = null;
    let lastDisplayFrame = null;
    let visualizationUpdateTimer = null;
    let resetFallbackTimer = null;
    let streamStimulus = {
      A: Number(els.amp.value) || 0,
      period: Number(els.period.value) || 1,
      duty: Number(els.duty.value) || 50
    };
    const meanFieldParams = {
      Aee: 10.0,
      Aei: 12.0,
      Aie: 8.5,
      Aii: 3.0,
      He: 2.0,
      Hi: 3.5,
      Ge: 1.0,
      Gi: 0.0
    };

    function activeColorStops() {
      return colorMaps[els.colorMap.value] || colorMaps.plasma;
    }

    function colorStopString(stops = activeColorStops()) {
      return stops.map((rgb, idx) => `rgb(${rgb.join(",")}) ${(idx / (stops.length - 1)) * 100}%`).join(", ");
    }

    function palette(v) {
      const stops = activeColorStops();
      const t = Math.max(0, Math.min(1, v / 255));
      const scaled = t * (stops.length - 1);
      const idx = Math.min(Math.floor(scaled), stops.length - 2);
      const f = scaled - idx;
      const a = stops[idx], b = stops[idx + 1];
      return [
        Math.round((1 - f) * a[0] + f * b[0]),
        Math.round((1 - f) * a[1] + f * b[1]),
        Math.round((1 - f) * a[2] + f * b[2])
      ];
    }

    function updateLegend() {
      els.legend.style.background = `linear-gradient(90deg, ${colorStopString()})`;
    }

    function formatSimTime(ms) {
      if (!Number.isFinite(ms)) return "0 ms";
      if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
      return `${ms.toFixed(1)} ms`;
    }

    function currentSpeedValue() {
      return els.maxSpeed.checked ? 0 : Math.max(0.1, Number(els.speed.value) || 1);
    }

    function syncSpeedControls() {
      els.speed.disabled = els.maxSpeed.checked;
    }

    function formatSpeed(value) {
      return value === 0 ? "max" : `${Number(value).toFixed(1).replace(/\.0$/, "")}x`;
    }

    function boundaryHasReflection() {
      if (els.fieldGeometry.value === "double_sech") {
        return els.boundary.value === "partial_reflect";
      }
      return els.boundaryX.value === "partial_reflect" || els.boundaryY.value === "partial_reflect";
    }

    function syncReflectControl() {
      const showReflect = boundaryHasReflection();
      els.partialReflectControl.classList.toggle("hidden-control", !showReflect);
      els.partialReflectStrength.disabled = !showReflect;
    }

    function setOptionAvailable(select, value, available) {
      const option = Array.from(select.options).find((item) => item.value === value);
      if (!option) return;
      option.disabled = !available;
      option.hidden = !available;
    }

    function setCanvasSize(canvas, rows, cols) {
      if (canvas.width !== cols || canvas.height !== rows) {
        canvas.width = cols;
        canvas.height = rows;
      }
      canvas.style.aspectRatio = `${cols} / ${rows}`;
    }

    function drawValues(canvas, values, rows, cols) {
      setCanvasSize(canvas, rows, cols);
      const ctx = canvas.getContext("2d");
      const image = ctx.createImageData(cols, rows);
      for (let i = 0; i < values.length; i++) {
        const [r, g, b] = palette(values[i]);
        const j = i * 4;
        image.data[j] = r;
        image.data[j + 1] = g;
        image.data[j + 2] = b;
        image.data[j + 3] = 255;
      }
      ctx.putImageData(image, 0, 0);
    }

    function setPixel(image, pixelIndex, value) {
      const [r, g, b] = palette(value);
      const j = pixelIndex * 4;
      image.data[j] = r;
      image.data[j + 1] = g;
      image.data[j + 2] = b;
      image.data[j + 3] = 255;
    }

    function updateCorticalLabels(isCoupled, leftCenter = 50, rightCenter = 50, positions = null) {
      const isDoubleSech = els.fieldGeometry.value === "double_sech";
      els.corticalFrame.classList.toggle("coupled", isCoupled);
      els.corticalFrame.classList.toggle("stacked", isCoupled);
      els.corticalFrame.classList.toggle("double-sech", isDoubleSech);
      els.corticalFrame.style.setProperty("--left-center", `${leftCenter}%`);
      els.corticalFrame.style.setProperty("--right-center", `${rightCenter}%`);
      if (positions) {
        Object.entries(positions).forEach(([key, value]) => {
          els.corticalFrame.style.setProperty(key, value);
        });
      } else {
        els.corticalFrame.style.setProperty("--left-label-y", "10px");
        els.corticalFrame.style.setProperty("--right-label-y", "10px");
        els.corticalFrame.style.setProperty("--left-top-y", "30px");
        els.corticalFrame.style.setProperty("--right-top-y", "30px");
        els.corticalFrame.style.setProperty("--left-bottom-y", "calc(100% - 30px)");
        els.corticalFrame.style.setProperty("--right-bottom-y", "calc(100% - 30px)");
        els.corticalFrame.style.setProperty("--left-ecc-y", "50%");
        els.corticalFrame.style.setProperty("--right-ecc-y", "50%");
      }
      els.hemiLeft.textContent = isCoupled ? "Left hemisphere" : "Cortical sheet";
      els.hemiRight.textContent = "Right hemisphere";
      document.querySelectorAll(".axis-top-left, .axis-top-right").forEach((label) => {
        label.textContent = isCoupled ? "90\u00b0" : "0\u00b0";
      });
      document.querySelectorAll(".axis-bottom-left, .axis-bottom-right").forEach((label) => {
        label.textContent = isCoupled ? "270\u00b0" : "0\u00b0";
      });
    }

    function drawCortical(canvas, values, rows, cols) {
      const isCoupled = cols >= rows * 1.5;
      if (!isCoupled) {
        updateCorticalLabels(false);
        drawValues(canvas, values, rows, cols);
        return;
      }

      const hemiCols = Math.floor(cols / 2);
      const gapRows = Math.max(24, Math.round(rows * 0.32));
      const drawRows = 2 * rows + gapRows;
      const drawCols = hemiCols;
      setCanvasSize(canvas, drawRows, drawCols);
      updateCorticalLabels(true, 50, 50, {
        "--left-label-y": "12px",
        "--left-top-y": "40px",
        "--left-bottom-y": "calc(50% - 20px)",
        "--left-ecc-y": "calc(25% + 11px)",
        "--right-label-y": "calc(100% - 13px)",
        "--right-top-y": "calc(50% + 20px)",
        "--right-bottom-y": "calc(100% - 30px)",
        "--right-ecc-y": "calc(75% + 8px)"
      });

      const ctx = canvas.getContext("2d");
      const image = ctx.createImageData(drawCols, drawRows);
      for (let i = 0; i < image.data.length; i += 4) {
        image.data[i] = 248;
        image.data[i + 1] = 251;
        image.data[i + 2] = 253;
        image.data[i + 3] = 255;
      }

      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < hemiCols; col++) {
          const rightRow = els.fieldGeometry.value === "double_sech" ? rows - 1 - row : row;
          setPixel(image, row * drawCols + col, values[row * cols + col]);
          setPixel(
            image,
            (row + rows + gapRows) * drawCols + col,
            values[rightRow * cols + hemiCols + col]
          );
        }
      }
      ctx.putImageData(image, 0, 0);
    }

    function setPauseUi(isPaused) {
      paused = isPaused;
      els.pausePlay.textContent = isPaused ? "Play" : "Pause";
      els.pausePlay.classList.toggle("paused", isPaused);
    }

    function resetMetrics() {
      els.simTime.textContent = "0 ms";
      els.streamFps.textContent = "0";
      els.msStep.textContent = "0";
      els.rtx.textContent = "0";
      els.legendLow.textContent = "low";
      els.legendHigh.textContent = "high";
      lastRateFrameAt = null;
      lastRateSimTime = null;
      drawStimulusGraph(0);
    }

    function gaussian1dValue(x, sigma) {
      return Math.exp(-(x * x) / (sigma * sigma)) / (Math.sqrt(Math.PI) * sigma);
    }

    function kernelMass1d(sigma, radius) {
      let sum = 0;
      for (let x = -radius; x <= radius; x++) sum += gaussian1dValue(x, sigma);
      return sum;
    }

    function drawKernelGraph() {
      const canvas = els.kernelGraph;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(360, Math.round(rect.width * dpr));
      const height = Math.max(160, Math.round(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      let n = Math.max(5, Number(els.n.value) || 101);
      if (els.fieldGeometry.value === "double_sech") {
        n = Math.max(5, Math.round(81 * (Number(els.fieldDensity.value) || 1)));
        if (n % 2 === 0) n += 1;
      }
      const se = Math.max(0.1, Number(els.se.value) || 2);
      const si = Math.max(0.1, Number(els.si.value) || 5);
      const cutoff = Math.max(0.5, Number(els.kernelCutoff.value) || 3);
      const radiusE = Math.max(1, Math.ceil(cutoff * se));
      const radiusI = Math.max(1, Math.ceil(cutoff * si));
      const fullRadius = Math.max(radiusI, Math.floor(n / 2));
      const retainedE = Math.pow(kernelMass1d(se, radiusE) / kernelMass1d(se, fullRadius), 2) * 100;
      const retainedI = Math.pow(kernelMass1d(si, radiusI) / kernelMass1d(si, fullRadius), 2) * 100;
      const maxRadius = Math.max(radiusE, radiusI, 4);
      const samples = [];
      let maxValue = 0;
      for (let x = -maxRadius; x <= maxRadius; x++) {
        const e = gaussian1dValue(x, se);
        const i = gaussian1dValue(x, si);
        samples.push({ x, e, i });
        maxValue = Math.max(maxValue, e, i);
      }

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createLinearGradient(0, 0, width, height);
      bg.addColorStop(0, "#ffffff");
      bg.addColorStop(1, "#eef7f8");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);
      const padL = 42 * dpr, padR = 20 * dpr, padT = 22 * dpr, padB = 34 * dpr;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      const xToPx = (x) => padL + ((x + maxRadius) / (2 * maxRadius)) * plotW;
      const yToPx = (v) => padT + (1 - v / maxValue) * plotH;

      ctx.lineWidth = 1 * dpr;
      ctx.strokeStyle = "#e1edf2";
      ctx.beginPath();
      for (let i = 0; i <= 4; i++) {
        const y = padT + (i / 4) * plotH;
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
      }
      ctx.stroke();

      function drawRadius(radius, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2 * dpr;
        ctx.setLineDash([4 * dpr, 4 * dpr]);
        [-radius, radius].forEach((xValue) => {
          const x = xToPx(xValue);
          ctx.beginPath();
          ctx.moveTo(x, padT);
          ctx.lineTo(x, padT + plotH);
          ctx.stroke();
        });
        ctx.setLineDash([]);
      }

      drawRadius(radiusI, "rgba(243, 179, 61, 0.58)");
      drawRadius(radiusE, "rgba(0, 158, 170, 0.58)");

      ctx.strokeStyle = "#b9ccd7";
      ctx.beginPath();
      ctx.moveTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.moveTo(xToPx(0), padT);
      ctx.lineTo(xToPx(0), padT + plotH);
      ctx.stroke();

      function drawLine(key, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5 * dpr;
        ctx.beginPath();
        samples.forEach((sample, idx) => {
          const x = xToPx(sample.x);
          const y = yToPx(sample[key]);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      drawLine("i", "#f3b33d");
      drawLine("e", "#009eaa");
      ctx.fillStyle = "#607284";
      ctx.font = `${11 * dpr}px IBM Plex Sans, sans-serif`;
      ctx.textAlign = "left";
      ctx.fillText(`-${maxRadius}`, padL, height - 10 * dpr);
      ctx.textAlign = "center";
      ctx.fillText(`0`, xToPx(0) - 3 * dpr, height - 10 * dpr);
      ctx.textAlign = "right";
      ctx.fillText(`+${maxRadius}`, padL + plotW, height - 10 * dpr);
      ctx.textAlign = "left";
      ctx.fillStyle = "#009eaa";
      ctx.fillText(`\u03c3\u2091, r=${radiusE}`, padL + 8 * dpr, padT + 14 * dpr);
      ctx.fillStyle = "#f3b33d";
      ctx.fillText(`\u03c3\u1d62, r=${radiusI}`, padL + 110 * dpr, padT + 14 * dpr);
      els.kernelInfo.textContent =
        `r_e=ceil(${cutoff} x ${se})=${radiusE}; r_i=ceil(${cutoff} x ${si})=${radiusI}; mass ${retainedE.toFixed(3)}% / ${retainedI.toFixed(3)}%`;
    }

    function drawRetinal(canvas, values, rows, cols) {
      drawValues(canvas, values, rows, cols);
    }

    function stimulusThreshold(duty) {
      return Math.sin(Math.PI * (0.5 - Math.max(0, Math.min(100, duty)) / 100));
    }

    function strobeValue(t, stimulus = streamStimulus) {
      const period = Math.max(1e-6, stimulus.period);
      const threshold = stimulusThreshold(stimulus.duty);
      return Math.sin((2 * Math.PI * t) / period) - threshold > 0 ? stimulus.A : 0;
    }

    function controlStimulus() {
      return {
        A: Number(els.amp.value) || 0,
        period: Math.max(1e-6, Number(els.period.value) || 1),
        duty: Number(els.duty.value) || 50
      };
    }

    function drawStimulusGraph(t = 0) {
      const canvas = els.stimulusGraph;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(320, Math.round(rect.width * dpr));
      const height = Math.max(72, Math.round(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#f8fbfd";
      ctx.fillRect(0, 0, width, height);

      const padL = 34 * dpr, padR = 12 * dpr, padT = 10 * dpr, padB = 18 * dpr;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      const windowMs = 500;
      const halfWindowMs = windowMs / 2;
      const start = t - halfWindowMs;
      const samples = Math.max(80, Math.round(plotW));
      const maxA = Math.max(0.001, streamStimulus.A);
      const xFor = (i) => padL + (i / (samples - 1)) * plotW;
      const yFor = (value) => padT + (1 - value / maxA) * plotH;

      ctx.strokeStyle = "#dbe7ef";
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      for (let i = 0; i <= 4; i++) {
        const x = padL + (i / 4) * plotW;
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
      }
      ctx.moveTo(padL, padT + plotH);
      ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      ctx.strokeStyle = "#009eaa";
      ctx.lineWidth = 2.2 * dpr;
      ctx.beginPath();
      for (let i = 0; i < samples; i++) {
        const sampleT = start + (i / (samples - 1)) * windowMs;
        const x = xFor(i);
        const y = yFor(strobeValue(sampleT));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.strokeStyle = "#f3b33d";
      ctx.lineWidth = 1.4 * dpr;
      const nowX = padL + plotW / 2;
      ctx.beginPath();
      ctx.moveTo(nowX, padT - 2 * dpr);
      ctx.lineTo(nowX, padT + plotH + 2 * dpr);
      ctx.stroke();

      ctx.fillStyle = "#607284";
      ctx.font = `${10 * dpr}px IBM Plex Sans, sans-serif`;
      ctx.textAlign = "left";
      ctx.fillText("-0.25 s", padL, height - 5 * dpr);
      ctx.textAlign = "center";
      ctx.fillText("now", nowX, height - 5 * dpr);
      ctx.textAlign = "right";
      ctx.fillText("+0.25 s", padL + plotW, height - 5 * dpr);
      ctx.textAlign = "left";
      els.stimulusInfo.textContent = "moving 0.5 s window";
    }

    function fieldDimensions() {
      if (lastDisplayFrame) {
        const coupledFrame = lastDisplayFrame.cols >= lastDisplayFrame.rows * 1.5;
        return {
          rows: lastDisplayFrame.rows,
          cols: coupledFrame ? Math.floor(lastDisplayFrame.cols / 2) : lastDisplayFrame.cols,
          coupled: coupledFrame
        };
      }
      let n = Math.max(5, Math.round(Number(els.n.value) || 81));
      const coupled = els.fieldGeometry.value === "double_sech" || els.coupling.value !== "off";
      if (els.fieldGeometry.value === "double_sech") {
        n = Math.max(5, Math.round(81 * (Number(els.fieldDensity.value) || 1)));
        if (n % 2 === 0) n += 1;
        const doubleSechCols = Math.max(7, Math.round(n * 1.57));
        return { rows: n, cols: doubleSechCols % 2 === 0 ? doubleSechCols + 1 : doubleSechCols, coupled };
      }
      return { rows: n, cols: n, coupled };
    }

    function drawFieldGraph() {
      const canvas = els.fieldGraph;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(560, Math.round(rect.width * dpr));
      const height = Math.max(250, Math.round(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createLinearGradient(0, 0, width, height);
      bg.addColorStop(0, "#ffffff");
      bg.addColorStop(1, "#eef7f8");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      const { rows, cols, coupled } = fieldDimensions();
      const overlap = Math.max(0, Math.round(Number(els.overlapRows.value) || 0));
      const hasOverlap = els.coupling.value === "overlap" && overlap > 0;
      const geometry = els.fieldGeometry.value;
      const leftW = width * 0.54;
      const rightW = width - leftW;
      const pad = 22 * dpr;
      const sheetW = leftW - 2 * pad;
      const sheetGap = coupled ? 18 * dpr : 0;
      const sheetH = coupled ? (height - 2 * pad - sheetGap) / 2 : height - 2 * pad;
      const pointStep = Math.max(1, Math.ceil(Math.max(rows, cols) / 96));
      const pointRadius = Math.max(0.55 * dpr, Math.min(1.6 * dpr, 58 / Math.max(rows, cols) * dpr));

      function sheetX(col, x, w) {
        return x + (cols <= 1 ? 0.5 : col / (cols - 1)) * w;
      }

      function sheetY(row, y, h) {
        return y + (rows <= 1 ? 0.5 : row / (rows - 1)) * h;
      }

      function inDoubleSechMask(row, col) {
        if (geometry !== "double_sech") return true;
        const yn = rows <= 1 ? 0 : -1 + 2 * row / (rows - 1);
        const xn = cols <= 1 ? 0 : -1 + 2 * col / (cols - 1);
        const halfWidth = 0.35 + 0.52 * Math.sqrt(Math.max(0, 1 - 0.42 * yn * yn));
        return Math.abs(xn) <= halfWidth;
      }

      function isDoubleSechBorder(row, col) {
        if (!inDoubleSechMask(row, col)) return false;
        const radius = Math.max(1, Math.floor(overlap / 2));
        for (let dr = -radius; dr <= radius; dr++) {
          for (let dc = -radius; dc <= radius; dc++) {
            const rr = row + dr;
            const cc = col + dc;
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols || !inDoubleSechMask(rr, cc)) {
              return true;
            }
          }
        }
        return false;
      }

      function nodeIsOverlap(row, col) {
        if (!hasOverlap) return false;
        if (geometry === "double_sech") return isDoubleSechBorder(row, col);
        return row < overlap || row >= rows - overlap;
      }

      function drawSheet(x, y, w, h, label) {
        ctx.strokeStyle = "rgba(13, 38, 56, 0.18)";
        ctx.lineWidth = 1.1 * dpr;
        ctx.strokeRect(x, y, w, h);
        if (hasOverlap && geometry !== "double_sech") {
          const band = Math.max(1, overlap / Math.max(rows, 1)) * h;
          ctx.fillStyle = "rgba(243, 179, 61, 0.14)";
          ctx.fillRect(x, y, w, band);
          ctx.fillRect(x, y + h - band, w, band);
        }
        ctx.fillStyle = "#0b3146";
        ctx.font = `${11 * dpr}px IBM Plex Sans, sans-serif`;
        ctx.textAlign = "left";
        ctx.fillText(label, x, y - 7 * dpr);

        for (let row = 0; row < rows; row += pointStep) {
          for (let col = 0; col < cols; col += pointStep) {
            if (!inDoubleSechMask(row, col)) continue;
            ctx.fillStyle = nodeIsOverlap(row, col) ? "rgba(243, 179, 61, 0.88)" : "rgba(0, 112, 124, 0.62)";
            ctx.beginPath();
            ctx.arc(sheetX(col, x, w), sheetY(row, y, h), pointRadius, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        if (geometry === "double_sech") {
          ctx.fillStyle = "#607284";
          ctx.font = `${9 * dpr}px IBM Plex Sans, sans-serif`;
          const eccY = y + h + 13 * dpr < height - 4 * dpr ? y + h + 13 * dpr : y - 8 * dpr;
          ctx.textAlign = "left";
          ctx.fillText("fovea", x, eccY);
          ctx.textAlign = "right";
          ctx.fillText("periphery", x + w, eccY);
        }
      }

      if (coupled) {
        drawSheet(pad, pad + 16 * dpr, sheetW, sheetH, "Left hemisphere");
        drawSheet(pad, pad + 16 * dpr + sheetH + sheetGap, sheetW, sheetH, "Right hemisphere");
      } else {
        drawSheet(pad, pad + 16 * dpr, sheetW, sheetH, "Cortical sheet");
      }

      const cx = leftW + rightW / 2;
      const cy = height / 2 + 8 * dpr;
      const rMax = Math.min(rightW, height) * 0.34;
      ctx.strokeStyle = "rgba(13, 38, 56, 0.22)";
      ctx.lineWidth = 1.2 * dpr;
      ctx.beginPath();
      ctx.arc(cx, cy, rMax, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - rMax, cy);
      ctx.lineTo(cx + rMax, cy);
      ctx.moveTo(cx, cy - rMax);
      ctx.lineTo(cx, cy + rMax);
      ctx.stroke();
      ctx.fillStyle = "#0b3146";
      ctx.font = `${11 * dpr}px IBM Plex Sans, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText("Retinal projection", cx, pad + 9 * dpr);

      for (let row = 0; row < rows; row += pointStep) {
        for (let col = 0; col < cols; col += pointStep) {
          if (!inDoubleSechMask(row, col)) continue;
          const theta = Math.PI / 2 - (rows <= 1 ? 0 : row / (rows - 1)) * Math.PI * 2;
          const radius = rMax * (0.08 + 0.9 * (cols <= 1 ? 0 : col / (cols - 1)));
          ctx.fillStyle = nodeIsOverlap(row, col) ? "rgba(243, 179, 61, 0.88)" : "rgba(0, 112, 124, 0.46)";
          ctx.beginPath();
          ctx.arc(cx + Math.cos(theta) * radius, cy - Math.sin(theta) * radius, pointRadius, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.fillStyle = "#607284";
      ctx.font = `${9 * dpr}px IBM Plex Sans, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText("fovea", cx, cy + 3 * dpr);
      ctx.fillText("periphery", cx + rMax * 0.72, cy + rMax * 0.72);

      els.fieldInfo.textContent =
        `${rows} x ${cols} nodes${pointStep > 1 ? `, showing every ${pointStep}th node` : ""}${hasOverlap ? `, overlap ${geometry === "double_sech" ? "border ring" : "rows"} highlighted: ${overlap}` : ""}`;
    }

    function drawPhasePlane() {
      const canvas = els.phaseGraph;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const size = Math.max(420, Math.round(Math.max(rect.width, 320) * dpr));
      if (canvas.width !== size || canvas.height !== size) {
        canvas.width = size;
        canvas.height = size;
      }
      const width = size;
      const height = size;

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createLinearGradient(0, 0, width, height);
      bg.addColorStop(0, "#ffffff");
      bg.addColorStop(1, "#eef7f8");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      const padL = 58 * dpr, padR = 20 * dpr, padT = 22 * dpr, padB = 54 * dpr;
      const availableW = width - padL - padR;
      const availableH = height - padT - padB;
      const plotSize = Math.max(120 * dpr, Math.min(availableW, availableH));
      const plotL = padL + (availableW - plotSize) / 2;
      const plotT = padT + (availableH - plotSize) / 2;
      const plotW = plotSize;
      const plotH = plotSize;
      const clampRate = (value) => Math.max(0, Math.min(1, value));
      const xRate = (value) => plotL + clampRate(value) * plotW;
      const yRate = (value) => plotT + (1 - clampRate(value)) * plotH;
      const xFor = (value) => xRate(value / 255);
      const yFor = (value) => yRate(value / 255);
      const dotPath = (x, y, radius) => {
        ctx.moveTo(x + radius, y);
        ctx.arc(x, y, radius, 0, Math.PI * 2);
      };
      const logit = (value) => {
        const safe = Math.max(1e-5, Math.min(1 - 1e-5, value));
        return Math.log(safe / (1 - safe));
      };
      const drawCurve = (sampler, color) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.2 * dpr;
        ctx.beginPath();
        let drawing = false;
        for (let idx = 0; idx <= 256; idx++) {
          const input = idx / 256;
          const point = sampler(input);
          const valid = Number.isFinite(point.x) && Number.isFinite(point.y) &&
            point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1;
          if (!valid) {
            drawing = false;
            continue;
          }
          const x = xRate(point.x);
          const y = yRate(point.y);
          if (!drawing) {
            ctx.moveTo(x, y);
            drawing = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      };
      const drawMeanFieldNullclines = (stim) => {
        const p = meanFieldParams;
        drawCurve((ue) => ({
          x: ue,
          y: (p.Aee * ue - p.He + p.Ge * stim - logit(ue)) / p.Aie
        }), "#009eaa");
        drawCurve((ui) => ({
          x: (logit(ui) + p.Aii * ui + p.Hi - p.Gi * stim) / p.Aei,
          y: ui
        }), "#f3b33d");

        ctx.font = `${10 * dpr}px IBM Plex Sans, sans-serif`;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#009eaa";
        ctx.fillText("dUe/dt=0", plotL + plotW - 4 * dpr, plotT + 14 * dpr);
        ctx.fillStyle = "#b37500";
        ctx.fillText("dUi/dt=0", plotL + plotW - 4 * dpr, plotT + 30 * dpr);
      };

      ctx.strokeStyle = "#dbe7ef";
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      for (let i = 0; i <= 4; i++) {
        const x = plotL + (i / 4) * plotW;
        const y = plotT + (i / 4) * plotH;
        ctx.moveTo(x, plotT);
        ctx.lineTo(x, plotT + plotH);
        ctx.moveTo(plotL, y);
        ctx.lineTo(plotL + plotW, y);
      }
      ctx.stroke();

      ctx.strokeStyle = "#8aa2af";
      ctx.lineWidth = 1.2 * dpr;
      ctx.beginPath();
      ctx.moveTo(plotL, plotT + plotH);
      ctx.lineTo(plotL + plotW, plotT + plotH);
      ctx.moveTo(plotL, plotT);
      ctx.lineTo(plotL, plotT + plotH);
      ctx.stroke();

      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.strokeStyle = "rgba(13, 38, 56, 0.26)";
      ctx.beginPath();
      ctx.moveTo(plotL, plotT + plotH);
      ctx.lineTo(plotL + plotW, plotT);
      ctx.stroke();
      ctx.setLineDash([]);

      const currentT = lastDisplayFrame?.t || 0;
      const currentStim = strobeValue(currentT, controlStimulus());
      drawMeanFieldNullclines(currentStim);

      const phaseE = lastDisplayFrame?.phaseEValues || new Uint8Array();
      const phaseI = lastDisplayFrame?.phaseIValues || new Uint8Array();
      const n = Math.min(lastDisplayFrame?.phaseCount || phaseE.length, phaseE.length, phaseI.length);
      const includeAverage = els.phaseIncludeAverage?.checked ?? true;
      if (n === 0) {
        ctx.fillStyle = "#607284";
        ctx.font = `${12 * dpr}px IBM Plex Sans, sans-serif`;
        ctx.textAlign = "center";
        ctx.fillText("Waiting for a live frame...", plotL + plotW / 2, plotT + plotH / 2);
        els.phaseInfo.textContent = "E/I firing-rate state cloud";
      } else {
        let meanE = 0;
        let meanI = 0;
        const pointRadius = Math.max(0.85, Math.min(2.1, 32 / Math.sqrt(n)) * dpr);
        ctx.fillStyle = "rgba(0, 87, 99, 0.34)";
        ctx.beginPath();
        for (let idx = 0; idx < n; idx++) {
          const e = phaseE[idx];
          const i = phaseI[idx];
          meanE += e;
          meanI += i;
          dotPath(xFor(e), yFor(i), pointRadius);
        }
        ctx.fill();
        meanE /= n;
        meanI /= n;

        if (includeAverage) {
          ctx.fillStyle = "#071018";
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.4 * dpr;
          ctx.beginPath();
          ctx.arc(xFor(meanE), yFor(meanI), 4.8 * dpr, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
        els.phaseInfo.textContent =
          `${n.toLocaleString()} nodes, mean E=${(meanE / 255).toFixed(3)}, mean I=${(meanI / 255).toFixed(3)}, S=${currentStim.toFixed(3)}${includeAverage ? "" : ", average hidden"}`;
      }

      ctx.fillStyle = "#607284";
      ctx.font = `${10 * dpr}px IBM Plex Sans, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i <= 4; i++) {
        const label = (i / 4).toFixed(i === 0 || i === 4 ? 0 : 2);
        ctx.fillText(label, plotL + (i / 4) * plotW, plotT + plotH + 7 * dpr);
      }
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 4; i++) {
        const value = 1 - i / 4;
        const label = value.toFixed(value === 0 || value === 1 ? 0 : 2);
        ctx.fillText(label, plotL - 8 * dpr, plotT + (i / 4) * plotH);
      }

      ctx.fillStyle = "#0b3146";
      ctx.font = `${11 * dpr}px IBM Plex Sans, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.fillText("Excitatory firing rate (Ue)", plotL + plotW / 2, height - 12 * dpr);
      ctx.save();
      ctx.translate(16 * dpr, plotT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Inhibitory firing rate (Ui)", 0, 0);
      ctx.restore();
    }

    function drawCurrentFrame() {
      if (!lastDisplayFrame) return;
      drawCortical(
        els.cortical,
        lastDisplayFrame.values,
        lastDisplayFrame.rows,
        lastDisplayFrame.cols
      );
      drawRetinal(
        els.retinal,
        lastDisplayFrame.retinalValues,
        lastDisplayFrame.retinalRows,
        lastDisplayFrame.retinalCols
      );
      if (els.frameSelect.value === "field") drawFieldGraph();
      if (els.frameSelect.value === "phase") drawPhasePlane();
    }

    function updateFramePanel() {
      const selected = els.frameSelect.value;
      els.framePanels.forEach((panel) => {
        panel.classList.toggle("hidden-control", panel.dataset.frame !== selected);
      });
      if (selected === "stimulus") {
        drawStimulusGraph(lastDisplayFrame?.t || 0);
      } else if (selected === "kernel") {
        drawKernelGraph();
      } else if (selected === "field") {
        drawFieldGraph();
      } else if (selected === "phase") {
        drawPhasePlane();
      }
    }

    function decodeFrame(data) {
      const binary = atob(data);
      const values = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) values[i] = binary.charCodeAt(i);
      return values;
    }

    function streamParams() {
      syncConvolutionControls();
      const params = new URLSearchParams();
      const isDoubleSech = els.fieldGeometry.value === "double_sech";
      params.set("field_geometry", els.fieldGeometry.value);
      if (isDoubleSech) {
        params.set("field_density", els.fieldDensity.value);
      } else {
        params.set("N", els.n.value);
        params.set("fast_n", els.fastN.checked ? "true" : "false");
      }
      params.set("fps", els.fps.value);
      params.set("backend", els.backend.value);
      params.set("conv", els.conv.value);
      params.set("activity_scale", els.activityScale.value);
      if (isDoubleSech) {
        params.set("boundary", els.boundary.value);
      } else {
        params.set("boundary_x", els.boundaryX.value);
        params.set("boundary_y", els.boundaryY.value);
      }
      params.set("partial_reflect_strength", els.partialReflectStrength.value);
      params.set("coupling", els.coupling.value);
      params.set("speed", String(currentSpeedValue()));
      params.set("kernel_cutoff", els.kernelCutoff.value);
      params.set("A", els.amp.value);
      params.set("T", els.period.value);
      params.set("duty_cycle", els.duty.value);
      params.set("coupling_strength", els.couplingStrength.value);
      params.set("overlap_rows", els.overlapRows.value);
      params.set("Se", els.se.value);
      params.set("Si", els.si.value);
      params.set("dt", els.dt.value);
      if (els.seed.value.trim()) params.set("seed", els.seed.value.trim());
      return params;
    }

    function collectParameterSnapshot() {
      return {
        visualization: {
          fps: Number(els.fps.value) || 30,
          speed: els.maxSpeed.checked ? "max" : Number(els.speed.value) || 1,
          colormap: els.colorMap.value,
          activity_scale: els.activityScale.value
        },
        backend: {
          backend: els.backend.value,
          convolution: els.conv.value,
          kernel_cutoff: Number(els.kernelCutoff.value) || 3,
          seed: els.seed.value.trim() || null,
          fast_n: els.fastN.checked
        },
        boundary: {
          boundary: els.fieldGeometry.value === "double_sech" ? els.boundary.value : null,
          boundary_x: els.fieldGeometry.value === "double_sech" ? null : els.boundaryX.value,
          boundary_y: els.fieldGeometry.value === "double_sech" ? null : els.boundaryY.value,
          reflect_gain: boundaryHasReflection() ? Number(els.partialReflectStrength.value) || 0 : null
        },
        coupling: {
          mode: els.coupling.value,
          overlap_rows: Number(els.overlapRows.value) || 0,
          g: Number(els.couplingStrength.value) || 0
        },
        strobe: {
          amplitude: Number(els.amp.value) || 0,
          period_ms: Number(els.period.value) || 0,
          duty_cycle_percent: Number(els.duty.value) || 0
        },
        neural_field: {
          geometry: els.fieldGeometry.value,
          density: els.fieldGeometry.value === "double_sech" ? Number(els.fieldDensity.value) || 1 : null,
          N: els.fieldGeometry.value === "double_sech" ? null : Number(els.n.value) || 0,
          sigma_e: Number(els.se.value) || 0,
          sigma_i: Number(els.si.value) || 0,
          dt_ms: Number(els.dt.value) || 0
        }
      };
    }

    function printParameters(label = "Current parameters") {
      const params = streamParams();
      const query = params.toString();
      const snapshot = collectParameterSnapshot();
      els.paramOutput.textContent =
        `${label}\n${JSON.stringify(snapshot, null, 2)}\n\nstream query:\n${query}\n\nstream path:\n/stream?${query}`;
    }

    function setControlValue(id, value) {
      const el = els[id];
      if (!el || value === undefined) return;
      el.value = String(value);
    }

    function applyPreset(key) {
      const preset = presets[key];
      if (!preset) return;
      const values = preset.values;
      Object.entries(values).forEach(([id, value]) => setControlValue(id, value));
      if (values.fieldGeometry === "square") {
        els.fieldDensity.value = "1";
      }
      applyGeometryDefaults();
      syncReflectControl();
      drawKernelGraph();
      drawStimulusGraph(lastDisplayFrame?.t || 0);
      drawFieldGraph();
      printParameters(`Applied preset: ${preset.label}`);
      resetStream();
    }

    function syncGeometryControls() {
      const isDoubleSech = els.fieldGeometry.value === "double_sech";
      els.nControl.classList.toggle("hidden-control", isDoubleSech);
      els.fastNControl.classList.toggle("hidden-control", isDoubleSech);
      els.fieldDensityControl.classList.toggle("hidden-control", !isDoubleSech);
      els.boundaryControl.classList.toggle("hidden-control", !isDoubleSech);
      els.boundaryXControl.classList.toggle("hidden-control", isDoubleSech);
      els.boundaryYControl.classList.toggle("hidden-control", isDoubleSech);
      els.n.disabled = isDoubleSech;
      els.fastN.disabled = isDoubleSech;
      els.fieldDensity.disabled = !isDoubleSech;
      els.boundary.disabled = !isDoubleSech;
      els.boundaryX.disabled = isDoubleSech;
      els.boundaryY.disabled = isDoubleSech;
      syncReflectControl();
    }

    function syncConvolutionControls() {
      const isDoubleSech = els.fieldGeometry.value === "double_sech";
      if (isDoubleSech) {
        els.backend.value = "metal";
      }
      els.backend.disabled = isDoubleSech;

      const isCpu = els.backend.value === "cpu";
      const squareUsesNonPeriodic =
        !isDoubleSech && (els.boundaryX.value !== "periodic" || els.boundaryY.value !== "periodic");

      if (isCpu) {
        els.boundaryX.value = "periodic";
        els.boundaryY.value = "periodic";
        els.boundaryX.disabled = true;
        els.boundaryY.disabled = true;
      } else if (!isDoubleSech) {
        els.boundaryX.disabled = false;
        els.boundaryY.disabled = false;
      }

      if (isDoubleSech || squareUsesNonPeriodic) {
        els.conv.value = "separable";
        els.conv.disabled = true;
        setOptionAvailable(els.conv, "separable", true);
        setOptionAvailable(els.conv, "fft", false);
      } else if (isCpu) {
        els.conv.value = "fft";
        els.conv.disabled = true;
        setOptionAvailable(els.conv, "separable", false);
        setOptionAvailable(els.conv, "fft", true);
      } else {
        setOptionAvailable(els.conv, "separable", true);
        setOptionAvailable(els.conv, "fft", true);
        els.conv.disabled = false;
      }
      syncReflectControl();
    }

    function applyGeometryDefaults() {
      syncGeometryControls();
      syncConvolutionControls();
      if (els.fieldGeometry.value !== "double_sech") return;
      els.backend.value = "metal";
      els.conv.value = "separable";
    }

    function sendVisualizationUpdate() {
      visualizationUpdateTimer = null;
      const fps = Math.max(1, Math.round(Number(els.fps.value) || 30));
      const speed = currentSpeedValue();
      const activityScale = els.activityScale.value === "simulation" ? "simulation" : "frame";
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(`visual:fps=${fps}&speed=${speed}&activity_scale=${activityScale}`);
        const scaleText = activityScale === "simulation" ? "simulation min/max" : "frame min/max";
        els.status.textContent = `Updated visualization: target ${fps} fps, target speed ${formatSpeed(speed)}, activity scale ${scaleText}. Simulation state preserved.`;
      }
    }

    function queueVisualizationUpdate(delayMs = 120) {
      if (visualizationUpdateTimer) clearTimeout(visualizationUpdateTimer);
      visualizationUpdateTimer = setTimeout(sendVisualizationUpdate, delayMs);
    }

    function startStream() {
      stopStream();
      applyGeometryDefaults();
      syncReflectControl();
      syncSpeedControls();
      resetMetrics();
      drawKernelGraph();
      drawFieldGraph();
      updateFramePanel();
      updateLegend();
      lastDisplayFrame = null;
      setPauseUi(false);
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${location.host}/stream?${streamParams().toString()}`;
      socket = new WebSocket(url);
      const currentSocket = socket;
      const token = ++streamToken;
      els.status.textContent = "Connecting...";
      lastFrameAt = performance.now();

      socket.onopen = () => {
        if (token !== streamToken || socket !== currentSocket) return;
        els.status.textContent = "Streaming. Use Pause to hold the current state or Reset to restart with new parameters.";
      };

      socket.onmessage = (event) => {
        if (token !== streamToken || socket !== currentSocket) return;
        const msg = JSON.parse(event.data);
        if (msg.type === "hello") {
          const duty = msg.dutyCycle === null ? "default" : `${msg.dutyCycle.toFixed(1)}% duty`;
          const coupling = msg.coupling === "overlap" ? `, overlap g=${msg.couplingStrength}` : (msg.coupling === "no_connection" || msg.fieldGeometry === "double_sech") ? ", two hemispheres no connection" : "";
          if (msg.speed === 0) {
            els.maxSpeed.checked = true;
          } else {
            els.maxSpeed.checked = false;
            els.speed.value = String(msg.speed);
          }
          syncSpeedControls();
          streamStimulus = {
            A: Number(msg.A) || 0,
            period: Number(msg.T) || 1,
            duty: msg.dutyCycle === null ? Number(els.duty.value) || 50 : Number(msg.dutyCycle)
          };
          drawStimulusGraph(0);
          const speedText = msg.speed === 0 ? "max speed" : `${formatSpeed(msg.speed)} speed`;
          if (msg.activityScale) els.activityScale.value = msg.activityScale;
          const scaleText = msg.activityScale === "simulation" ? "simulation min/max" : "frame min/max";
          const geometryText = msg.fieldGeometry === "double_sech" ? `, double-sech V1 density ${msg.fieldDensity}` : "";
          const boundaryText = msg.boundaryX === msg.boundaryY ? `boundary:${msg.boundaryX}` : `x:${msg.boundaryX} y:${msg.boundaryY}`;
          const reflectText = (msg.boundaryX === "partial_reflect" || msg.boundaryY === "partial_reflect") ? ` reflect=${msg.partialReflectStrength}` : "";
          els.status.textContent = `Streaming ${msg.backend}/${msg.conv} ${boundaryText}${reflectText}${geometryText}, \u03c3\u2091=${msg.Se}, \u03c3\u1d62=${msg.Si}, dt=${msg.dt} ms, target ${msg.fps} fps, target ${speedText}, ${scaleText}, ${duty}${coupling}.`;
          return;
        }
        if (msg.type === "done") {
          els.status.textContent = `Stream finished after ${msg.frames} frames.`;
          return;
        }
        if (msg.type === "error") {
          els.status.textContent = `Stream error: ${msg.message}`;
          stopStream();
          return;
        }
        if (msg.type !== "frame") return;

        const now = performance.now();
        const observedFps = 1000 / Math.max(1, now - lastFrameAt);
        const actualRealtimeX =
          lastRateFrameAt === null || lastRateSimTime === null
            ? 0
            : (msg.t - lastRateSimTime) / Math.max(1, now - lastRateFrameAt);
        lastFrameAt = now;
        lastRateFrameAt = now;
        lastRateSimTime = msg.t;
        const values = decodeFrame(msg.data);
        const rows = msg.rows || msg.N;
        const cols = msg.cols || msg.N;
        const retinalValues = decodeFrame(msg.retinalData || msg.data);
        const retinalRows = msg.retinalRows || msg.retinalN || msg.N;
        const retinalCols = msg.retinalCols || msg.retinalN || msg.N;
        const phaseEValues = msg.phaseEData ? decodeFrame(msg.phaseEData) : new Uint8Array();
        const phaseIValues = msg.phaseIData ? decodeFrame(msg.phaseIData) : new Uint8Array();
        lastDisplayFrame = {
          values,
          rows,
          cols,
          retinalValues,
          retinalRows,
          retinalCols,
          phaseEValues,
          phaseIValues,
          phaseCount: msg.phaseCount || phaseEValues.length,
          t: msg.t
        };
        drawCurrentFrame();
        els.simTime.textContent = formatSimTime(msg.t);
        els.streamFps.textContent = observedFps.toFixed(1);
        els.msStep.textContent = msg.msPerStep.toFixed(3);
        els.rtx.textContent = actualRealtimeX.toFixed(2);
        els.legendLow.textContent = msg.min.toFixed(3);
        els.legendHigh.textContent = msg.max.toFixed(3);
        drawStimulusGraph(msg.t);
      };

      socket.onclose = () => {
        if (token !== streamToken) return;
        if (!resetting) {
          els.status.textContent = "Paused/disconnected. Press Play to start a stream.";
          setPauseUi(true);
        }
        if (socket === currentSocket) socket = null;
      };

      socket.onerror = () => {
        if (token !== streamToken || socket !== currentSocket) return;
        els.status.textContent = "Stream error. Check the Julia terminal for details.";
      };
    }

    function stopStream() {
      if (!socket) return null;
      const closingSocket = socket;
      socket = null;
      try { closingSocket.send("close"); } catch (_) {}
      try { closingSocket.close(); } catch (_) {}
      return closingSocket;
    }

    function resetStream() {
      if (resetFallbackTimer) clearTimeout(resetFallbackTimer);
      resetting = true;
      resetMetrics();
      setPauseUi(false);
      els.status.textContent = "Resetting stream...";

      const closingSocket = socket;
      if (!closingSocket || closingSocket.readyState === WebSocket.CLOSED) {
        resetting = false;
        startStream();
        return;
      }

      socket = null;
      streamToken += 1;
      let restarted = false;
      const restart = () => {
        if (restarted) return;
        restarted = true;
        if (resetFallbackTimer) clearTimeout(resetFallbackTimer);
        resetFallbackTimer = null;
        resetting = false;
        startStream();
      };

      closingSocket.onmessage = () => {};
      closingSocket.onclose = restart;
      closingSocket.onerror = restart;
      try { closingSocket.send("close"); } catch (_) {}
      try { closingSocket.close(); } catch (_) { restart(); }
      resetFallbackTimer = setTimeout(restart, 1200);
    }

    function togglePausePlay() {
      if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
        startStream();
        return;
      }
      if (paused) {
        socket.send("play");
        setPauseUi(false);
        els.status.textContent = "Streaming resumed.";
      } else {
        socket.send("pause");
        setPauseUi(true);
        els.status.textContent = "Paused. The simulation state is held on the Julia side.";
      }
    }

    els.pausePlay.addEventListener("click", togglePausePlay);
    els.reset.addEventListener("click", resetStream);
    els.fieldGeometry.addEventListener("change", () => {
      applyGeometryDefaults();
      resetStream();
    });
    [els.backend, els.boundary, els.boundaryX, els.boundaryY].forEach((el) => {
      el.addEventListener("change", syncConvolutionControls);
    });
    els.fps.addEventListener("input", () => queueVisualizationUpdate(160));
    els.speed.addEventListener("change", () => queueVisualizationUpdate(0));
    els.speed.addEventListener("input", () => queueVisualizationUpdate(180));
    els.maxSpeed.addEventListener("change", () => {
      syncSpeedControls();
      queueVisualizationUpdate(0);
    });
    els.activityScale.addEventListener("change", () => queueVisualizationUpdate(0));
    els.frameSelect.addEventListener("change", () => {
      updateFramePanel();
    });
    [els.phaseIncludeAverage].forEach((el) => {
      el.addEventListener("change", drawPhasePlane);
    });
    [els.amp, els.period, els.duty].forEach((el) => {
      el.addEventListener("input", drawPhasePlane);
      el.addEventListener("change", drawPhasePlane);
    });
    document.querySelectorAll("[data-preset]").forEach((button) => {
      button.addEventListener("click", () => applyPreset(button.dataset.preset));
    });
    els.printParams.addEventListener("click", () => printParameters());
    els.colorMap.addEventListener("change", () => {
      updateLegend();
      drawCurrentFrame();
    });
    document.addEventListener("keydown", (event) => {
      const active = document.activeElement;
      const tag = active?.tagName;
      const inputType = active?.getAttribute?.("type") || "";
      const isTextEntry = active?.isContentEditable || tag === "TEXTAREA" || tag === "SELECT" || (tag === "INPUT" && inputType !== "number");
      if (event.code === "Space" && !event.repeat && !isTextEntry) {
        event.preventDefault();
        togglePausePlay();
      }
      if (event.code === "Enter" && !event.repeat && tag !== "TEXTAREA" && !active?.isContentEditable) {
        event.preventDefault();
        resetStream();
      }
    });
    [
      els.n, els.fieldDensity, els.kernelCutoff, els.se, els.si
    ].forEach((el) => el.addEventListener("input", drawKernelGraph));
    [
      els.n, els.fieldDensity, els.overlapRows, els.coupling, els.fieldGeometry
    ].forEach((el) => {
      el.addEventListener("input", drawFieldGraph);
      el.addEventListener("change", drawFieldGraph);
    });
    [
      els.coupling, els.overlapRows, els.couplingStrength
    ].forEach((el) => el.addEventListener("change", resetStream));
    window.addEventListener("resize", () => {
      drawKernelGraph();
      drawStimulusGraph(lastDisplayFrame?.t || 0);
      drawFieldGraph();
      drawPhasePlane();
    });
    syncSpeedControls();
    syncReflectControl();
    drawKernelGraph();
    drawFieldGraph();
    updateFramePanel();
    updateLegend();
    startStream();
  </script>
</body>
</html>
"""
