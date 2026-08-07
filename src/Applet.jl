using Base64
using HTTP

Base.@kwdef struct LiveConfig
    N::Int = 81
    fast_n::Bool = true
    backend::Symbol = :metal
    convolution::Symbol = :auto
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
    coupling::Symbol = :none
    coupling_strength::Float32 = 0.02f0
    overlap_rows::Int = 6
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
end

function normalize_live_config(config::LiveConfig)
    backend = config.backend == :gpu ? :metal : config.backend
    backend in (:cpu, :metal) || throw(ArgumentError("backend must be :cpu or :metal."))
    boundary_x, boundary_y = _resolve_boundaries(config.boundary, config.boundary_x, config.boundary_y)

    convolution = if config.convolution == :auto
        _default_convolution(backend, boundary_x, boundary_y)
    else
        config.convolution
    end
    convolution in (:fft, :separable) || throw(ArgumentError("convolution must be :auto, :fft, or :separable."))
    backend == :metal || convolution == :fft ||
        throw(ArgumentError("The CPU live backend currently supports FFT convolution only."))
    _validate_boundaries(boundary_x, boundary_y, convolution, backend)
    coupling = _normalize_live_coupling(config.coupling)

    target_fps = max(1, config.target_fps)
    N = config.fast_n ? next_fast_odd_size(config.N) : odd_positive_int(config.N)
    config.speed >= 0 || throw(ArgumentError("speed must be non-negative."))
    config.dt > 0 || throw(ArgumentError("dt must be positive."))
    config.gpu_threads > 0 || throw(ArgumentError("gpu_threads must be positive."))
    config.kernel_cutoff > 0 || throw(ArgumentError("kernel_cutoff must be positive."))
    config.max_frames >= 0 || throw(ArgumentError("max_frames must be non-negative."))
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
        coupling=coupling,
        coupling_strength=config.coupling_strength,
        overlap_rows=overlap_rows,
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
    return config.coupling in (:no_connection, :overlap)
end

function _uses_overlap_coupling(config::LiveConfig)
    return config.coupling == :overlap && config.coupling_strength > 0
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

    return normalize_live_config(LiveConfig(
        N=_parse_int(params, "N", 81),
        fast_n=_parse_bool(_get(params, "fast_n", "true"), true),
        backend=_parse_symbol(params, "backend", :metal),
        convolution=_parse_symbol(params, "conv", :auto),
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
        coupling=_parse_symbol(params, "coupling", :none),
        coupling_strength=_parse_float32(params, "coupling_strength", 0.02f0),
        overlap_rows=_parse_int(params, "overlap_rows", 6),
        max_frames=_parse_int(params, "max_frames", 0),
    ))
end

function _live_model_params(config::LiveConfig)
    return ModelParams{Float32}(dt=config.dt)
end

function _steps_per_frame(config::LiveConfig, p::ModelParams)
    target_frame_ms = 1000 / config.target_fps
    return max(1, round(Int, target_frame_ms / p.dt))
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

function _make_live_frame(
    activity::AbstractMatrix,
    frame_idx::Integer,
    t,
    step_ms::Float64,
    frame_start_ns::UInt64,
    steps_per_frame::Integer,
    p::ModelParams,
    retinal_activity::AbstractMatrix=activity,
)
    bytes, lo, hi = _activity_bytes(activity)
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

function _throttle!(stream_start_ns::UInt64, config::LiveConfig, sim_elapsed_ms::Float64)
    config.speed <= 0 && return
    target_elapsed = sim_elapsed_ms / (1000 * config.speed)
    actual_elapsed = (time_ns() - stream_start_ns) / 1e9
    delay = target_elapsed - actual_elapsed
    delay > 0.001 && sleep(delay)
    return
end

function _stream_cpu_frames(callback::Function, config::LiveConfig)
    _uses_two_hemispheres(config) && return _stream_cpu_coupled_frames(callback, config)
    p = _live_model_params(config)
    rng = _rng(config.seed)
    Ue = rand(rng, Float32, config.N, config.N)
    Ui = rand(rng, Float32, config.N, config.N)
    Ke = generate_gaussian_kernel(config.Se, config.N; dtype=Float32)
    Ki = generate_gaussian_kernel(config.Si, config.N; dtype=Float32)
    excitatory_convolver = FFTConvolver(Ke, Ue)
    inhibitory_convolver = FFTConvolver(Ki, Ui)
    Uec = similar(Ue)
    Uic = similar(Ui)
    noise = Array{Float32}(undef, 2, config.N, config.N)

    steps_per_frame = _steps_per_frame(config, p)
    stream_start = time_ns()
    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
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
            step_idx += 1
        end
        step_ms = (time_ns() - step_start) / 1e6
        activity = abs.(Ue .- Ui)
        retinal_activity = retinal_transform(activity)
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(activity, frame_idx, t, step_ms, frame_start, steps_per_frame, p, retinal_activity)
        callback(frame) === false && break
        _throttle!(stream_start, config, frame_idx * steps_per_frame * Float64(p.dt))
    end

    return frame_idx
end

function _stream_cpu_coupled_frames(callback::Function, config::LiveConfig)
    p = _live_model_params(config)
    rng = _rng(config.seed)
    N = config.N
    Ue_left = rand(rng, Float32, N, N)
    Ui_left = rand(rng, Float32, N, N)
    Ue_right = rand(rng, Float32, N, N)
    Ui_right = rand(rng, Float32, N, N)
    Ke = generate_gaussian_kernel(config.Se, N; dtype=Float32)
    Ki = generate_gaussian_kernel(config.Si, N; dtype=Float32)

    convolver_e_left = FFTConvolver(Ke, Ue_left)
    convolver_i_left = FFTConvolver(Ki, Ui_left)
    convolver_e_right = FFTConvolver(Ke, Ue_right)
    convolver_i_right = FFTConvolver(Ki, Ui_right)
    Uec_left = similar(Ue_left)
    Uic_left = similar(Ui_left)
    Uec_right = similar(Ue_right)
    Uic_right = similar(Ui_right)
    noise_left = Array{Float32}(undef, 2, N, N)
    noise_right = Array{Float32}(undef, 2, N, N)
    activity_left = similar(Ue_left)
    activity_right = similar(Ue_right)
    display_activity = Matrix{Float32}(undef, N, 2N)
    retinal_source = Matrix{Float32}(undef, 2N, N)

    steps_per_frame = _steps_per_frame(config, p)
    stream_start = time_ns()
    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
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
            if _uses_overlap_coupling(config)
                _apply_midline_coupling!(
                    Ue_left,
                    Ui_left,
                    Ue_right,
                    Ui_right,
                    config.coupling_strength,
                    config.overlap_rows,
                )
            end
            step_idx += 1
        end
        step_ms = (time_ns() - step_start) / 1e6
        @. activity_left = abs(Ue_left - Ui_left)
        @. activity_right = abs(Ue_right - Ui_right)
        _fill_coupled_views!(display_activity, activity_left, activity_right)
        _fill_coupled_retinal_source!(retinal_source, activity_left, activity_right)
        retinal_activity = retinal_transform(retinal_source; output_size=(N, N), angle_origin=Float32(pi / 2))
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(
            display_activity,
            frame_idx,
            t,
            step_ms,
            frame_start,
            steps_per_frame,
            p,
            retinal_activity,
        )
        callback(frame) === false && break
        _throttle!(stream_start, config, frame_idx * steps_per_frame * Float64(p.dt))
    end

    return frame_idx
end

function _stream_metal_frames(callback::Function, config::LiveConfig)
    _uses_two_hemispheres(config) && return _stream_metal_coupled_frames(callback, config)
    Metal.functional() || throw(ErrorException("Metal.jl is not functional on this machine."))
    p = _live_model_params(config)
    config.seed === nothing || Metal.seed!(config.seed)

    Ue = Metal.rand(Float32, config.N, config.N)
    Ui = Metal.rand(Float32, config.N, config.N)
    excitatory_convolver = if config.convolution == :fft
        Ke = generate_gaussian_kernel(config.Se, config.N; dtype=Float32)
        MetalFFTConvolver(Ke, Ue)
    else
        MetalSeparableConvolver(config.Se, Ue; cutoff=config.kernel_cutoff)
    end
    inhibitory_convolver = if config.convolution == :fft
        Ki = generate_gaussian_kernel(config.Si, config.N; dtype=Float32)
        MetalFFTConvolver(Ki, Ui)
    else
        MetalSeparableConvolver(config.Si, Ui; cutoff=config.kernel_cutoff)
    end

    Uec = similar(Ue)
    Uic = similar(Ui)
    noise_E = similar(Ue)
    noise_I = similar(Ui)
    cortical_gpu = similar(Ue)

    steps_per_frame = _steps_per_frame(config, p)
    stream_start = time_ns()
    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
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
                    config.duty_cycle_percent,
                )
            end
            step_idx += 1
        end
        Metal.synchronize()
        step_ms = (time_ns() - step_start) / 1e6

        cortical_gpu .= abs.(Ue .- Ui)
        Metal.synchronize()
        activity = Array(cortical_gpu)
        retinal_activity = retinal_transform(activity)
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(activity, frame_idx, t, step_ms, frame_start, steps_per_frame, p, retinal_activity)
        callback(frame) === false && break
        _throttle!(stream_start, config, frame_idx * steps_per_frame * Float64(p.dt))
    end

    return frame_idx
end

function _stream_metal_coupled_frames(callback::Function, config::LiveConfig)
    Metal.functional() || throw(ErrorException("Metal.jl is not functional on this machine."))
    p = _live_model_params(config)
    config.seed === nothing || Metal.seed!(config.seed)
    N = config.N

    Ue_left = Metal.rand(Float32, N, N)
    Ui_left = Metal.rand(Float32, N, N)
    Ue_right = Metal.rand(Float32, N, N)
    Ui_right = Metal.rand(Float32, N, N)

    if config.convolution == :fft
        Ke = generate_gaussian_kernel(config.Se, N; dtype=Float32)
        Ki = generate_gaussian_kernel(config.Si, N; dtype=Float32)
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
    activity_left_gpu = similar(Ue_left)
    activity_right_gpu = similar(Ue_right)
    display_activity = Matrix{Float32}(undef, N, 2N)
    retinal_source = Matrix{Float32}(undef, 2N, N)

    steps_per_frame = _steps_per_frame(config, p)
    stream_start = time_ns()
    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
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
                    config.duty_cycle_percent,
                )
            end

            if _uses_overlap_coupling(config)
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
            step_idx += 1
        end
        Metal.synchronize()
        step_ms = (time_ns() - step_start) / 1e6

        activity_left_gpu .= abs.(Ue_left .- Ui_left)
        activity_right_gpu .= abs.(Ue_right .- Ui_right)
        Metal.synchronize()
        activity_left = Array(activity_left_gpu)
        activity_right = Array(activity_right_gpu)
        _fill_coupled_views!(display_activity, activity_left, activity_right)
        _fill_coupled_retinal_source!(retinal_source, activity_left, activity_right)
        retinal_activity = retinal_transform(retinal_source; output_size=(N, N), angle_origin=Float32(pi / 2))
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(
            display_activity,
            frame_idx,
            t,
            step_ms,
            frame_start,
            steps_per_frame,
            p,
            retinal_activity,
        )
        callback(frame) === false && break
        _throttle!(stream_start, config, frame_idx * steps_per_frame * Float64(p.dt))
    end

    return frame_idx
end

function stream_live_frames(callback::Function, config::LiveConfig=LiveConfig())
    normalized = normalize_live_config(config)
    if normalized.backend == :metal
        return _stream_metal_frames(callback, normalized)
    else
        return _stream_cpu_frames(callback, normalized)
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
        ",\"coupling\":", _json_string(config.coupling),
        ",\"couplingStrength\":", _json_number(config.coupling_strength; digits=5),
        ",\"overlapRows\":", config.overlap_rows,
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

function _stream_websocket(ws, params)
    paused = Ref(false)
    closed = Ref(false)
    control_task = @async try
        while !closed[]
            message = String(HTTP.WebSockets.receive(ws))
            if message == "pause"
                paused[] = true
            elseif message == "play"
                paused[] = false
            elseif message == "close"
                closed[] = true
                break
            end
        end
    catch err
        if !(err isa HTTP.WebSockets.WebSocketError || err isa EOFError || err isa IOError)
            @warn "Applet control channel failed." exception=(err, catch_backtrace())
        end
        closed[] = true
    end

    try
        config = live_config_from_query(params)
        _safe_ws_send(ws, _hello_json(config)) || return
        frames = stream_live_frames(config) do frame
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
  <title>RSE Real-Time Viewer</title>
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
      grid-template-columns: 330px 1fr;
      gap: 22px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 54px);
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
      margin: 0 0 22px;
      line-height: 1.45;
      font-size: 14px;
    }

    .control-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.04em;
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
      padding: 10px 0 2px;
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

    .status {
      margin-top: 18px;
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
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 8px;
    }

    .metric {
      border: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(246, 251, 252, 0.92));
      border-radius: 16px;
      padding: 9px 10px;
      min-width: 0;
      box-shadow: var(--soft-shadow);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 9.5px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: clamp(14px, 1.6vw, 21px);
      letter-spacing: -0.045em;
      white-space: nowrap;
    }

    .views {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      min-width: 0;
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
      margin-bottom: 10px;
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
      border-radius: 18px;
      image-rendering: pixelated;
      background: #071018;
      box-shadow: inset 0 0 0 1px rgba(13, 38, 56, 0.12);
    }

    .view canvas {
      aspect-ratio: 1 / 1;
    }

    .canvas-frame {
      position: relative;
      padding: 54px 44px 44px;
      border-radius: 22px;
      background:
        radial-gradient(circle at 50% 18%, rgba(0, 158, 170, 0.07), transparent 16rem),
        linear-gradient(180deg, #fbfdfe, #f3f8fa);
      border: 1px solid rgba(198, 216, 226, 0.72);
    }

    .canvas-frame canvas {
      width: 100%;
    }

    .axis-label,
    .hemi-label {
      position: absolute;
      z-index: 2;
      color: #33495c;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.06em;
      line-height: 1;
      pointer-events: none;
      text-transform: uppercase;
    }

    .axis-label {
      color: #607284;
      font-variant-numeric: tabular-nums;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(198, 216, 226, 0.82);
      border-radius: 999px;
      padding: 4px 7px;
      box-shadow: 0 6px 18px rgba(25, 56, 82, 0.07);
    }

    .hemi-label {
      color: #0b3146;
      background: rgba(231, 246, 248, 0.92);
      border: 1px solid rgba(0, 158, 170, 0.18);
      border-radius: 999px;
      padding: 5px 8px;
    }

    .cortical-frame {
      --left-center: 50%;
      --right-center: 50%;
      padding-top: 76px;
    }

    .hemi-left,
    .axis-top-left,
    .axis-bottom-left {
      left: var(--left-center);
      transform: translateX(-50%);
    }

    .hemi-right,
    .axis-top-right,
    .axis-bottom-right {
      left: var(--right-center);
      transform: translateX(-50%);
    }

    .hemi-label {
      top: 12px;
    }

    .axis-top-left,
    .axis-top-right {
      top: 50px;
    }

    .axis-bottom-left,
    .axis-bottom-right {
      bottom: 13px;
    }

    .cortical-frame:not(.coupled) .hemi-right,
    .cortical-frame:not(.coupled) .axis-top-right,
    .cortical-frame:not(.coupled) .axis-bottom-right {
      display: none;
    }

    .retinal-angle-90 {
      top: 14px;
      left: 50%;
      transform: translateX(-50%);
    }

    .retinal-angle-270 {
      bottom: 14px;
      left: 50%;
      transform: translateX(-50%);
    }

    .retinal-angle-0 {
      right: 12px;
      top: 50%;
      transform: translateY(-50%);
    }

    .retinal-angle-180 {
      left: 12px;
      top: 50%;
      transform: translateY(-50%);
    }

    .kernel-card {
      margin-top: 16px;
      padding: 13px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: #ffffff;
    }

    .kernel-canvas {
      aspect-ratio: 2.15 / 1;
      background: #f8fbfd;
      image-rendering: auto;
    }

    .tiny-note {
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }

    .legend {
      height: 10px;
      border-radius: 999px;
      background: var(--legend-gradient);
      border: 1px solid var(--line);
      box-shadow: var(--soft-shadow);
    }

    @media (max-width: 940px) {
      main { grid-template-columns: 1fr; padding: 18px; }
      .panel { position: static; }
      .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .views { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h2>RSE Model</h2>
      <h1>Real-time field viewer</h1>
      <p class="subtitle">Streams the Julia simulation into the browser. GPU mode keeps the model on Metal and sends only display frames.</p>
      <div class="control-grid">
        <label>N<input id="n" type="number" min="5" step="2" value="81"></label>
        <label>FPS<input id="fps" type="number" min="1" max="60" step="1" value="30"></label>
        <label>Backend<select id="backend"><option value="metal">metal</option><option value="cpu">cpu</option></select></label>
        <label>Convolution<select id="conv"><option value="auto">auto</option><option value="separable">separable</option><option value="fft">fft</option></select></label>
        <label>Boundary X<select id="boundaryX"><option value="periodic">periodic</option><option value="edge">edge</option><option value="zero">zero</option></select></label>
        <label>Boundary Y<select id="boundaryY"><option value="periodic">periodic</option><option value="edge">edge</option><option value="zero">zero</option></select></label>
        <label>Coupling<select id="coupling"><option value="off">off</option><option value="no_connection">no connection</option><option value="overlap">overlap</option></select></label>
        <label>Speed<select id="speed"><option value="1">1x real time</option><option value="0.5">0.5x</option><option value="2">2x</option><option value="0">max</option></select></label>
        <label>A<input id="amp" type="number" min="0" step="0.05" value="0.7"></label>
        <label>T (ms)<input id="period" type="number" min="1" step="1" value="115"></label>
        <label>Duty (%)<input id="duty" type="number" min="1" max="99" step="0.5" value="20.5"></label>
        <label>Se<input id="se" type="number" min="0.1" step="0.05" value="2"></label>
        <label>Si<input id="si" type="number" min="0.1" step="0.05" value="5"></label>
        <label>dt (ms)<input id="dt" type="number" min="0.01" step="0.05" value="0.2"></label>
        <label>Colormap<select id="colorMap"><option value="plasma">plasma</option><option value="viridis">viridis</option><option value="magma">magma</option><option value="inferno">inferno</option><option value="cividis">cividis</option><option value="turbo">turbo</option><option value="nipy_spectral">nipy_spectral</option><option value="gray">gray</option></select></label>
        <label>Kernel cutoff<input id="kernelCutoff" type="number" min="0.5" max="6" step="0.25" value="3"></label>
        <label>Coupling g<input id="couplingStrength" type="number" min="0" max="0.5" step="0.005" value="0.02"></label>
        <label>Overlap rows<input id="overlapRows" type="number" min="2" step="2" value="6"></label>
      </div>
      <label class="check-row"><input id="fastN" type="checkbox" checked> Snap to FFT-friendly odd N</label>
      <label>Seed<input id="seed" type="number" step="1" placeholder="optional"></label>
      <div class="kernel-card">
        <div class="view-head"><div class="view-title">Kernel window</div><div class="view-note" id="kernelInfo">-</div></div>
        <canvas id="kernelGraph" class="kernel-canvas"></canvas>
        <div class="tiny-note">Lines show the excitatory and inhibitory 1D Gaussian kernels. The applet uses the separable product of these kernels in x and y.</div>
      </div>
      <div class="button-row">
        <button id="pausePlay" class="pause">Pause</button>
        <button id="reset" class="secondary">Reset</button>
      </div>
      <div id="status" class="status">Streaming starts automatically. Use Pause or Reset while tuning parameters.</div>
    </section>

    <section class="stage">
      <div class="metrics">
        <div class="metric"><span>Sim time</span><strong id="simTime">0 ms</strong></div>
        <div class="metric"><span>Stream FPS</span><strong id="streamFps">0</strong></div>
        <div class="metric"><span>ms / step</span><strong id="msStep">0</strong></div>
        <div class="metric"><span>dt</span><strong id="dtMetric">0.2 ms</strong></div>
        <div class="metric"><span>Real-time x</span><strong id="rtx">0</strong></div>
        <div class="metric"><span>Grid</span><strong id="gridN">-</strong></div>
      </div>
      <div class="views">
        <div class="view">
          <div class="view-head"><div class="view-title">Cortical sheet</div><div class="view-note" id="range">range -</div></div>
          <div id="corticalFrame" class="canvas-frame cortical-frame">
            <canvas id="cortical"></canvas>
            <span id="hemiLeft" class="hemi-label hemi-left">Cortical sheet</span>
            <span id="hemiRight" class="hemi-label hemi-right">Right hemisphere</span>
            <span class="axis-label axis-top-left">90&deg;</span>
            <span class="axis-label axis-bottom-left">270&deg;</span>
            <span class="axis-label axis-top-right">90&deg;</span>
            <span class="axis-label axis-bottom-right">270&deg;</span>
          </div>
        </div>
        <div class="view">
          <div class="view-head"><div class="view-title">Retinal view</div><div class="view-note">server-side log-polar map</div></div>
          <div class="canvas-frame retinal-frame">
            <canvas id="retinal"></canvas>
            <span class="axis-label retinal-angle-90">90&deg;</span>
            <span class="axis-label retinal-angle-0">0&deg;</span>
            <span class="axis-label retinal-angle-180">180&deg;</span>
            <span class="axis-label retinal-angle-270">270&deg;</span>
          </div>
        </div>
      </div>
      <div id="legend" class="legend"></div>
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
      boundaryX: document.getElementById("boundaryX"),
      boundaryY: document.getElementById("boundaryY"),
      coupling: document.getElementById("coupling"),
      speed: document.getElementById("speed"),
      kernelCutoff: document.getElementById("kernelCutoff"),
      amp: document.getElementById("amp"),
      period: document.getElementById("period"),
      duty: document.getElementById("duty"),
      couplingStrength: document.getElementById("couplingStrength"),
      overlapRows: document.getElementById("overlapRows"),
      colorMap: document.getElementById("colorMap"),
      se: document.getElementById("se"),
      si: document.getElementById("si"),
      dt: document.getElementById("dt"),
      fastN: document.getElementById("fastN"),
      seed: document.getElementById("seed"),
      pausePlay: document.getElementById("pausePlay"),
      reset: document.getElementById("reset"),
      status: document.getElementById("status"),
      simTime: document.getElementById("simTime"),
      streamFps: document.getElementById("streamFps"),
      msStep: document.getElementById("msStep"),
      dtMetric: document.getElementById("dtMetric"),
      rtx: document.getElementById("rtx"),
      gridN: document.getElementById("gridN"),
      range: document.getElementById("range"),
      corticalFrame: document.getElementById("corticalFrame"),
      hemiLeft: document.getElementById("hemiLeft"),
      hemiRight: document.getElementById("hemiRight"),
      cortical: document.getElementById("cortical"),
      retinal: document.getElementById("retinal"),
      kernelGraph: document.getElementById("kernelGraph"),
      kernelInfo: document.getElementById("kernelInfo"),
      legend: document.getElementById("legend")
    };

    let socket = null;
    let paused = false;
    let resetting = false;
    let streamToken = 0;
    let lastFrameAt = performance.now();
    let lastDisplayFrame = null;

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

    function updateCorticalLabels(isCoupled, leftCenter = 50, rightCenter = 50) {
      els.corticalFrame.classList.toggle("coupled", isCoupled);
      els.corticalFrame.style.setProperty("--left-center", `${leftCenter}%`);
      els.corticalFrame.style.setProperty("--right-center", `${rightCenter}%`);
      els.hemiLeft.textContent = isCoupled ? "Left hemisphere" : "Cortical sheet";
      els.hemiRight.textContent = "Right hemisphere";
    }

    function drawCortical(canvas, values, rows, cols) {
      const isCoupled = cols >= rows * 1.5;
      if (!isCoupled) {
        updateCorticalLabels(false);
        drawValues(canvas, values, rows, cols);
        return;
      }

      const hemiCols = Math.floor(cols / 2);
      const gapCols = Math.max(6, Math.round(rows * 0.08));
      const drawCols = cols + gapCols;
      setCanvasSize(canvas, rows, drawCols);
      updateCorticalLabels(
        true,
        (hemiCols / 2) / drawCols * 100,
        (hemiCols + gapCols + hemiCols / 2) / drawCols * 100
      );

      const ctx = canvas.getContext("2d");
      const image = ctx.createImageData(drawCols, rows);
      for (let i = 0; i < image.data.length; i += 4) {
        image.data[i] = 248;
        image.data[i + 1] = 251;
        image.data[i + 2] = 253;
        image.data[i + 3] = 255;
      }

      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const targetCol = col < hemiCols ? col : col + gapCols;
          setPixel(image, row * drawCols + targetCol, values[row * cols + col]);
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
      els.dtMetric.textContent = `${Number(els.dt.value).toFixed(3)} ms`;
      els.rtx.textContent = "0";
      els.gridN.textContent = "-";
      els.range.textContent = "range -";
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

      const n = Math.max(5, Number(els.n.value) || 101);
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
      ctx.fillStyle = "#f8fbfd";
      ctx.fillRect(0, 0, width, height);
      const padL = 38 * dpr, padR = 18 * dpr, padT = 18 * dpr, padB = 30 * dpr;
      const plotW = width - padL - padR;
      const plotH = height - padT - padB;
      const xToPx = (x) => padL + ((x + maxRadius) / (2 * maxRadius)) * plotW;
      const yToPx = (v) => padT + (1 - v / maxValue) * plotH;

      ctx.strokeStyle = "#dbe7ef";
      ctx.lineWidth = 1 * dpr;
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
      ctx.fillText(`-r`, padL, height - 10 * dpr);
      ctx.fillText(`0`, xToPx(0) - 3 * dpr, height - 10 * dpr);
      ctx.fillText(`+r`, padL + plotW - 12 * dpr, height - 10 * dpr);
      ctx.fillStyle = "#009eaa";
      ctx.fillText("Se", padL + 8 * dpr, padT + 14 * dpr);
      ctx.fillStyle = "#f3b33d";
      ctx.fillText("Si", padL + 36 * dpr, padT + 14 * dpr);
      els.kernelInfo.textContent =
        `E r=${radiusE}, I r=${radiusI}; retained ${retainedE.toFixed(3)}% / ${retainedI.toFixed(3)}%`;
    }

    function drawRetinal(canvas, values, rows, cols) {
      drawValues(canvas, values, rows, cols);
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
    }

    function decodeFrame(data) {
      const binary = atob(data);
      const values = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) values[i] = binary.charCodeAt(i);
      return values;
    }

    function streamParams() {
      const params = new URLSearchParams();
      params.set("N", els.n.value);
      params.set("fps", els.fps.value);
      params.set("backend", els.backend.value);
      params.set("conv", els.conv.value);
      params.set("boundary_x", els.boundaryX.value);
      params.set("boundary_y", els.boundaryY.value);
      params.set("coupling", els.coupling.value);
      params.set("speed", els.speed.value);
      params.set("kernel_cutoff", els.kernelCutoff.value);
      params.set("A", els.amp.value);
      params.set("T", els.period.value);
      params.set("duty_cycle", els.duty.value);
      params.set("coupling_strength", els.couplingStrength.value);
      params.set("overlap_rows", els.overlapRows.value);
      params.set("Se", els.se.value);
      params.set("Si", els.si.value);
      params.set("dt", els.dt.value);
      params.set("fast_n", els.fastN.checked ? "true" : "false");
      if (els.seed.value.trim()) params.set("seed", els.seed.value.trim());
      return params;
    }

    function startStream() {
      stopStream();
      resetMetrics();
      drawKernelGraph();
      updateLegend();
      lastDisplayFrame = null;
      setPauseUi(false);
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${location.host}/stream?${streamParams().toString()}`;
      socket = new WebSocket(url);
      const token = ++streamToken;
      els.status.textContent = "Connecting...";
      lastFrameAt = performance.now();

      socket.onopen = () => {
        els.status.textContent = "Streaming. Use Pause to hold the current state or Reset to restart with new parameters.";
      };

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "hello") {
          els.gridN.textContent = String(msg.N);
          const duty = msg.dutyCycle === null ? "default" : `${msg.dutyCycle.toFixed(1)}% duty`;
          const coupling = msg.coupling === "overlap" ? `, overlap g=${msg.couplingStrength}` : msg.coupling === "no_connection" ? ", two hemispheres no connection" : "";
          els.dtMetric.textContent = `${Number(msg.dt).toFixed(3)} ms`;
          els.status.textContent = `Streaming ${msg.backend}/${msg.conv} x:${msg.boundaryX} y:${msg.boundaryY}, Se=${msg.Se}, Si=${msg.Si}, dt=${msg.dt} ms, target ${msg.fps} fps, ${duty}${coupling}.`;
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
        lastFrameAt = now;
        const values = decodeFrame(msg.data);
        const rows = msg.rows || msg.N;
        const cols = msg.cols || msg.N;
        const retinalValues = decodeFrame(msg.retinalData || msg.data);
        const retinalRows = msg.retinalRows || msg.retinalN || msg.N;
        const retinalCols = msg.retinalCols || msg.retinalN || msg.N;
        lastDisplayFrame = { values, rows, cols, retinalValues, retinalRows, retinalCols };
        drawCurrentFrame();
        els.simTime.textContent = `${msg.t.toFixed(1)} ms`;
        els.streamFps.textContent = observedFps.toFixed(1);
        els.msStep.textContent = msg.msPerStep.toFixed(3);
        els.rtx.textContent = msg.realtimeX.toFixed(2);
        els.gridN.textContent = String(msg.N);
        els.range.textContent = `${msg.min.toFixed(3)} to ${msg.max.toFixed(3)}`;
      };

      socket.onclose = () => {
        if (token !== streamToken) return;
        if (!resetting) {
          els.status.textContent = "Paused/disconnected. Press Play to start a stream.";
          setPauseUi(true);
        }
        socket = null;
      };

      socket.onerror = () => {
        els.status.textContent = "Stream error. Check the Julia terminal for details.";
      };
    }

    function stopStream() {
      if (socket) {
        try { socket.send("close"); } catch (_) {}
        socket.close();
        socket = null;
      }
    }

    function resetStream() {
      resetting = true;
      stopStream();
      resetMetrics();
      setTimeout(() => {
        resetting = false;
        startStream();
      }, 60);
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
    els.colorMap.addEventListener("change", () => {
      updateLegend();
      drawCurrentFrame();
    });
    document.addEventListener("keydown", (event) => {
      const active = document.activeElement;
      const tag = active?.tagName;
      const isTyping = active?.isContentEditable || tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || tag === "BUTTON";
      if (event.code === "Space" && !event.repeat && !isTyping) {
        event.preventDefault();
        togglePausePlay();
      }
    });
    [
      els.n, els.kernelCutoff, els.se, els.si
    ].forEach((el) => el.addEventListener("input", drawKernelGraph));
    window.addEventListener("resize", drawKernelGraph);
    drawKernelGraph();
    updateLegend();
    startStream();
  </script>
</body>
</html>
"""
