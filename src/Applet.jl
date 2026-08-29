using Base64
using HTTP

# Web assets are loaded once when the package starts; no frontend build step is required.
const WEB_ROOT = normpath(joinpath(@__DIR__, "..", "web"))
const APPLET_HTML_PATH = joinpath(WEB_ROOT, "index.html")
const APPLET_CSS_PATH = joinpath(WEB_ROOT, "styles.css")
const APPLET_JS_PATH = joinpath(WEB_ROOT, "app.js")
Base.include_dependency(APPLET_HTML_PATH)
Base.include_dependency(APPLET_CSS_PATH)
Base.include_dependency(APPLET_JS_PATH)
const APPLET_HTML = read(APPLET_HTML_PATH, String)
const APPLET_CSS = read(APPLET_CSS_PATH, String)
const APPLET_JS = read(APPLET_JS_PATH, String)
const ACTIVITY_SCALE_WARMUP_MS = 500.0

# Configuration and frame protocol

"""Validated configuration for one live web-app simulation stream."""
Base.@kwdef struct LiveConfig
    N::Int = 81
    fast_n::Bool = true
    backend::Symbol = :metal
    convolution::Symbol = :separable
    A::Float32 = 0.7f0
    period::Float32 = 115.0f0
    duty_cycle_percent::Union{Nothing,Float32} = Float32(duty_cycle_percent_from_threshold(ModelParams{Float32}().V))
    Ge::Float32 = 1.0f0
    Gi::Float32 = 0.0f0
    Se::Float32 = 2.0f0
    Si::Float32 = 5.0f0
    Aee::Float32 = 10.0f0
    Aei::Float32 = 12.0f0
    Aie::Float32 = 8.5f0
    Aii::Float32 = 3.0f0
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
    retinal_resolution::Int = 321
    retinal_rendering::Symbol = :interpolated
    activity_scale::Symbol = :frame
    max_frames::Int = 0
end

"""One encoded cortical, retinal, and phase-plane update sent to the browser."""
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

"""Visualization controls and adaptive timing state that can change without resetting the model."""
Base.@kwdef mutable struct LiveRuntime
    target_fps::Int = 30
    speed::Float64 = 1.0
    max_steps_per_frame::Int = 1
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
    retinal_rendering = _normalize_retinal_rendering(config.retinal_rendering)
    activity_scale = _normalize_activity_scale(config.activity_scale)

    target_fps = max(1, config.target_fps)
    retinal_resolution = odd_positive_int(clamp(config.retinal_resolution, 5, 801))
    N = if geometry_kind == :double_sech
        field_geometry(:double_sech; density=config.field_density).rows
    else
        config.fast_n ? next_fast_odd_size(config.N) : odd_positive_int(config.N)
    end
    config.speed >= 0 || throw(ArgumentError("speed must be non-negative."))
    config.dt > 0 || throw(ArgumentError("dt must be positive."))
    for (name, value) in (
        ("Aee", config.Aee), ("Aei", config.Aei), ("Aie", config.Aie), ("Aii", config.Aii),
        ("Ge", config.Ge), ("Gi", config.Gi),
    )
        isfinite(value) && value >= 0 || throw(ArgumentError("$name must be finite and non-negative."))
    end
    config.gpu_threads > 0 || throw(ArgumentError("gpu_threads must be positive."))
    config.kernel_cutoff > 0 || throw(ArgumentError("kernel_cutoff must be positive."))
    config.max_frames >= 0 || throw(ArgumentError("max_frames must be non-negative."))
    if config.seed !== nothing
        1 <= config.seed <= 999 || throw(ArgumentError("seed must be between 1 and 999."))
    end
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
        Ge=config.Ge,
        Gi=config.Gi,
        Se=config.Se,
        Si=config.Si,
        Aee=config.Aee,
        Aei=config.Aei,
        Aie=config.Aie,
        Aii=config.Aii,
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
        retinal_resolution=retinal_resolution,
        retinal_rendering=retinal_rendering,
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

function _normalize_retinal_rendering(rendering::Symbol)
    rendering in (:interpolated, :fast) && return :interpolated
    rendering in (:mapped, :precise, :direct) && return :mapped
    throw(ArgumentError("retinal_rendering must be :interpolated or :mapped."))
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
        Ge=_parse_float32(params, "Ge", 1.0f0),
        Gi=_parse_float32(params, "Gi", 0.0f0),
        Se=_parse_float32(params, "Se", 2.0f0),
        Si=_parse_float32(params, "Si", 5.0f0),
        Aee=_parse_float32(params, "Aee", 10.0f0),
        Aei=_parse_float32(params, "Aei", 12.0f0),
        Aie=_parse_float32(params, "Aie", 8.5f0),
        Aii=_parse_float32(params, "Aii", 3.0f0),
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
        retinal_resolution=_parse_int(params, "retinal_resolution", 321),
        retinal_rendering=_parse_symbol(params, "retinal_rendering", :interpolated),
        activity_scale=_parse_symbol(params, "activity_scale", :frame),
        max_frames=_parse_int(params, "max_frames", 0),
    ))
end

# Runtime timing and frame encoding

function _live_model_params(config::LiveConfig)
    return ModelParams{Float32}(
        dt=config.dt,
        Aee=config.Aee,
        Aei=config.Aei,
        Aie=config.Aie,
        Aii=config.Aii,
        Ge=config.Ge,
        Gi=config.Gi,
    )
end

function _live_runtime(config::LiveConfig)
    return LiveRuntime(
        target_fps=config.target_fps,
        speed=config.speed,
        max_steps_per_frame=max(1, round(Int, 1000 / (config.target_fps * config.dt))),
        activity_scale=config.activity_scale,
    )
end

function _live_field_geometry(config::LiveConfig)
    if config.field_geometry == :double_sech
        return field_geometry(:double_sech; density=config.field_density)
    end
    return field_geometry(:square, config.N)
end

function _steps_per_frame(target_fps::Integer, speed::Real, p::ModelParams)
    speed <= 0 && return 1
    target_frame_ms = 1000 / max(1, target_fps)
    return max(1, round(Int, target_frame_ms * Float64(speed) / p.dt))
end

function _steps_per_frame(runtime::LiveRuntime, p::ModelParams)
    runtime.speed <= 0 && return max(1, runtime.max_steps_per_frame)
    return _steps_per_frame(runtime.target_fps, runtime.speed, p)
end

function _update_max_steps_per_frame!(
    runtime::LiveRuntime,
    completed_steps::Integer,
    step_ms::Real,
    frame_ms::Real,
)
    runtime.speed <= 0 || return runtime.max_steps_per_frame
    completed_steps > 0 || return runtime.max_steps_per_frame
    step_ms > 0 || return runtime.max_steps_per_frame

    target_frame_ms = 1000 / max(1, runtime.target_fps)
    step_cost_ms = Float64(step_ms) / completed_steps
    nonstep_cost_ms = max(0.0, Float64(frame_ms) - Float64(step_ms))
    step_budget_ms = max(step_cost_ms, 0.95 * target_frame_ms - nonstep_cost_ms)
    ideal_steps = clamp(floor(Int, step_budget_ms / step_cost_ms), 1, 1_000_000)

    current_steps = max(1, runtime.max_steps_per_frame)
    runtime.max_steps_per_frame = ideal_steps < current_steps ?
        ideal_steps : min(ideal_steps, max(current_steps + 1, 2 * current_steps))
    return runtime.max_steps_per_frame
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

function _activity_scale_bounds!(runtime::Union{Nothing,LiveRuntime}, activity::AbstractMatrix, t)
    frame_lo = Float32(minimum(activity))
    frame_hi = Float32(maximum(activity))
    runtime === nothing && return frame_lo, frame_hi

    if runtime.activity_scale != :simulation || Float64(t) < ACTIVITY_SCALE_WARMUP_MS
        return frame_lo, frame_hi
    end

    runtime.scale_lo = min(runtime.scale_lo, frame_lo)
    runtime.scale_hi = max(runtime.scale_hi, frame_hi)
    return runtime.scale_lo, runtime.scale_hi
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
    scale_lo, scale_hi = _activity_scale_bounds!(runtime, activity, t)
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

function _retinal_output_size(config::LiveConfig)
    resolution = config.retinal_rendering == :interpolated ? config.N : config.retinal_resolution
    return (resolution, resolution)
end

function _coupled_retinal_plan(retinal_source, geometry::FieldGeometry, config::LiveConfig)
    output_size = _retinal_output_size(config)
    if geometry.kind == :double_sech
        return double_sech_retinal_plan(geometry; output_size=output_size)
    end

    return retinal_map_plan(
        size(retinal_source);
        output_size=output_size,
        angle_origin=Float32(pi / 2),
    )
end

function _fill_coupled_retinal_activity!(
    retinal_activity,
    left_activity,
    right_activity,
    retinal_source,
    plan::DoubleSechRetinalPlan,
)
    return double_sech_retinal_transform!(retinal_activity, left_activity, right_activity, plan)
end

function _fill_coupled_retinal_activity!(
    retinal_activity,
    left_activity,
    right_activity,
    retinal_source,
    plan::RetinalMapPlan,
)
    _fill_coupled_retinal_source!(retinal_source, left_activity, right_activity)
    return retinal_transform!(retinal_activity, retinal_source, plan)
end

function _reset_throttle!(runtime::LiveRuntime)
    runtime.throttle_deadline_ns = 0
    return
end

function _throttle!(runtime::LiveRuntime, frame_start_ns::UInt64)
    yield()
    interval_ns = UInt64(max(0, round(Int, 1e9 / max(1, runtime.target_fps))))
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

# CPU and Metal stream loops

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
    retinal_size = _retinal_output_size(config)
    retinal_activity = Matrix{Float32}(undef, retinal_size)
    retinal_plan = retinal_map_plan(size(Ue); output_size=retinal_size)

    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        steps_per_frame = _steps_per_frame(runtime, p)
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
        retinal_transform!(retinal_activity, activity, retinal_plan)
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
        _update_max_steps_per_frame!(runtime, steps_per_frame, step_ms, frame.frame_ms)
        callback(frame) === false && break
        _throttle!(runtime, frame_start)
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
    retinal_activity = Matrix{Float32}(undef, _retinal_output_size(config))
    retinal_plan = _coupled_retinal_plan(retinal_source, geometry, config)
    border_coupling_mask = has_field_mask(geometry) ?
        field_border_mask(geometry.mask, max(1, div(config.overlap_rows, 2))) : nothing

    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        steps_per_frame = _steps_per_frame(runtime, p)
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
        _fill_coupled_retinal_activity!(
            retinal_activity,
            activity_left,
            activity_right,
            retinal_source,
            retinal_plan,
        )
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
        _update_max_steps_per_frame!(runtime, steps_per_frame, step_ms, frame.frame_ms)
        callback(frame) === false && break
        _throttle!(runtime, frame_start)
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
    retinal_size = _retinal_output_size(config)
    retinal_activity = Matrix{Float32}(undef, retinal_size)
    retinal_plan = retinal_map_plan(size(Ue); output_size=retinal_size)
    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        steps_per_frame = _steps_per_frame(runtime, p)
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
        retinal_transform!(retinal_activity, activity, retinal_plan)
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
        _update_max_steps_per_frame!(runtime, steps_per_frame, step_ms, frame.frame_ms)
        callback(frame) === false && break
        _throttle!(runtime, frame_start)
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
    retinal_activity = Matrix{Float32}(undef, _retinal_output_size(config))
    retinal_plan = _coupled_retinal_plan(retinal_source, geometry, config)

    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        steps_per_frame = _steps_per_frame(runtime, p)
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
        _fill_coupled_retinal_activity!(
            retinal_activity,
            activity_left,
            activity_right,
            retinal_source,
            retinal_plan,
        )
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
        _update_max_steps_per_frame!(runtime, steps_per_frame, step_ms, frame.frame_ms)
        callback(frame) === false && break
        _throttle!(runtime, frame_start)
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

# WebSocket protocol and HTTP server

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
        ",\"Ge\":", _json_number(config.Ge),
        ",\"Gi\":", _json_number(config.Gi),
        ",\"Se\":", _json_number(config.Se),
        ",\"Si\":", _json_number(config.Si),
        ",\"Aee\":", _json_number(config.Aee),
        ",\"Aei\":", _json_number(config.Aei),
        ",\"Aie\":", _json_number(config.Aie),
        ",\"Aii\":", _json_number(config.Aii),
        ",\"dt\":", _json_number(config.dt),
        ",\"seed\":", _json_number(config.seed),
        ",\"kernelCutoff\":", _json_number(config.kernel_cutoff),
        ",\"boundaryX\":", _json_string(config.boundary_x),
        ",\"boundaryY\":", _json_string(config.boundary_y),
        ",\"partialReflectStrength\":", _json_number(config.partial_reflect_strength; digits=3),
        ",\"coupling\":", _json_string(config.coupling),
        ",\"couplingStrength\":", _json_number(config.coupling_strength; digits=5),
        ",\"overlapRows\":", config.overlap_rows,
        ",\"fieldGeometry\":", _json_string(config.field_geometry),
        ",\"fieldDensity\":", _json_number(config.field_density; digits=3),
        ",\"retinalResolution\":", config.retinal_resolution,
        ",\"retinalRendering\":", _json_string(config.retinal_rendering),
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
        ",\"stepInterval\":", frame.steps_per_frame,
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
    elseif uri.path == "/styles.css"
        _write_response(stream, 200, "text/css; charset=utf-8", APPLET_CSS)
    elseif uri.path == "/app.js"
        _write_response(stream, 200, "text/javascript; charset=utf-8", APPLET_JS)
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

"""Start the applet server and return the running `HTTP.Server` immediately."""
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

"""Start the applet server and block until it is stopped."""
function serve_applet(;
    host::AbstractString="127.0.0.1",
    port::Integer=8088,
    verbose::Bool=true,
)
    server = serve_applet_async(host=host, port=port, verbose=verbose)
    wait(server)
    return server
end
