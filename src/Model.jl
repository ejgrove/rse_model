using FFTW
using LinearAlgebra
using Metal
using Random

const DEFAULT_FFT_FLAGS = FFTW.MEASURE
const BOUNDARY_PERIODIC = UInt32(0)
const BOUNDARY_EDGE = UInt32(1)
const BOUNDARY_ZERO = UInt32(2)
const BOUNDARY_PARTIAL_REFLECT = UInt32(3)

firing_rate(x) = inv(one(x) + exp(-x))

function step_function(x)
    return max(sign(x), zero(x))
end

function duty_cycle_percent_from_threshold(threshold)
    v = clamp(Float64(threshold), -1.0, 1.0)
    return 100.0 * (0.5 - asin(v) / pi)
end

function stimulus_threshold_from_duty_cycle_percent(duty_cycle_percent)
    duty = Float64(duty_cycle_percent)
    0 <= duty <= 100 || throw(ArgumentError("duty cycle percent must be between 0 and 100."))
    return sin(pi * (0.5 - duty / 100))
end

function _stimulus_threshold(p::ModelParams, duty_cycle_percent)
    duty_cycle_percent === nothing && return p.V
    return stimulus_threshold_from_duty_cycle_percent(duty_cycle_percent)
end

function strobe_stimulus(t, A, period, p::ModelParams, duty_cycle_percent=nothing)
    T = promote_type(typeof(t), typeof(A), typeof(period), typeof(p.V))
    threshold = T(_stimulus_threshold(p, duty_cycle_percent))
    return T(A) * step_function(sin((T(2) * T(pi) * T(t)) / T(period)) - threshold)
end

function _boundary_mode(boundary::Symbol)
    boundary in (:periodic, :edge, :zero, :partial_reflect) && return boundary
    boundary in (:partial_reflective, :partially_reflective, :reflect) && return :partial_reflect
    throw(ArgumentError("boundary must be :periodic, :edge, :zero, or :partial_reflect."))
end

function _boundary_code(boundary::Symbol)
    mode = _boundary_mode(boundary)
    mode == :periodic && return BOUNDARY_PERIODIC
    mode == :edge && return BOUNDARY_EDGE
    mode == :partial_reflect && return BOUNDARY_PARTIAL_REFLECT
    return BOUNDARY_ZERO
end

function _resolve_boundaries(boundary, boundary_x::Symbol, boundary_y::Symbol)
    if boundary !== nothing
        mode = _boundary_mode(boundary)
        return mode, mode
    end
    return _boundary_mode(boundary_x), _boundary_mode(boundary_y)
end

function _validate_boundaries(boundary_x::Symbol, boundary_y::Symbol, convolution::Symbol, backend::Symbol)
    _boundary_code(boundary_x)
    _boundary_code(boundary_y)
    if (boundary_x != :periodic || boundary_y != :periodic) && (backend != :metal || convolution != :separable)
        throw(ArgumentError("Only the Metal separable convolution path supports non-periodic boundaries."))
    end
    return boundary_x, boundary_y
end

function _default_convolution(backend::Symbol, boundary_x::Symbol=:periodic, boundary_y::Symbol=:periodic)
    backend = backend == :gpu ? :metal : backend
    backend in (:cpu, :metal) || throw(ArgumentError("backend must be :cpu or :metal."))
    return backend == :metal && (boundary_x != :periodic || boundary_y != :periodic) ? :separable : :fft
end

struct FFTConvolver{T,P,Q}
    forward_plan::P
    inverse_plan::Q
    kernel_fft::Matrix{Complex{T}}
    work::Matrix{Complex{T}}
end

function FFTConvolver(kernel::Matrix{T}, template::Matrix{T}; flags=DEFAULT_FFT_FLAGS) where {T<:AbstractFloat}
    plan_input = zeros(T, size(template))
    forward_plan = plan_rfft(plan_input, (1, 2); flags=flags)
    plan_work = forward_plan * plan_input
    inverse_plan = plan_irfft(similar(plan_work), size(template, 1), (1, 2); flags=flags)
    kernel_fft = forward_plan * kernel
    return FFTConvolver{T,typeof(forward_plan),typeof(inverse_plan)}(
        forward_plan,
        inverse_plan,
        kernel_fft,
        similar(plan_work),
    )
end

function fft_convolution!(out::AbstractMatrix, convolver::FFTConvolver, U::AbstractMatrix)
    mul!(convolver.work, convolver.forward_plan, U)
    convolver.work .*= convolver.kernel_fft
    mul!(out, convolver.inverse_plan, convolver.work)
    return out
end

Base.@kwdef struct Snapshot{T<:AbstractFloat}
    t::Float64
    cortical_activity::Matrix{T}
    time::Vector{Float64}
    pointE::Vector{T}
    pointI::Vector{T}
    StimE::Vector{T}
    StimI::Vector{T}
end

Base.@kwdef struct SimulationOutput{T<:AbstractFloat}
    gif::Vector{Snapshot{T}}
    images::Vector{Snapshot{T}}
    compute_seconds::Float64 = NaN
end

function _rng(seed)
    return seed === nothing ? Random.default_rng() : Xoshiro(seed)
end

function _params_as(::Type{T}, p::ModelParams) where {T<:AbstractFloat}
    return p isa ModelParams{T} ? p : ModelParams{T}(;
        dt=T(p.dt),
        Te=T(p.Te),
        Ti=T(p.Ti),
        Aee=T(p.Aee),
        Aei=T(p.Aei),
        Aie=T(p.Aie),
        Aii=T(p.Aii),
        He=T(p.He),
        Hi=T(p.Hi),
        Ge=T(p.Ge),
        Gi=T(p.Gi),
        Ne=T(p.Ne),
        Ni=T(p.Ni),
        V=T(p.V),
    )
end

function _snapshot(t, Ue, Ui, time, pointE, pointI, StimE, StimI)
    return Snapshot(
        t=Float64(t),
        cortical_activity=abs.(Ue .- Ui),
        time=copy(time),
        pointE=copy(pointE),
        pointI=copy(pointI),
        StimE=copy(StimE),
        StimI=copy(StimI),
    )
end

function _step!(
    Ue,
    Ui,
    Uec,
    Uic,
    excitatory_convolver,
    inhibitory_convolver,
    noise,
    A,
    period,
    t,
    p,
    duty_cycle_percent=nothing,
)
    fft_convolution!(Uec, excitatory_convolver, Ue)
    fft_convolution!(Uic, inhibitory_convolver, Ui)

    noise_E = @view noise[1, :, :]
    noise_I = @view noise[2, :, :]
    stim = strobe_stimulus(t, A, period, p, duty_cycle_percent)

    @. Ue += (p.dt / p.Te) * (-Ue + firing_rate(p.Aee * Uec - p.Aie * Uic - p.He + p.Ge * stim + p.Ne * noise_E))
    @. Ui += (p.dt / p.Ti) * (-Ui + firing_rate(p.Aei * Uec - p.Aii * Uic - p.Hi + p.Gi * stim + p.Ni * noise_I))

    return Ue, Ui
end

function _apply_midline_coupling!(Ue_left, Ui_left, Ue_right, Ui_right, strength, overlap_rows)
    strength <= 0 && return Ue_left, Ui_left, Ue_right, Ui_right
    rows, cols = size(Ue_left)
    band_rows = max(1, div(overlap_rows, 2))
    band_rows = min(band_rows, div(rows, 2))
    mix = eltype(Ue_left)(strength)

    @inbounds for col in 1:cols, row_offset in 0:(band_rows - 1)
        left_top = row_offset + 1
        right_top = band_rows - row_offset
        _mix_pair!(Ue_left, Ue_right, left_top, right_top, col, mix)
        _mix_pair!(Ui_left, Ui_right, left_top, right_top, col, mix)

        left_bottom = rows - band_rows + row_offset + 1
        right_bottom = rows - row_offset
        _mix_pair!(Ue_left, Ue_right, left_bottom, right_bottom, col, mix)
        _mix_pair!(Ui_left, Ui_right, left_bottom, right_bottom, col, mix)
    end

    return Ue_left, Ui_left, Ue_right, Ui_right
end

function _mix_pair!(left, right, left_row, right_row, col, mix)
    left_value = left[left_row, col]
    right_value = right[right_row, col]
    left[left_row, col] = left_value + mix * (right_value - left_value)
    right[right_row, col] = right_value + mix * (left_value - right_value)
    return
end

struct MetalFFTConvolver{P,Q,K,W}
    forward_plan::P
    inverse_plan::Q
    kernel_fft::K
    work::W
end

struct MetalSeparableConvolver{K,W}
    kernel::K
    scratch::W
    radius::Int
end

function MetalSeparableConvolver(
    sigma,
    template;
    cutoff::Real=2.0,
)
    cutoff > 0 || throw(ArgumentError("kernel cutoff must be positive."))
    N = size(template, 1)
    radius = min(div(N, 2), max(1, ceil(Int, cutoff * Float64(sigma))))
    kernel = generate_gaussian_kernel_1d(sigma, radius; dtype=Float32)
    full_kernel = generate_gaussian_kernel_1d(sigma, div(N, 2); dtype=Float32)
    kernel .*= sum(full_kernel) / sum(kernel)
    return MetalSeparableConvolver(Metal.MtlArray(kernel), similar(template), radius)
end

function MetalFFTConvolver(kernel::Matrix{T}, template) where {T<:AbstractFloat}
    plan_input = Metal.zeros(T, size(template)...)
    forward_plan = plan_rfft(plan_input, (1, 2))
    plan_work = forward_plan * plan_input
    inverse_plan = plan_irfft(similar(plan_work), size(template, 1), (1, 2))
    kernel_fft = forward_plan * Metal.MtlArray(kernel)
    return MetalFFTConvolver(
        forward_plan,
        inverse_plan,
        kernel_fft,
        similar(plan_work),
    )
end

function fft_convolution!(out, convolver::MetalFFTConvolver, U)
    mul!(convolver.work, convolver.forward_plan, U)
    convolver.work .*= convolver.kernel_fft
    mul!(out, convolver.inverse_plan, convolver.work)
    return out
end

function _metal_conv_cols_kernel!(
    out,
    input,
    kernel,
    radius,
    rows,
    cols,
    klen,
    n,
    boundary_code,
    fft_origin_shift,
    partial_reflect_strength,
)
    i = thread_position_in_grid().x
    if i <= n
        row0 = (i - 1) % rows
        col0 = (i - 1) ÷ rows
        acc = 0.0f0
        for k in 1:klen
            offset = Int32(k) - Int32(radius) - Int32(1)
            source_col = Int32(col0) + offset
            if boundary_code == BOUNDARY_PERIODIC
                source_col0 = mod(source_col - Int32(fft_origin_shift), Int32(cols))
                source_idx = row0 + UInt32(source_col0) * rows + UInt32(1)
                acc += input[source_idx] * kernel[k]
            elseif boundary_code == BOUNDARY_EDGE
                source_col0 = min(max(source_col, Int32(0)), Int32(cols) - Int32(1))
                source_idx = row0 + UInt32(source_col0) * rows + UInt32(1)
                acc += input[source_idx] * kernel[k]
            elseif boundary_code == BOUNDARY_PARTIAL_REFLECT
                source_col0 = source_col
                reflected = false
                if source_col < Int32(0)
                    source_col0 = -source_col - Int32(1)
                    reflected = true
                elseif source_col >= Int32(cols)
                    source_col0 = Int32(2) * Int32(cols) - source_col - Int32(1)
                    reflected = true
                end
                source_col0 = min(max(source_col0, Int32(0)), Int32(cols) - Int32(1))
                source_idx = row0 + UInt32(source_col0) * rows + UInt32(1)
                acc += input[source_idx] * kernel[k] * (reflected ? partial_reflect_strength : 1.0f0)
            elseif source_col >= Int32(0) && source_col < Int32(cols)
                source_idx = row0 + UInt32(source_col) * rows + UInt32(1)
                acc += input[source_idx] * kernel[k]
            end
        end
        out[i] = acc
    end
    return
end

function _metal_conv_rows_kernel!(
    out,
    input,
    kernel,
    radius,
    rows,
    cols,
    klen,
    n,
    boundary_code,
    fft_origin_shift,
    partial_reflect_strength,
)
    i = thread_position_in_grid().x
    if i <= n
        row0 = (i - 1) % rows
        col0 = (i - 1) ÷ rows
        acc = 0.0f0
        for k in 1:klen
            offset = Int32(k) - Int32(radius) - Int32(1)
            source_row = Int32(row0) + offset
            if boundary_code == BOUNDARY_PERIODIC
                source_row0 = mod(source_row - Int32(fft_origin_shift), Int32(rows))
                source_idx = UInt32(source_row0) + col0 * rows + UInt32(1)
                acc += input[source_idx] * kernel[k]
            elseif boundary_code == BOUNDARY_EDGE
                source_row0 = min(max(source_row, Int32(0)), Int32(rows) - Int32(1))
                source_idx = UInt32(source_row0) + col0 * rows + UInt32(1)
                acc += input[source_idx] * kernel[k]
            elseif boundary_code == BOUNDARY_PARTIAL_REFLECT
                source_row0 = source_row
                reflected = false
                if source_row < Int32(0)
                    source_row0 = -source_row - Int32(1)
                    reflected = true
                elseif source_row >= Int32(rows)
                    source_row0 = Int32(2) * Int32(rows) - source_row - Int32(1)
                    reflected = true
                end
                source_row0 = min(max(source_row0, Int32(0)), Int32(rows) - Int32(1))
                source_idx = UInt32(source_row0) + col0 * rows + UInt32(1)
                acc += input[source_idx] * kernel[k] * (reflected ? partial_reflect_strength : 1.0f0)
            elseif source_row >= Int32(0) && source_row < Int32(rows)
                source_idx = UInt32(source_row) + col0 * rows + UInt32(1)
                acc += input[source_idx] * kernel[k]
            end
        end
        out[i] = acc
    end
    return
end

function _metal_conv_cols_pair_kernel!(
    out_e,
    out_i,
    input_e,
    input_i,
    kernel_e,
    kernel_i,
    radius_e,
    radius_i,
    rows,
    cols,
    klen_e,
    klen_i,
    n,
    boundary_code,
    fft_origin_shift,
    partial_reflect_strength,
)
    i = thread_position_in_grid().x
    if i <= n
        row0 = (i - 1) % rows
        col0 = (i - 1) ÷ rows

        acc_e = 0.0f0
        for k in 1:klen_e
            offset = Int32(k) - Int32(radius_e) - Int32(1)
            source_col = Int32(col0) + offset
            if boundary_code == BOUNDARY_PERIODIC
                source_col0 = mod(source_col - Int32(fft_origin_shift), Int32(cols))
                source_idx = row0 + UInt32(source_col0) * rows + UInt32(1)
                acc_e += input_e[source_idx] * kernel_e[k]
            elseif boundary_code == BOUNDARY_EDGE
                source_col0 = min(max(source_col, Int32(0)), Int32(cols) - Int32(1))
                source_idx = row0 + UInt32(source_col0) * rows + UInt32(1)
                acc_e += input_e[source_idx] * kernel_e[k]
            elseif boundary_code == BOUNDARY_PARTIAL_REFLECT
                source_col0 = source_col
                reflected = false
                if source_col < Int32(0)
                    source_col0 = -source_col - Int32(1)
                    reflected = true
                elseif source_col >= Int32(cols)
                    source_col0 = Int32(2) * Int32(cols) - source_col - Int32(1)
                    reflected = true
                end
                source_col0 = min(max(source_col0, Int32(0)), Int32(cols) - Int32(1))
                source_idx = row0 + UInt32(source_col0) * rows + UInt32(1)
                acc_e += input_e[source_idx] * kernel_e[k] * (reflected ? partial_reflect_strength : 1.0f0)
            elseif source_col >= Int32(0) && source_col < Int32(cols)
                source_idx = row0 + UInt32(source_col) * rows + UInt32(1)
                acc_e += input_e[source_idx] * kernel_e[k]
            end
        end
        out_e[i] = acc_e

        acc_i = 0.0f0
        for k in 1:klen_i
            offset = Int32(k) - Int32(radius_i) - Int32(1)
            source_col = Int32(col0) + offset
            if boundary_code == BOUNDARY_PERIODIC
                source_col0 = mod(source_col - Int32(fft_origin_shift), Int32(cols))
                source_idx = row0 + UInt32(source_col0) * rows + UInt32(1)
                acc_i += input_i[source_idx] * kernel_i[k]
            elseif boundary_code == BOUNDARY_EDGE
                source_col0 = min(max(source_col, Int32(0)), Int32(cols) - Int32(1))
                source_idx = row0 + UInt32(source_col0) * rows + UInt32(1)
                acc_i += input_i[source_idx] * kernel_i[k]
            elseif boundary_code == BOUNDARY_PARTIAL_REFLECT
                source_col0 = source_col
                reflected = false
                if source_col < Int32(0)
                    source_col0 = -source_col - Int32(1)
                    reflected = true
                elseif source_col >= Int32(cols)
                    source_col0 = Int32(2) * Int32(cols) - source_col - Int32(1)
                    reflected = true
                end
                source_col0 = min(max(source_col0, Int32(0)), Int32(cols) - Int32(1))
                source_idx = row0 + UInt32(source_col0) * rows + UInt32(1)
                acc_i += input_i[source_idx] * kernel_i[k] * (reflected ? partial_reflect_strength : 1.0f0)
            elseif source_col >= Int32(0) && source_col < Int32(cols)
                source_idx = row0 + UInt32(source_col) * rows + UInt32(1)
                acc_i += input_i[source_idx] * kernel_i[k]
            end
        end
        out_i[i] = acc_i
    end
    return
end

function _metal_conv_rows_pair_kernel!(
    out_e,
    out_i,
    input_e,
    input_i,
    kernel_e,
    kernel_i,
    radius_e,
    radius_i,
    rows,
    cols,
    klen_e,
    klen_i,
    n,
    boundary_code,
    fft_origin_shift,
    partial_reflect_strength,
)
    i = thread_position_in_grid().x
    if i <= n
        row0 = (i - 1) % rows
        col0 = (i - 1) ÷ rows

        acc_e = 0.0f0
        for k in 1:klen_e
            offset = Int32(k) - Int32(radius_e) - Int32(1)
            source_row = Int32(row0) + offset
            if boundary_code == BOUNDARY_PERIODIC
                source_row0 = mod(source_row - Int32(fft_origin_shift), Int32(rows))
                source_idx = UInt32(source_row0) + col0 * rows + UInt32(1)
                acc_e += input_e[source_idx] * kernel_e[k]
            elseif boundary_code == BOUNDARY_EDGE
                source_row0 = min(max(source_row, Int32(0)), Int32(rows) - Int32(1))
                source_idx = UInt32(source_row0) + col0 * rows + UInt32(1)
                acc_e += input_e[source_idx] * kernel_e[k]
            elseif boundary_code == BOUNDARY_PARTIAL_REFLECT
                source_row0 = source_row
                reflected = false
                if source_row < Int32(0)
                    source_row0 = -source_row - Int32(1)
                    reflected = true
                elseif source_row >= Int32(rows)
                    source_row0 = Int32(2) * Int32(rows) - source_row - Int32(1)
                    reflected = true
                end
                source_row0 = min(max(source_row0, Int32(0)), Int32(rows) - Int32(1))
                source_idx = UInt32(source_row0) + col0 * rows + UInt32(1)
                acc_e += input_e[source_idx] * kernel_e[k] * (reflected ? partial_reflect_strength : 1.0f0)
            elseif source_row >= Int32(0) && source_row < Int32(rows)
                source_idx = UInt32(source_row) + col0 * rows + UInt32(1)
                acc_e += input_e[source_idx] * kernel_e[k]
            end
        end
        out_e[i] = acc_e

        acc_i = 0.0f0
        for k in 1:klen_i
            offset = Int32(k) - Int32(radius_i) - Int32(1)
            source_row = Int32(row0) + offset
            if boundary_code == BOUNDARY_PERIODIC
                source_row0 = mod(source_row - Int32(fft_origin_shift), Int32(rows))
                source_idx = UInt32(source_row0) + col0 * rows + UInt32(1)
                acc_i += input_i[source_idx] * kernel_i[k]
            elseif boundary_code == BOUNDARY_EDGE
                source_row0 = min(max(source_row, Int32(0)), Int32(rows) - Int32(1))
                source_idx = UInt32(source_row0) + col0 * rows + UInt32(1)
                acc_i += input_i[source_idx] * kernel_i[k]
            elseif boundary_code == BOUNDARY_PARTIAL_REFLECT
                source_row0 = source_row
                reflected = false
                if source_row < Int32(0)
                    source_row0 = -source_row - Int32(1)
                    reflected = true
                elseif source_row >= Int32(rows)
                    source_row0 = Int32(2) * Int32(rows) - source_row - Int32(1)
                    reflected = true
                end
                source_row0 = min(max(source_row0, Int32(0)), Int32(rows) - Int32(1))
                source_idx = UInt32(source_row0) + col0 * rows + UInt32(1)
                acc_i += input_i[source_idx] * kernel_i[k] * (reflected ? partial_reflect_strength : 1.0f0)
            elseif source_row >= Int32(0) && source_row < Int32(rows)
                source_idx = UInt32(source_row) + col0 * rows + UInt32(1)
                acc_i += input_i[source_idx] * kernel_i[k]
            end
        end
        out_i[i] = acc_i
    end
    return
end

function separable_convolution!(
    out,
    convolver::MetalSeparableConvolver,
    U;
    gpu_threads::Integer=256,
    boundary=nothing,
    boundary_x::Symbol=:periodic,
    boundary_y::Symbol=:periodic,
    partial_reflect_strength::Real=0.5,
)
    rows_u, cols_u = size(U)
    rows = UInt32(rows_u)
    cols = UInt32(cols_u)
    n = UInt32(length(U))
    klen = UInt32(length(convolver.kernel))
    radius = UInt32(convolver.radius)
    boundary_x, boundary_y = _resolve_boundaries(boundary, boundary_x, boundary_y)
    boundary_x_code = _boundary_code(boundary_x)
    boundary_y_code = _boundary_code(boundary_y)
    0 <= partial_reflect_strength <= 1 || throw(ArgumentError("partial_reflect_strength must be between 0 and 1."))
    reflect_strength = Float32(partial_reflect_strength)
    # The reference FFT path uses a centered Gaussian matrix, so periodic
    # separable convolution needs the same circular origin shift.
    fft_origin_shift_x = UInt32(boundary_x == :periodic ? div(cols_u, 2) : 0)
    fft_origin_shift_y = UInt32(boundary_y == :periodic ? div(rows_u, 2) : 0)
    threads = min(gpu_threads, length(U))
    groups = cld(length(U), threads)

    @metal threads=threads groups=groups _metal_conv_cols_kernel!(
        convolver.scratch,
        U,
        convolver.kernel,
        radius,
        rows,
        cols,
        klen,
        n,
        boundary_x_code,
        fft_origin_shift_x,
        reflect_strength,
    )
    @metal threads=threads groups=groups _metal_conv_rows_kernel!(
        out,
        convolver.scratch,
        convolver.kernel,
        radius,
        rows,
        cols,
        klen,
        n,
        boundary_y_code,
        fft_origin_shift_y,
        reflect_strength,
    )

    return out
end

function separable_convolution_pair!(
    out_e,
    out_i,
    convolver_e::MetalSeparableConvolver,
    convolver_i::MetalSeparableConvolver,
    Ue,
    Ui;
    gpu_threads::Integer=256,
    boundary=nothing,
    boundary_x::Symbol=:periodic,
    boundary_y::Symbol=:periodic,
    partial_reflect_strength::Real=0.5,
)
    rows_u, cols_u = size(Ue)
    rows = UInt32(rows_u)
    cols = UInt32(cols_u)
    n = UInt32(length(Ue))
    klen_e = UInt32(length(convolver_e.kernel))
    klen_i = UInt32(length(convolver_i.kernel))
    radius_e = UInt32(convolver_e.radius)
    radius_i = UInt32(convolver_i.radius)
    boundary_x, boundary_y = _resolve_boundaries(boundary, boundary_x, boundary_y)
    boundary_x_code = _boundary_code(boundary_x)
    boundary_y_code = _boundary_code(boundary_y)
    0 <= partial_reflect_strength <= 1 || throw(ArgumentError("partial_reflect_strength must be between 0 and 1."))
    reflect_strength = Float32(partial_reflect_strength)
    # Match the legacy centered-kernel FFT origin for periodic separable runs.
    fft_origin_shift_x = UInt32(boundary_x == :periodic ? div(cols_u, 2) : 0)
    fft_origin_shift_y = UInt32(boundary_y == :periodic ? div(rows_u, 2) : 0)
    threads = min(gpu_threads, length(Ue))
    groups = cld(length(Ue), threads)

    @metal threads=threads groups=groups _metal_conv_cols_pair_kernel!(
        convolver_e.scratch,
        convolver_i.scratch,
        Ue,
        Ui,
        convolver_e.kernel,
        convolver_i.kernel,
        radius_e,
        radius_i,
        rows,
        cols,
        klen_e,
        klen_i,
        n,
        boundary_x_code,
        fft_origin_shift_x,
        reflect_strength,
    )
    @metal threads=threads groups=groups _metal_conv_rows_pair_kernel!(
        out_e,
        out_i,
        convolver_e.scratch,
        convolver_i.scratch,
        convolver_e.kernel,
        convolver_i.kernel,
        radius_e,
        radius_i,
        rows,
        cols,
        klen_e,
        klen_i,
        n,
        boundary_y_code,
        fft_origin_shift_y,
        reflect_strength,
    )

    return out_e, out_i
end

function _metal_apply_mask_kernel!(U, mask, n)
    i = thread_position_in_grid().x
    if i <= n
        U[i] *= mask[i]
    end
    return
end

function apply_field_mask!(U, mask_gpu, gpu_threads::Integer)
    n = length(U)
    threads = min(gpu_threads, n)
    groups = cld(n, threads)
    @metal threads=threads groups=groups _metal_apply_mask_kernel!(U, mask_gpu, UInt32(n))
    return U
end

function apply_field_mask!(Ue, Ui, mask_gpu, gpu_threads::Integer)
    apply_field_mask!(Ue, mask_gpu, gpu_threads)
    apply_field_mask!(Ui, mask_gpu, gpu_threads)
    return Ue, Ui
end

function _metal_euler_kernel!(
    Ue,
    Ui,
    Uec,
    Uic,
    noise_E,
    noise_I,
    stim,
    dt_over_Te,
    dt_over_Ti,
    Aee,
    Aei,
    Aie,
    Aii,
    He,
    Hi,
    Ge,
    Gi,
    Ne,
    Ni,
    n,
)
    i = thread_position_in_grid().x
    if i <= n
        uec = Uec[i]
        uic = Uic[i]
        input_e = Aee * uec - Aie * uic - He + Ge * stim + Ne * noise_E[i]
        input_i = Aei * uec - Aii * uic - Hi + Gi * stim + Ni * noise_I[i]
        Ue[i] += dt_over_Te * (-Ue[i] + inv(1.0f0 + exp(-input_e)))
        Ui[i] += dt_over_Ti * (-Ui[i] + inv(1.0f0 + exp(-input_i)))
    end
    return
end

function _metal_midline_coupling_kernel!(
    Ue_left,
    Ui_left,
    Ue_right,
    Ui_right,
    strength,
    rows,
    cols,
    band_rows,
    n,
)
    i = thread_position_in_grid().x
    if i <= n
        band_area = band_rows * cols
        band = (i - UInt32(1)) ÷ band_area
        local_i = (i - UInt32(1)) % band_area
        row_offset = local_i % band_rows
        col0 = local_i ÷ band_rows

        left_row0 = row_offset
        right_row0 = band_rows - row_offset - UInt32(1)
        if band == UInt32(1)
            left_row0 = rows - band_rows + row_offset
            right_row0 = rows - row_offset - UInt32(1)
        end

        left_idx = left_row0 + col0 * rows + UInt32(1)
        right_idx = right_row0 + col0 * rows + UInt32(1)

        left_e = Ue_left[left_idx]
        right_e = Ue_right[right_idx]
        left_i = Ui_left[left_idx]
        right_i = Ui_right[right_idx]

        Ue_left[left_idx] = left_e + strength * (right_e - left_e)
        Ue_right[right_idx] = right_e + strength * (left_e - right_e)
        Ui_left[left_idx] = left_i + strength * (right_i - left_i)
        Ui_right[right_idx] = right_i + strength * (left_i - right_i)
    end
    return
end

function apply_midline_coupling!(
    Ue_left,
    Ui_left,
    Ue_right,
    Ui_right;
    strength::Real,
    overlap_rows::Integer,
    gpu_threads::Integer=256,
)
    strength <= 0 && return Ue_left, Ui_left, Ue_right, Ui_right
    rows_u, cols_u = size(Ue_left)
    band_rows_u = min(max(1, div(overlap_rows, 2)), div(rows_u, 2))
    n_pairs = 2 * band_rows_u * cols_u
    threads = min(gpu_threads, n_pairs)
    groups = cld(n_pairs, threads)

    @metal threads=threads groups=groups _metal_midline_coupling_kernel!(
        Ue_left,
        Ui_left,
        Ue_right,
        Ui_right,
        Float32(strength),
        UInt32(rows_u),
        UInt32(cols_u),
        UInt32(band_rows_u),
        UInt32(n_pairs),
    )

    return Ue_left, Ui_left, Ue_right, Ui_right
end

function _metal_step!(
    Ue,
    Ui,
    Uec,
    Uic,
    excitatory_convolver,
    inhibitory_convolver,
    noise_E,
    noise_I,
    A::Float32,
    period::Float32,
    t::Float32,
    p::ModelParams{Float32},
    gpu_threads::Integer,
    duty_cycle_percent=nothing,
)
    fft_convolution!(Uec, excitatory_convolver, Ue)
    fft_convolution!(Uic, inhibitory_convolver, Ui)

    stim = strobe_stimulus(t, A, period, p, duty_cycle_percent)
    n = UInt32(length(Ue))
    threads = min(gpu_threads, length(Ue))
    groups = cld(length(Ue), threads)
    @metal threads=threads groups=groups _metal_euler_kernel!(
        Ue,
        Ui,
        Uec,
        Uic,
        noise_E,
        noise_I,
        stim,
        p.dt / p.Te,
        p.dt / p.Ti,
        p.Aee,
        p.Aei,
        p.Aie,
        p.Aii,
        p.He,
        p.Hi,
        p.Ge,
        p.Gi,
        p.Ne,
        p.Ni,
        n,
    )

    return Ue, Ui
end

function _metal_step_separable!(
    Ue,
    Ui,
    Uec,
    Uic,
    excitatory_convolver,
    inhibitory_convolver,
    noise_E,
    noise_I,
    A::Float32,
    period::Float32,
    t::Float32,
    p::ModelParams{Float32},
    gpu_threads::Integer,
    boundary_x::Symbol=:periodic,
    boundary_y::Symbol=:periodic,
    partial_reflect_strength::Real=0.5,
    duty_cycle_percent=nothing,
)
    separable_convolution_pair!(
        Uec,
        Uic,
        excitatory_convolver,
        inhibitory_convolver,
        Ue,
        Ui;
        gpu_threads=gpu_threads,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
        partial_reflect_strength=partial_reflect_strength,
    )

    stim = strobe_stimulus(t, A, period, p, duty_cycle_percent)
    n = UInt32(length(Ue))
    threads = min(gpu_threads, length(Ue))
    groups = cld(length(Ue), threads)
    @metal threads=threads groups=groups _metal_euler_kernel!(
        Ue,
        Ui,
        Uec,
        Uic,
        noise_E,
        noise_I,
        stim,
        p.dt / p.Te,
        p.dt / p.Ti,
        p.Aee,
        p.Aei,
        p.Aie,
        p.Aii,
        p.He,
        p.Hi,
        p.Ge,
        p.Gi,
        p.Ne,
        p.Ni,
        n,
    )

    return Ue, Ui
end

function _snapshot_gpu(t, Ue, Ui, cortical_gpu, time, pointE, pointI, StimE, StimI)
    cortical_gpu .= abs.(Ue .- Ui)
    Metal.synchronize()
    return Snapshot(
        t=Float64(t),
        cortical_activity=Array(cortical_gpu),
        time=copy(time),
        pointE=copy(pointE),
        pointI=copy(pointI),
        StimE=copy(StimE),
        StimI=copy(StimI),
    )
end

function run_simulation_gpu(;
    N::Integer,
    A,
    T,
    Se,
    Si,
    start_time::Integer,
    end_time::Integer,
    seed=nothing,
    plot::Bool,
    gif::Bool,
    interval::Integer,
    p::ModelParams,
    fps::Integer=50,
    dtype::Type{F}=Float32,
    gpu_threads::Integer=256,
    convolution::Symbol=:separable,
    kernel_cutoff::Real=3.0,
    boundary=nothing,
    boundary_x::Symbol=:periodic,
    boundary_y::Symbol=:periodic,
    partial_reflect_strength::Real=0.5,
    duty_cycle_percent=nothing,
) where {F<:AbstractFloat}
    Metal.functional() || throw(ErrorException("Metal.jl is not functional on this machine."))
    dtype === Float32 || throw(ArgumentError("The Metal backend currently supports Float32 only."))
    gpu_threads > 0 || throw(ArgumentError("gpu_threads must be positive."))
    0 <= partial_reflect_strength <= 1 ||
        throw(ArgumentError("partial_reflect_strength must be between 0 and 1."))
    boundary_x, boundary_y = _resolve_boundaries(boundary, boundary_x, boundary_y)
    _validate_boundaries(boundary_x, boundary_y, convolution, :metal)

    timer_start = time_ns()
    pT = _params_as(Float32, p)
    if seed !== nothing
        Metal.seed!(seed)
    end

    Ue = Metal.rand(Float32, N, N)
    Ui = Metal.rand(Float32, N, N)

    Ke = generate_gaussian_kernel(Se, N; dtype=Float32)
    Ki = generate_gaussian_kernel(Si, N; dtype=Float32)
    excitatory_convolver = if convolution == :fft
        MetalFFTConvolver(Ke, Ue)
    elseif convolution == :separable
        MetalSeparableConvolver(Se, Ue; cutoff=kernel_cutoff)
    else
        throw(ArgumentError("Metal convolution must be :fft or :separable"))
    end
    inhibitory_convolver = if convolution == :fft
        MetalFFTConvolver(Ki, Ui)
    else
        MetalSeparableConvolver(Si, Ui; cutoff=kernel_cutoff)
    end

    Uec = similar(Ue)
    Uic = similar(Ui)
    noise_E = similar(Ue)
    noise_I = similar(Ui)
    cortical_gpu = similar(Ue)

    pointE = Float32[]
    pointI = Float32[]
    time = Float64[]
    StimE = Float32[]
    StimI = Float32[]
    image_snapshots = Snapshot{Float32}[]
    gif_snapshots = Snapshot{Float32}[]

    plot_every_steps = max(1, round(Int, interval / pT.dt))
    gif_every_steps = max(1, round(Int, (1000 / fps) / pT.dt))
    steps = round(Int, end_time / pT.dt)
    point_index = min(3, N)

    for step_idx in 0:steps
        t = Float32(step_idx) * pT.dt
        Metal.randn!(noise_E)
        Metal.randn!(noise_I)
        if convolution == :fft
            _metal_step!(
                Ue,
                Ui,
                Uec,
                Uic,
                excitatory_convolver,
                inhibitory_convolver,
                noise_E,
                noise_I,
                Float32(A),
                Float32(T),
                t,
                pT,
                gpu_threads,
                duty_cycle_percent,
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
                Float32(A),
                Float32(T),
                t,
                pT,
                gpu_threads,
                boundary_x,
                boundary_y,
                partial_reflect_strength,
                duty_cycle_percent,
            )
        end

        if plot
            Metal.synchronize()
            push!(pointE, Ue[point_index, point_index])
            push!(pointI, Ui[point_index, point_index])
            push!(time, Float64(t))
            stim = strobe_stimulus(t, Float32(A), Float32(T), pT, duty_cycle_percent)
            push!(StimE, pT.Ge * stim)
            push!(StimI, pT.Gi * stim)
        end

        if step_idx != 0 && step_idx % plot_every_steps == 0 && start_time <= t <= end_time
            push!(image_snapshots, _snapshot_gpu(t, Ue, Ui, cortical_gpu, time, pointE, pointI, StimE, StimI))
        end

        if gif && step_idx % gif_every_steps == 0 && start_time <= floor(Int, t) <= end_time
            push!(gif_snapshots, _snapshot_gpu(t, Ue, Ui, cortical_gpu, time, pointE, pointI, StimE, StimI))
        end
    end

    Metal.synchronize()
    compute_seconds = (time_ns() - timer_start) / 1e9
    return SimulationOutput(gif=gif_snapshots, images=image_snapshots, compute_seconds=compute_seconds)
end

function run_simulation(;
    N::Integer,
    A,
    T,
    Se,
    Si,
    start_time::Integer,
    end_time::Integer,
    seed=nothing,
    plot::Bool,
    gif::Bool,
    interval::Integer,
    p::ModelParams,
    fps::Integer=50,
    dtype::Type{F}=Float32,
    fft_flags=DEFAULT_FFT_FLAGS,
    backend::Symbol=:cpu,
    gpu_threads::Integer=256,
    convolution::Symbol=:fft,
    kernel_cutoff::Real=3.0,
    boundary=nothing,
    boundary_x::Symbol=:periodic,
    boundary_y::Symbol=:periodic,
    partial_reflect_strength::Real=0.5,
    duty_cycle_percent=nothing,
) where {F<:AbstractFloat}
    boundary_x, boundary_y = _resolve_boundaries(boundary, boundary_x, boundary_y)
    if backend in (:metal, :gpu)
        return run_simulation_gpu(
            N=N,
            A=A,
            T=T,
            Se=Se,
            Si=Si,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            plot=plot,
            gif=gif,
            interval=interval,
            p=p,
            fps=fps,
            dtype=dtype,
            gpu_threads=gpu_threads,
            convolution=convolution,
            kernel_cutoff=kernel_cutoff,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
            partial_reflect_strength=partial_reflect_strength,
            duty_cycle_percent=duty_cycle_percent,
        )
    elseif backend != :cpu
        throw(ArgumentError("backend must be :cpu or :metal"))
    end
    convolution == :fft || throw(ArgumentError("The CPU backend currently supports :fft convolution only."))
    _validate_boundaries(boundary_x, boundary_y, convolution, :cpu)

    timer_start = time_ns()
    pT = _params_as(dtype, p)
    rng = _rng(seed)

    Ue = rand(rng, dtype, N, N)
    Ui = rand(rng, dtype, N, N)

    Ke = generate_gaussian_kernel(Se, N; dtype=dtype)
    Ki = generate_gaussian_kernel(Si, N; dtype=dtype)
    excitatory_convolver = FFTConvolver(Ke, Ue; flags=fft_flags)
    inhibitory_convolver = FFTConvolver(Ki, Ui; flags=fft_flags)
    Uec = similar(Ue)
    Uic = similar(Ui)
    noise = Array{F}(undef, 2, N, N)

    pointE = F[]
    pointI = F[]
    time = Float64[]
    StimE = F[]
    StimI = F[]
    image_snapshots = Snapshot{F}[]
    gif_snapshots = Snapshot{F}[]

    plot_every_steps = max(1, round(Int, interval / pT.dt))
    gif_every_steps = max(1, round(Int, (1000 / fps) / pT.dt))
    steps = round(Int, end_time / pT.dt)
    point_index = min(3, N)

    for step_idx in 0:steps
        t = dtype(step_idx) * pT.dt
        randn!(rng, noise)
        _step!(
            Ue,
            Ui,
            Uec,
            Uic,
            excitatory_convolver,
            inhibitory_convolver,
            noise,
            dtype(A),
            dtype(T),
            t,
            pT,
            duty_cycle_percent,
        )

        if plot
            push!(pointE, Ue[point_index, point_index])
            push!(pointI, Ui[point_index, point_index])
            push!(time, Float64(t))
            stim = strobe_stimulus(t, dtype(A), dtype(T), pT, duty_cycle_percent)
            push!(StimE, pT.Ge * stim)
            push!(StimI, pT.Gi * stim)
        end

        if step_idx != 0 && step_idx % plot_every_steps == 0 && start_time <= t <= end_time
            push!(image_snapshots, _snapshot(t, Ue, Ui, time, pointE, pointI, StimE, StimI))
        end

        if gif && step_idx % gif_every_steps == 0 && start_time <= floor(Int, t) <= end_time
            push!(gif_snapshots, _snapshot(t, Ue, Ui, time, pointE, pointI, StimE, StimI))
        end
    end

    compute_seconds = (time_ns() - timer_start) / 1e9
    return SimulationOutput(gif=gif_snapshots, images=image_snapshots, compute_seconds=compute_seconds)
end
