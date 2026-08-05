using FFTW
using LinearAlgebra
using Random

const DEFAULT_FFT_FLAGS = FFTW.MEASURE

firing_rate(x) = inv(one(x) + exp(-x))

function step_function(x)
    return max(sign(x), zero(x))
end

function strobe_stimulus(t, A, period, p::ModelParams)
    T = promote_type(typeof(t), typeof(A), typeof(period), typeof(p.V))
    return T(A) * step_function(sin((T(2) * T(pi) * T(t)) / T(period)) - T(p.V))
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
)
    fft_convolution!(Uec, excitatory_convolver, Ue)
    fft_convolution!(Uic, inhibitory_convolver, Ui)

    noise_E = @view noise[1, :, :]
    noise_I = @view noise[2, :, :]
    stim = strobe_stimulus(t, A, period, p)

    @. Ue += (p.dt / p.Te) * (-Ue + firing_rate(p.Aee * Uec - p.Aie * Uic - p.He + p.Ge * stim + p.Ne * noise_E))
    @. Ui += (p.dt / p.Ti) * (-Ui + firing_rate(p.Aei * Uec - p.Aii * Uic - p.Hi + p.Gi * stim + p.Ni * noise_I))

    return Ue, Ui
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
) where {F<:AbstractFloat}
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
        _step!(Ue, Ui, Uec, Uic, excitatory_convolver, inhibitory_convolver, noise, dtype(A), dtype(T), t, pT)

        if plot
            push!(pointE, Ue[point_index, point_index])
            push!(pointI, Ui[point_index, point_index])
            push!(time, Float64(t))
            stim = strobe_stimulus(t, dtype(A), dtype(T), pT)
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

    return SimulationOutput(gif=gif_snapshots, images=image_snapshots)
end
