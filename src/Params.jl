struct ModelParams{T<:AbstractFloat}
    dt::T
    Te::T
    Ti::T
    Aee::T
    Aei::T
    Aie::T
    Aii::T
    He::T
    Hi::T
    Ge::T
    Gi::T
    Ne::T
    Ni::T
    V::T
end

function ModelParams{T}(;
    dt=T(0.2),
    Te=T(10.0),
    Ti=T(20.0),
    Aee=T(10.0),
    Aei=T(12.0),
    Aie=T(8.5),
    Aii=T(3.0),
    He=T(2.0),
    Hi=T(3.5),
    Ge=T(1.0),
    Gi=T(0.0),
    Ne=T(0.05),
    Ni=T(0.05),
    V=T(0.8),
) where {T<:AbstractFloat}
    return ModelParams{T}(dt, Te, Ti, Aee, Aei, Aie, Aii, He, Hi, Ge, Gi, Ne, Ni, V)
end

ModelParams(; kwargs...) = ModelParams{Float64}(; kwargs...)
