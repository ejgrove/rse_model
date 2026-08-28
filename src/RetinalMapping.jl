"""Wrap a zero-based sampling index onto a one-based Julia array axis."""
_wrap_index(i::Integer, n::Integer) = mod(i, n) + 1

"""Cached wrapped bilinear coordinates for one cortical-to-retinal projection."""
struct RetinalMapPlan{T<:AbstractFloat}
    source_size::Tuple{Int,Int}
    output_size::Tuple{Int,Int}
    index00::Vector{Int32}
    index01::Vector{Int32}
    index10::Vector{Int32}
    index11::Vector{Int32}
    weight00::Vector{T}
    weight01::Vector{T}
    weight10::Vector{T}
    weight11::Vector{T}
end

function _gridwrap_bilinear_stencil(height::Int, width::Int, y::T, x::T) where {T<:AbstractFloat}
    x0 = floor(Int, x)
    y0 = floor(Int, y)
    dx = x - T(x0)
    dy = y - T(y0)
    x1 = x0 + 1
    y1 = y0 + 1

    linear = LinearIndices((height, width))
    index00 = Int32(linear[_wrap_index(y0, height), _wrap_index(x0, width)])
    index01 = Int32(linear[_wrap_index(y0, height), _wrap_index(x1, width)])
    index10 = Int32(linear[_wrap_index(y1, height), _wrap_index(x0, width)])
    index11 = Int32(linear[_wrap_index(y1, height), _wrap_index(x1, width)])
    weight00 = (one(T) - dy) * (one(T) - dx)
    weight01 = (one(T) - dy) * dx
    weight10 = dy * (one(T) - dx)
    weight11 = dy * dx

    return index00, index01, index10, index11, weight00, weight01, weight10, weight11
end


"""
    retinal_map_plan(source_size; output_size=source_size, angle_origin=0, dtype=Float32)

Precompute the grid-wrapped log-polar sampling coordinates used by
`retinal_transform!`. Building the plan once avoids logarithms, angles, and
index wrapping in every streamed frame.
"""
function retinal_map_plan(
    source_size::Tuple{Int,Int};
    output_size::Tuple{Int,Int}=source_size,
    angle_origin=0,
    dtype::Type{T}=Float32,
) where {T<:AbstractFloat}
    source_height, source_width = source_size
    height, width = output_size
    source_height > 0 && source_width > 0 || throw(ArgumentError("retinal-map source dimensions must be positive."))
    height > 0 && width > 0 || throw(ArgumentError("retinal-map output dimensions must be positive."))
    count = height * width
    index00 = Vector{Int32}(undef, count)
    index01 = similar(index00)
    index10 = similar(index00)
    index11 = similar(index00)
    weight00 = Vector{T}(undef, count)
    weight01 = similar(weight00)
    weight10 = similar(weight00)
    weight11 = similar(weight00)
    angle_origin_t = T(angle_origin)

    @inbounds for col in 1:width, row in 1:height
        x = T(-1) + T(2) * T(col - 1) / T(max(width - 1, 1))
        y = T(1) - T(2) * T(row - 1) / T(max(height - 1, 1))
        radius = hypot(x, y)
        angle = mod(atan(y, x) + T(2pi), T(2pi))
        cortical_x = log(radius + T(1e-26)) / T(2pi) * T(source_width)
        cortical_y = mod(angle - angle_origin_t, T(2pi)) / T(2pi) * T(source_height)
        pixel = row + (col - 1) * height
        stencil = _gridwrap_bilinear_stencil(source_height, source_width, cortical_y, cortical_x)
        index00[pixel], index01[pixel], index10[pixel], index11[pixel] = stencil[1:4]
        weight00[pixel], weight01[pixel], weight10[pixel], weight11[pixel] = stencil[5:8]
    end

    return RetinalMapPlan(
        source_size,
        output_size,
        index00,
        index01,
        index10,
        index11,
        weight00,
        weight01,
        weight10,
        weight11,
    )
end

"""Apply a cached wrapped retinal projection without allocating a new image."""
function retinal_transform!(output::AbstractMatrix, input_img::AbstractMatrix, plan::RetinalMapPlan)
    size(input_img) == plan.source_size || throw(DimensionMismatch("retinal-map input does not match its plan."))
    size(output) == plan.output_size || throw(DimensionMismatch("retinal-map output does not match its plan."))

    @inbounds for pixel in eachindex(output)
        output[pixel] =
            plan.weight00[pixel] * input_img[plan.index00[pixel]] +
            plan.weight01[pixel] * input_img[plan.index01[pixel]] +
            plan.weight10[pixel] * input_img[plan.index10[pixel]] +
            plan.weight11[pixel] * input_img[plan.index11[pixel]]
    end

    return output
end

"""
    retinal_transform(input; output_size=size(input), angle_origin=0)

Map a cortical sheet back into retinal coordinates. Cortical rows encode polar
angle and cortical columns encode logarithmic radius/eccentricity. Sampling
wraps across both cortical-grid axes, including the square corners of the
output.
"""
function retinal_transform(input_img::AbstractMatrix; output_size=size(input_img), angle_origin=0)
    T = eltype(input_img)
    T <: AbstractFloat || throw(ArgumentError("retinal_transform requires floating-point input."))
    plan = retinal_map_plan(size(input_img); output_size=output_size, angle_origin=angle_origin, dtype=T)
    output = Matrix{T}(undef, output_size)
    return retinal_transform!(output, input_img, plan)
end
