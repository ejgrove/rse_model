"""Node lattice, mask, and cortical-coordinate bounds for one field geometry."""
Base.@kwdef struct FieldGeometry
    kind::Symbol = :square
    rows::Int
    cols::Int
    density::Float64 = 1.0
    mask::Matrix{Bool}
    mask_float32::Matrix{Float32}
    x_min::Float64 = 0.0
    x_max::Float64 = 1.0
    y_min::Float64 = 0.0
    y_max::Float64 = 1.0
end

const DOUBLE_SECH_BASE_ROWS = 81
const DOUBLE_SECH_A = 1.05
const DOUBLE_SECH_B = 90.0
const DOUBLE_SECH_K = 19.3
const DOUBLE_SECH_E_MAX = 90.0

# Geometry construction

function _normalize_field_geometry(kind::Symbol)
    kind in (:square, :rect, :rectangular) && return :square
    kind in (:double_sech, :double_sech_v1, :banded_double_sech) && return :double_sech
    throw(ArgumentError("field geometry must be :square or :double_sech."))
end

function _sech(x)
    return inv(cosh(x))
end

function _density_scaled_size(n::Integer, density::Real)
    density > 0 || throw(ArgumentError("field density must be positive."))
    return odd_positive_int(max(5, round(Int, n * Float64(density))))
end

function double_sech_shear(eccentricity::Real, polar::Real, pole::Real)
    E = max(Float64(eccentricity), 1e-6)
    exponent = _sech(log(E / Float64(pole)) * 0.76) * 0.1821
    return _sech(Float64(polar))^exponent
end

"""Evaluate the dipole double-sech cortical map described by Schira et al. (2010)."""
function dipole_double_sech_map(
    eccentricity::Real,
    polar::Real;
    a::Real=DOUBLE_SECH_A,
    b::Real=DOUBLE_SECH_B,
    k::Real=DOUBLE_SECH_K,
)
    E = Float64(eccentricity)
    P = Float64(polar)
    fa = double_sech_shear(E, P, a)
    fb = double_sech_shear(E, P, b)
    numerator = E * cis(P * fa) + Float64(a)
    denominator = E * cis(P * fb) + Float64(b)
    return Float64(k) * log(numerator / denominator)
end

function _eccentricity_sample(i::Integer, n::Integer)
    t = (i - 1) / max(n - 1, 1)
    return DOUBLE_SECH_E_MAX * t^1.35
end

function _double_sech_polygon(samples_e::Integer=360, samples_p::Integer=120)
    top = [dipole_double_sech_map(_eccentricity_sample(i, samples_e), pi / 2) for i in 1:samples_e]
    top_edge = top[end]
    bottom_edge = dipole_double_sech_map(DOUBLE_SECH_E_MAX, -pi / 2)
    peripheral_vertex = dipole_double_sech_map(DOUBLE_SECH_E_MAX, 0.0)
    top_control = Complex(real(peripheral_vertex), imag(top_edge))
    bottom_control = Complex(real(peripheral_vertex), imag(bottom_edge))
    top_cap = [
        (1 - t)^2 * top_edge + 2 * (1 - t) * t * top_control + t^2 * peripheral_vertex
        for t in range(0, 1; length=max(3, div(samples_p, 2)))[2:end]
    ]
    bottom_cap = [
        (1 - t)^2 * peripheral_vertex + 2 * (1 - t) * t * bottom_control + t^2 * bottom_edge
        for t in range(0, 1; length=max(3, div(samples_p, 2)))[2:end]
    ]
    bottom = [dipole_double_sech_map(_eccentricity_sample(i, samples_e), -pi / 2) for i in samples_e:-1:1]
    return vcat(top, top_cap, bottom_cap, bottom)
end

function _double_sech_bounds(poly)
    xs = real.(poly)
    ys = imag.(poly)
    x_min, x_max = extrema(xs)
    y_min, y_max = extrema(ys)
    pad_x = 0.035 * (x_max - x_min)
    pad_y = 0.035 * (y_max - y_min)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y
end

function _point_in_polygon(x::Float64, y::Float64, poly)
    inside = false
    j = length(poly)
    @inbounds for i in eachindex(poly)
        xi = real(poly[i])
        yi = imag(poly[i])
        xj = real(poly[j])
        yj = imag(poly[j])
        intersects = (yi > y) != (yj > y) &&
            x < (xj - xi) * (y - yi) / (yj - yi + eps(Float64)) + xi
        intersects && (inside = !inside)
        j = i
    end
    return inside
end

function _double_sech_v1_mask(rows::Integer, cols::Integer, bounds, poly)
    x_min, x_max, y_min, y_max = bounds
    mask = Matrix{Bool}(undef, rows, cols)

    @inbounds for col in 1:cols, row in 1:rows
        x = x_min + (col - 1) / max(cols - 1, 1) * (x_max - x_min)
        y = y_max - (row - 1) / max(rows - 1, 1) * (y_max - y_min)
        mask[row, col] = _point_in_polygon(x, y, poly)
    end

    return mask
end

"""
    field_geometry(kind, n=81; density=1)

Construct either an unmasked square lattice or a uniformly sampled V1-shaped
double-sech lattice. `density` scales both double-sech lattice dimensions.
"""
function field_geometry(kind::Symbol, n::Integer=DOUBLE_SECH_BASE_ROWS; density::Real=1.0)
    normalized = _normalize_field_geometry(kind)
    rows = normalized == :square ? _density_scaled_size(n, density) : _density_scaled_size(DOUBLE_SECH_BASE_ROWS, density)
    poly = normalized == :double_sech ? _double_sech_polygon() : ComplexF64[]
    bounds = normalized == :double_sech ? _double_sech_bounds(poly) : (0.0, 1.0, 0.0, 1.0)
    x_min, x_max, y_min, y_max = bounds
    aspect = normalized == :double_sech ? (x_max - x_min) / max(y_max - y_min, eps(Float64)) : 1.0
    cols = normalized == :square ? rows : odd_positive_int(max(7, round(Int, rows * aspect)))
    mask = normalized == :square ? fill(true, rows, cols) : _double_sech_v1_mask(rows, cols, bounds, poly)
    mask_float32 = Float32.(mask)
    return FieldGeometry(
        kind=normalized,
        rows=rows,
        cols=cols,
        density=Float64(density),
        mask=mask,
        mask_float32=mask_float32,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
end

# Field masks and coupling borders

function has_field_mask(geometry::FieldGeometry)
    return geometry.kind != :square
end

function apply_field_mask!(U::AbstractMatrix, geometry::FieldGeometry)
    has_field_mask(geometry) || return U
    @. U *= geometry.mask_float32
    return U
end

function apply_field_mask!(Ue::AbstractMatrix, Ui::AbstractMatrix, geometry::FieldGeometry)
    apply_field_mask!(Ue, geometry)
    apply_field_mask!(Ui, geometry)
    return Ue, Ui
end

function field_border_mask(mask::AbstractMatrix{Bool}, width::Integer)
    rows, cols = size(mask)
    border = falses(rows, cols)
    radius = max(0, width)
    radius == 0 && return border

    @inbounds for col in 1:cols, row in 1:rows
        mask[row, col] || continue
        is_border = false
        for dc in -radius:radius
            is_border && break
            cc = col + dc
            for dr in -radius:radius
                rr = row + dr
                if rr < 1 || rr > rows || cc < 1 || cc > cols || !mask[rr, cc]
                    is_border = true
                    break
                end
            end
        end
        border[row, col] = is_border
    end

    return border
end

# Double-sech inverse mapping

struct MaskedBilinearPlan{T<:AbstractFloat}
    index00::Vector{Int32}
    index01::Vector{Int32}
    index10::Vector{Int32}
    index11::Vector{Int32}
    weight00::Vector{T}
    weight01::Vector{T}
    weight10::Vector{T}
    weight11::Vector{T}
end

"""Cached masked bilinear coordinates for a two-hemisphere double-sech projection."""
struct DoubleSechRetinalPlan{T<:AbstractFloat}
    source_size::Tuple{Int,Int}
    output_size::Tuple{Int,Int}
    left::MaskedBilinearPlan{T}
    right::MaskedBilinearPlan{T}
end

function _empty_masked_bilinear_plan(count::Int, ::Type{T}) where {T<:AbstractFloat}
    return MaskedBilinearPlan(
        ones(Int32, count),
        ones(Int32, count),
        ones(Int32, count),
        ones(Int32, count),
        zeros(T, count),
        zeros(T, count),
        zeros(T, count),
        zeros(T, count),
    )
end

function _masked_bilinear_stencil(mask::AbstractMatrix{Bool}, y::Float64, x::Float64, ::Type{T}) where {T<:AbstractFloat}
    rows, cols = size(mask)
    if x < 1 || y < 1 || x > cols || y > rows
        return (Int32(1), Int32(1), Int32(1), Int32(1), zero(T), zero(T), zero(T), zero(T))
    end
    x0 = floor(Int, x)
    y0 = floor(Int, y)
    x1 = min(x0 + 1, cols)
    y1 = min(y0 + 1, rows)
    x0 = max(x0, 1)
    y0 = max(y0, 1)
    dx = x - x0
    dy = y - y0

    weights = (
        mask[y0, x0] ? (1 - dy) * (1 - dx) : 0.0,
        mask[y0, x1] ? (1 - dy) * dx : 0.0,
        mask[y1, x0] ? dy * (1 - dx) : 0.0,
        mask[y1, x1] ? dy * dx : 0.0,
    )
    total_weight = sum(weights)
    total_weight > 0 || return (Int32(1), Int32(1), Int32(1), Int32(1), zero(T), zero(T), zero(T), zero(T))
    linear = LinearIndices(mask)
    indices = (
        Int32(linear[y0, x0]),
        Int32(linear[y0, x1]),
        Int32(linear[y1, x0]),
        Int32(linear[y1, x1]),
    )
    normalized = ntuple(index -> T(weights[index] / total_weight), 4)
    return (
        indices[1], indices[2], indices[3], indices[4],
        normalized[1], normalized[2], normalized[3], normalized[4],
    )
end

function _double_sech_grid_position(geometry::FieldGeometry, eccentricity::Real, polar::Real)
    w = dipole_double_sech_map(eccentricity, polar)
    x = 1 + (real(w) - geometry.x_min) / (geometry.x_max - geometry.x_min) * (geometry.cols - 1)
    y = 1 + (geometry.y_max - imag(w)) / (geometry.y_max - geometry.y_min) * (geometry.rows - 1)
    return y, x
end

function _set_masked_stencil!(plan::MaskedBilinearPlan{T}, pixel::Int, stencil, blend::Real) where {T}
    plan.index00[pixel], plan.index01[pixel], plan.index10[pixel], plan.index11[pixel] = stencil[1:4]
    blend_t = T(blend)
    plan.weight00[pixel] = blend_t * stencil[5]
    plan.weight01[pixel] = blend_t * stencil[6]
    plan.weight10[pixel] = blend_t * stencil[7]
    plan.weight11[pixel] = blend_t * stencil[8]
    return
end

"""Precompute the masked inverse double-sech map for a fixed output resolution."""
function double_sech_retinal_plan(
    geometry::FieldGeometry;
    output_size=(geometry.rows, geometry.rows),
    seam_blend_pixels::Real=1,
    flip_right_angular_axis::Bool=true,
    dtype::Type{T}=Float32,
) where {T<:AbstractFloat}
    geometry.kind == :double_sech || throw(ArgumentError("double_sech_retinal_transform requires double-sech geometry."))
    height, width = output_size
    height > 0 && width > 0 || throw(ArgumentError("double-sech retinal-map output dimensions must be positive."))
    left = _empty_masked_bilinear_plan(height * width, T)
    right = _empty_masked_bilinear_plan(height * width, T)
    visual_pixel_width = 2 / max(width - 1, 1)
    seam_blend_width = max(0.0, Float64(seam_blend_pixels)) * visual_pixel_width

    @inbounds for col in 1:width, row in 1:height
        x_visual = -1 + 2 * (col - 1) / max(width - 1, 1)
        y_visual = 1 - 2 * (row - 1) / max(height - 1, 1)
        r = hypot(x_visual, y_visual)
        r > 1 && continue

        eccentricity = DOUBLE_SECH_E_MAX * r
        polar = r < 1e-9 ? 0.0 : atan(y_visual, abs(x_visual))
        left_blend = if seam_blend_width > 0 && abs(x_visual) <= seam_blend_width
            clamp(0.5 + 0.5 * x_visual / seam_blend_width, 0.0, 1.0)
        else
            x_visual > 0 ? 1.0 : 0.0
        end
        right_blend = 1 - left_blend
        pixel = row + (col - 1) * height

        if left_blend > 0
            left_y, left_x = _double_sech_grid_position(geometry, eccentricity, polar)
            left_stencil = _masked_bilinear_stencil(geometry.mask, left_y, left_x, T)
            _set_masked_stencil!(left, pixel, left_stencil, left_blend)
        end
        if right_blend > 0
            right_polar = flip_right_angular_axis ? -polar : polar
            right_y, right_x = _double_sech_grid_position(geometry, eccentricity, right_polar)
            right_stencil = _masked_bilinear_stencil(geometry.mask, right_y, right_x, T)
            _set_masked_stencil!(right, pixel, right_stencil, right_blend)
        end
    end

    return DoubleSechRetinalPlan((geometry.rows, geometry.cols), output_size, left, right)
end

function _apply_masked_bilinear_plan(input::AbstractMatrix, plan::MaskedBilinearPlan, pixel::Int)
    @inbounds return (
        plan.weight00[pixel] * input[plan.index00[pixel]] +
        plan.weight01[pixel] * input[plan.index01[pixel]] +
        plan.weight10[pixel] * input[plan.index10[pixel]] +
        plan.weight11[pixel] * input[plan.index11[pixel]]
    )
end

"""Apply a cached double-sech projection without rebuilding its geometry."""
function double_sech_retinal_transform!(
    output::AbstractMatrix,
    left_activity::AbstractMatrix,
    right_activity::AbstractMatrix,
    plan::DoubleSechRetinalPlan,
)
    size(left_activity) == plan.source_size || throw(DimensionMismatch("left double-sech activity does not match its plan."))
    size(right_activity) == plan.source_size || throw(DimensionMismatch("right double-sech activity does not match its plan."))
    size(output) == plan.output_size || throw(DimensionMismatch("double-sech output does not match its plan."))

    @inbounds for pixel in eachindex(output)
        output[pixel] =
            _apply_masked_bilinear_plan(left_activity, plan.left, pixel) +
            _apply_masked_bilinear_plan(right_activity, plan.right, pixel)
    end
    return output
end

function double_sech_retinal_transform(
    left_activity::AbstractMatrix,
    right_activity::AbstractMatrix,
    geometry::FieldGeometry;
    output_size=(geometry.rows, geometry.rows),
    seam_blend_pixels::Real=1,
    flip_right_angular_axis::Bool=true,
)
    T = promote_type(eltype(left_activity), eltype(right_activity))
    T <: AbstractFloat || throw(ArgumentError("double_sech_retinal_transform requires floating-point input."))
    plan = double_sech_retinal_plan(
        geometry;
        output_size=output_size,
        seam_blend_pixels=seam_blend_pixels,
        flip_right_angular_axis=flip_right_angular_axis,
        dtype=T,
    )
    output = Matrix{T}(undef, output_size)
    return double_sech_retinal_transform!(output, left_activity, right_activity, plan)
end
