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
    cap = [dipole_double_sech_map(DOUBLE_SECH_E_MAX, pi / 2 - pi * (i - 1) / max(samples_p - 1, 1)) for i in 2:samples_p]
    bottom = [dipole_double_sech_map(_eccentricity_sample(i, samples_e), -pi / 2) for i in samples_e:-1:1]
    return vcat(top, cap, bottom)
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

function _sample_bilinear_zero(img::AbstractMatrix, mask::AbstractMatrix{Bool}, y::Float64, x::Float64)
    rows, cols = size(img)
    if x < 1 || y < 1 || x > cols || y > rows
        return zero(eltype(img))
    end
    x0 = floor(Int, x)
    y0 = floor(Int, y)
    x1 = min(x0 + 1, cols)
    y1 = min(y0 + 1, rows)
    x0 = max(x0, 1)
    y0 = max(y0, 1)
    dx = x - x0
    dy = y - y0

    v00 = mask[y0, x0] ? img[y0, x0] : zero(eltype(img))
    v01 = mask[y0, x1] ? img[y0, x1] : zero(eltype(img))
    v10 = mask[y1, x0] ? img[y1, x0] : zero(eltype(img))
    v11 = mask[y1, x1] ? img[y1, x1] : zero(eltype(img))

    return (1 - dy) * ((1 - dx) * v00 + dx * v01) +
           dy * ((1 - dx) * v10 + dx * v11)
end

function _double_sech_grid_position(geometry::FieldGeometry, eccentricity::Real, polar::Real)
    w = dipole_double_sech_map(eccentricity, polar)
    x = 1 + (real(w) - geometry.x_min) / (geometry.x_max - geometry.x_min) * (geometry.cols - 1)
    y = 1 + (geometry.y_max - imag(w)) / (geometry.y_max - geometry.y_min) * (geometry.rows - 1)
    return y, x
end

function double_sech_retinal_transform(
    left_activity::AbstractMatrix,
    right_activity::AbstractMatrix,
    geometry::FieldGeometry;
    output_size=(geometry.rows, geometry.rows),
)
    geometry.kind == :double_sech || throw(ArgumentError("double_sech_retinal_transform requires double-sech geometry."))
    height, width = output_size
    T = promote_type(eltype(left_activity), eltype(right_activity))
    output = Matrix{T}(undef, height, width)

    @inbounds for col in 1:width, row in 1:height
        x_visual = -1 + 2 * (col - 1) / max(width - 1, 1)
        y_visual = 1 - 2 * (row - 1) / max(height - 1, 1)
        r = hypot(x_visual, y_visual)
        if r > 1
            output[row, col] = zero(T)
            continue
        end

        eccentricity = DOUBLE_SECH_E_MAX * r
        polar = r < 1e-9 ? 0.0 : atan(y_visual, abs(x_visual))
        y_source, x_source = _double_sech_grid_position(geometry, eccentricity, polar)
        source = x_visual >= 0 ? left_activity : right_activity
        output[row, col] = T(_sample_bilinear_zero(source, geometry.mask, y_source, x_source))
    end

    return output
end
