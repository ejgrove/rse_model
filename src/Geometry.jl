Base.@kwdef struct FieldGeometry
    kind::Symbol = :square
    rows::Int
    cols::Int
    density::Float64 = 1.0
    mask::Matrix{Bool}
    mask_float32::Matrix{Float32}
end

function _normalize_field_geometry(kind::Symbol)
    kind in (:square, :rect, :rectangular) && return :square
    kind in (:double_sech, :double_sech_v1, :banded_double_sech) && return :double_sech
    throw(ArgumentError("field geometry must be :square or :double_sech."))
end

function _sech(x)
    return inv(cosh(x))
end

function _smoothstep(edge0, edge1, x)
    edge0 == edge1 && return x < edge0 ? 0.0 : 1.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)
end

function _density_scaled_size(n::Integer, density::Real)
    density > 0 || throw(ArgumentError("field density must be positive."))
    return odd_positive_int(max(5, round(Int, n * Float64(density))))
end

function _double_sech_v1_mask(rows::Integer, cols::Integer)
    mask = Matrix{Bool}(undef, rows, cols)
    eps_ecc = 0.035

    @inbounds for col in 1:cols
        u = (col - 1) / max(cols - 1, 1)
        eccentricity = eps_ecc + (1 - eps_ecc) * u
        # Schira's shear is strongest near the foveal singularity and vertical
        # meridians; here it is used as a smooth shape term for V1-only layout.
        foveal_shear = _sech((log(eccentricity / 0.18) - 0.76) / 1.25)
        centerline = 0.5 + 0.12 * sin(pi * (u - 0.12)) * (0.35 + 0.65 * u)
        half_width = 0.30 + 0.045 * foveal_shear - 0.055 * u
        cap = _smoothstep(0.0, 0.18, u) * _smoothstep(1.0, 0.84, u)

        for row in 1:rows
            v = (row - 1) / max(rows - 1, 1)
            band_distance = abs(v - centerline)
            rounded_tip = ((u - 0.12) / 0.18)^2 + ((v - centerline) / max(half_width, 1e-6))^2
            in_band = band_distance <= half_width * (0.82 + 0.18 * cap)
            in_foveal_cap = u < 0.19 && rounded_tip <= 1.0
            mask[row, col] = in_band && (u >= 0.12 || in_foveal_cap)
        end
    end

    return mask
end

function field_geometry(kind::Symbol, n::Integer; density::Real=1.0)
    normalized = _normalize_field_geometry(kind)
    rows = _density_scaled_size(n, density)
    cols = normalized == :square ? rows : odd_positive_int(max(7, round(Int, rows * 1.45)))
    mask = normalized == :square ? fill(true, rows, cols) : _double_sech_v1_mask(rows, cols)
    mask_float32 = Float32.(mask)
    return FieldGeometry(
        kind=normalized,
        rows=rows,
        cols=cols,
        density=Float64(density),
        mask=mask,
        mask_float32=mask_float32,
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
