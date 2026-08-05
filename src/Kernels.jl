function gaussian_kernel_2d(x, y, sigma)
    s = float(sigma)
    norm_factor = inv(pi * s^2)
    exponent = -(abs(x)^2 + abs(y)^2) / s^2
    return norm_factor * exp(exponent)
end

function generate_gaussian_kernel(sigma, N::Integer; dtype::Type{T}=Float32) where {T<:AbstractFloat}
    isodd(N) || throw(ArgumentError("N must be odd to center the Gaussian kernel."))
    radius = div(N, 2)
    kernel = Matrix{T}(undef, N, N)
    s = T(sigma)
    norm_factor = inv(T(pi) * s^2)

    for col in 1:N, row in 1:N
        x = T(col - radius - 1)
        y = T(row - radius - 1)
        kernel[row, col] = norm_factor * exp(-(abs(x)^2 + abs(y)^2) / s^2)
    end

    return kernel
end
