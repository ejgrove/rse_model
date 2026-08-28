"""Return `value` when it is positive and odd, otherwise return the next odd integer."""
function odd_positive_int(value)
    n = round(Int, value)
    n > 0 || throw(ArgumentError("grid size must be positive."))
    return isodd(n) ? n : n + 1
end

"""Return whether `n` factors entirely into the small primes handled efficiently by FFTW."""
function is_fast_fft_size(n::Integer; factors=(2, 3, 5, 7))
    n > 0 || return false
    remaining = n
    for factor in factors
        while remaining % factor == 0
            remaining = div(remaining, factor)
        end
    end
    return remaining == 1
end

"""Return the first FFT-friendly odd grid size greater than or equal to `value`."""
function next_fast_odd_size(value)
    n = odd_positive_int(value)
    while !is_fast_fft_size(n)
        n += 2
    end
    return n
end
