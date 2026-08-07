function _wrap_index(i::Integer, n::Integer)
    return mod(i, n) + 1
end

function _gridwrap_bilinear(img::AbstractMatrix, y::T, x::T) where {T<:AbstractFloat}
    height, width = size(img)
    x0 = floor(Int, x)
    y0 = floor(Int, y)
    dx = x - T(x0)
    dy = y - T(y0)

    x1 = x0 + 1
    y1 = y0 + 1
    v00 = img[_wrap_index(y0, height), _wrap_index(x0, width)]
    v01 = img[_wrap_index(y0, height), _wrap_index(x1, width)]
    v10 = img[_wrap_index(y1, height), _wrap_index(x0, width)]
    v11 = img[_wrap_index(y1, height), _wrap_index(x1, width)]

    return (one(T) - dy) * ((one(T) - dx) * v00 + dx * v01) +
           dy * ((one(T) - dx) * v10 + dx * v11)
end

function retinal_transform(input_img::AbstractMatrix; output_size=size(input_img))
    source_height, source_width = size(input_img)
    height, width = output_size
    T = eltype(input_img)
    output = Matrix{T}(undef, height, width)

    for col in 1:width, row in 1:height
        x = T(-1) + T(2) * T(col - 1) / T(max(width - 1, 1))
        y = T(-1) + T(2) * T(row - 1) / T(max(height - 1, 1))
        r = hypot(x, y)
        theta = mod(atan(y, x) + T(2pi), T(2pi))
        r_scaled = log(r + T(1e-26)) / T(2pi)
        theta_scaled = theta / T(2pi)
        # Cortical rows encode polar angle; cortical columns encode radius/eccentricity.
        x_in = r_scaled * T(source_width)
        y_in = theta_scaled * T(source_height)
        output[row, col] = _gridwrap_bilinear(input_img, y_in, x_in)
    end

    return output
end

function ensure_unique_path(path::AbstractString; width::Integer=3, start::Integer=1)
    parent = dirname(path)
    base = basename(path)
    stem, suffix = splitext(base)
    match_suffix = match(r"_(\d+)$", stem)

    if match_suffix === nothing
        base_stem = stem
        number = start - 1
    else
        base_stem = stem[1:(first(match_suffix.offsets) - 2)]
        number = max(parse(Int, match_suffix.captures[1]), start - 1)
    end

    candidate = path
    while ispath(candidate)
        number += 1
        candidate_name = string(base_stem, "_", lpad(number, width, "0"), suffix)
        candidate = isempty(parent) ? candidate_name : joinpath(parent, candidate_name)
    end
    return candidate
end

const PLASMA_STOPS = (
    (13, 8, 135),
    (84, 3, 160),
    (139, 10, 165),
    (185, 50, 137),
    (219, 92, 104),
    (244, 136, 73),
    (254, 188, 43),
    (240, 249, 33),
)

const NIPY_STOPS = (
    (0, 0, 0),
    (0, 0, 180),
    (0, 140, 255),
    (0, 180, 80),
    (230, 230, 0),
    (255, 120, 0),
    (220, 0, 0),
    (255, 255, 255),
)

function _palette_stops(cmap::AbstractString)
    key = lowercase(cmap)
    if key == "nipy_spectral"
        return NIPY_STOPS
    elseif key in ("gray", "grey", "grayscale")
        return ((0, 0, 0), (255, 255, 255))
    else
        return PLASMA_STOPS
    end
end

function _palette_rgb(value, stops)
    t = clamp(Float64(value), 0.0, 1.0)
    scaled = t * (length(stops) - 1)
    idx = min(floor(Int, scaled) + 1, length(stops) - 1)
    frac = scaled - (idx - 1)
    c0 = stops[idx]
    c1 = stops[idx + 1]
    return (
        UInt8(round((1 - frac) * c0[1] + frac * c1[1])),
        UInt8(round((1 - frac) * c0[2] + frac * c1[2])),
        UInt8(round((1 - frac) * c0[3] + frac * c1[3])),
    )
end

function _palette_rgb(value, cmap::AbstractString)
    return _palette_rgb(value, _palette_stops(cmap))
end

function _normalized_values(img)
    lo = minimum(img)
    hi = maximum(img)
    if hi == lo
        return fill(0.0, size(img))
    end
    return clamp.((img .- lo) ./ (hi - lo), 0, 1)
end

function _heatmap_rgb(img; cmap::AbstractString="plasma")
    rows, cols = size(img)
    lo = minimum(img)
    hi = maximum(img)
    scale = hi == lo ? 0.0 : inv(Float64(hi - lo))
    stops = _palette_stops(cmap)
    rgb = Array{UInt8}(undef, rows, cols, 3)
    @inbounds for col in 1:cols, row in 1:rows
        value = hi == lo ? 0.0 : (Float64(img[row, col] - lo) * scale)
        r, g, b = _palette_rgb(value, stops)
        rgb[row, col, 1] = r
        rgb[row, col, 2] = g
        rgb[row, col, 3] = b
    end
    return rgb
end

function _uint32be(n)
    value = UInt32(n)
    return UInt8[
        UInt8((value >> 24) & 0xff),
        UInt8((value >> 16) & 0xff),
        UInt8((value >> 8) & 0xff),
        UInt8(value & 0xff),
    ]
end

function _crc32(bytes)
    crc = UInt32(0xffffffff)
    for byte in bytes
        crc = xor(crc, UInt32(byte))
        for _ in 1:8
            if (crc & UInt32(1)) == 0
                crc >>= 1
            else
                crc = xor(crc >> 1, UInt32(0xedb88320))
            end
        end
    end
    return ~crc
end

function _adler32(bytes)
    a = UInt32(1)
    b = UInt32(0)
    modulus = UInt32(65521)
    for byte in bytes
        a = mod(a + UInt32(byte), modulus)
        b = mod(b + a, modulus)
    end
    return (b << 16) | a
end

function _zlib_stored_stream(raw)
    out = UInt8[0x78, 0x01]
    start = 1
    while start <= length(raw)
        chunk_len = min(65535, length(raw) - start + 1)
        final_block = start + chunk_len - 1 == length(raw)
        len = UInt16(chunk_len)
        nlen = ~len
        push!(out, final_block ? UInt8(0x01) : UInt8(0x00))
        push!(out, UInt8(len & 0xff), UInt8((len >> 8) & 0xff))
        push!(out, UInt8(nlen & 0xff), UInt8((nlen >> 8) & 0xff))
        append!(out, raw[start:(start + chunk_len - 1)])
        start += chunk_len
    end
    append!(out, _uint32be(_adler32(raw)))
    return out
end

function _write_png_chunk(io, chunk_type, data)
    type_bytes = Vector{UInt8}(codeunits(chunk_type))
    write(io, _uint32be(length(data)))
    write(io, type_bytes)
    write(io, data)
    write(io, _uint32be(_crc32(vcat(type_bytes, data))))
end

function _write_rgb_png(path, rgb::Array{UInt8,3})
    rows, cols, channels = size(rgb)
    channels == 3 || throw(ArgumentError("RGB PNG data must have three channels."))

    raw = UInt8[]
    sizehint!(raw, rows * (1 + 3 * cols))
    for row in 1:rows
        push!(raw, 0x00)
        for col in 1:cols
            push!(raw, rgb[row, col, 1], rgb[row, col, 2], rgb[row, col, 3])
        end
    end

    ihdr = UInt8[]
    append!(ihdr, _uint32be(cols))
    append!(ihdr, _uint32be(rows))
    append!(ihdr, UInt8[8, 2, 0, 0, 0])

    open(path, "w") do io
        write(io, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
        _write_png_chunk(io, "IHDR", ihdr)
        _write_png_chunk(io, "IDAT", _zlib_stored_stream(raw))
        _write_png_chunk(io, "IEND", UInt8[])
    end
end

function _save_heatmap(path, img; cmap::AbstractString="plasma")
    _write_rgb_png(path, _heatmap_rgb(img; cmap=cmap))
end

function _label_text(t, Se, Si, A, period, N, p::ModelParams)
    return string(
        round(Int, t), " ms - ",
        "A:", round(A; digits=2), " ",
        "T:", period, " ",
        "Se:", round(Se; digits=2), " ",
        "Si:", round(Si; digits=2), " ",
        "dt:", p.dt, " ",
        "N:", N, " ",
        "V:", p.V,
    )
end

function make_images(snapshot::Snapshot; Se, Si, A, T, N, images, out_path, dpi, label, cmap, p::ModelParams, kwargs...)
    mkpath(out_path)
    cortical_activity = snapshot.cortical_activity
    retinal_activity = retinal_transform(cortical_activity)

    if images in ("cortical", "both")
        filename = ensure_unique_path(joinpath(out_path, "cortical_$(round(Int, snapshot.t))ms_N$(N).png"))
        _save_heatmap(filename, cortical_activity; cmap=cmap)
    end

    if images in ("retinal", "both")
        filename = ensure_unique_path(joinpath(out_path, "retinal_$(round(Int, snapshot.t))ms_N$(N).png"))
        _save_heatmap(filename, retinal_activity; cmap=cmap)
    end

    if label
        open(joinpath(out_path, "labels.txt"), "a") do io
            println(io, basename(out_path), ": ", _label_text(snapshot.t, Se, Si, A, T, N, p))
        end
    end
end

function _composite_side_by_side(left, right; gap::Integer=8)
    rows = max(size(left, 1), size(right, 1))
    cols = size(left, 2) + gap + size(right, 2)
    out = fill(UInt8(255), rows, cols, 3)
    out[1:size(left, 1), 1:size(left, 2), :] .= left
    start_col = size(left, 2) + gap + 1
    out[1:size(right, 1), start_col:(start_col + size(right, 2) - 1), :] .= right
    return out
end

function make_plot(snapshot::Snapshot; Se, Si, A, T, N, contours, cmap, p::ModelParams, out_file)
    cortical_rgb = _heatmap_rgb(snapshot.cortical_activity; cmap=cmap)
    retinal_rgb = _heatmap_rgb(retinal_transform(snapshot.cortical_activity); cmap=cmap)
    _write_rgb_png(out_file, _composite_side_by_side(cortical_rgb, retinal_rgb))
end

function _gif_palette(cmap::AbstractString)
    palette = UInt8[]
    for i in 0:255
        r, g, b = _palette_rgb(i / 255, cmap)
        push!(palette, r, g, b)
    end
    return palette
end

function _heatmap_indices(img)
    values = _normalized_values(img)
    rows, cols = size(values)
    indexed = Matrix{UInt8}(undef, rows, cols)
    for col in 1:cols, row in 1:rows
        indexed[row, col] = UInt8(round(Int, clamp(values[row, col] * 255, 0, 255)))
    end
    return indexed
end

function _write_le16(io, value)
    v = UInt16(value)
    write(io, UInt8(v & 0xff), UInt8((v >> 8) & 0xff))
end

function _pack_lzw_codes(codes)
    out = UInt8[]
    acc = UInt32(0)
    bits = 0
    for code in codes
        acc |= UInt32(code) << bits
        bits += 9
        while bits >= 8
            push!(out, UInt8(acc & 0xff))
            acc >>= 8
            bits -= 8
        end
    end
    if bits > 0
        push!(out, UInt8(acc & 0xff))
    end
    return out
end

function _gif_lzw_data(frame::Matrix{UInt8})
    clear_code = UInt16(256)
    end_code = UInt16(257)
    codes = UInt16[clear_code]
    emitted = 0

    for row in 1:size(frame, 1), col in 1:size(frame, 2)
        if emitted >= 240
            push!(codes, clear_code)
            emitted = 0
        end
        push!(codes, UInt16(frame[row, col]))
        emitted += 1
    end

    push!(codes, end_code)
    return _pack_lzw_codes(codes)
end

function _write_subblocks(io, data)
    start = 1
    while start <= length(data)
        chunk_len = min(255, length(data) - start + 1)
        write(io, UInt8(chunk_len))
        write(io, data[start:(start + chunk_len - 1)])
        start += chunk_len
    end
    write(io, UInt8(0))
end

function _write_gif(path, frames::Vector{Matrix{UInt8}}; fps::Integer=50, cmap::AbstractString="plasma")
    isempty(frames) && return
    height, width = size(first(frames))
    delay_cs = max(1, round(Int, 100 / fps))

    open(path, "w") do io
        write(io, "GIF89a")
        _write_le16(io, width)
        _write_le16(io, height)
        write(io, UInt8(0xf7), UInt8(0), UInt8(0))
        write(io, _gif_palette(cmap))

        write(io, UInt8[0x21, 0xff, 0x0b])
        write(io, "NETSCAPE2.0")
        write(io, UInt8[0x03, 0x01, 0x00, 0x00, 0x00])

        for frame in frames
            size(frame) == (height, width) || throw(ArgumentError("All GIF frames must have matching shapes."))
            write(io, UInt8[0x21, 0xf9, 0x04, 0x00])
            _write_le16(io, delay_cs)
            write(io, UInt8[0x00, 0x00])

            write(io, UInt8(0x2c))
            _write_le16(io, 0)
            _write_le16(io, 0)
            _write_le16(io, width)
            _write_le16(io, height)
            write(io, UInt8(0x00))
            write(io, UInt8(8))
            _write_subblocks(io, _gif_lzw_data(frame))
        end

        write(io, UInt8(0x3b))
    end
end

function make_gif(snapshots::Vector{<:Snapshot}; Se, Si, A, T, N, contours, cmap, out_path, label, dpi, p::ModelParams, fps=50)
    mkpath(out_path)
    frames = [_heatmap_indices(retinal_transform(snapshot.cortical_activity)) for snapshot in snapshots]
    _write_gif(joinpath(out_path, "simulation.gif"), frames; fps=fps, cmap=cmap)
end
