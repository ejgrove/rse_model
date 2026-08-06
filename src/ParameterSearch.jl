using ArgParse
using Distributed

Base.@kwdef struct ParameterSearchConfig
    N::Int = 81
    A_values::Vector{Float64} = _range_values(0.2, 1.5, 0.1)
    period_values::Vector{Float64} = _range_values(10.0, 150.0, 5.0)
    times_ms::Vector{Int} = [5000, 10000, 15000, 20000]
    Se::Float64 = 2.0
    Si::Float64 = 5.0
    backend::Symbol = :metal
    convolution::Symbol = :auto
    kernel_cutoff::Float64 = 3.0
    duty_cycle_percent::Union{Nothing,Float64} = 50.0
    seed::Int = 42
    seed_mode::Symbol = :same
    view::Symbol = :cortical
    cmap::String = "plasma"
    out_path::String = joinpath("outputs", "parameter_search")
    overwrite::Bool = false
    fft_flags = DEFAULT_FFT_FLAGS
    fftw_threads::Int = 1
    gpu_threads::Int = 256
    workers::Int = 1
    dry_run::Bool = false
end

const SEARCH_FONT = Dict{Char,Tuple{Vararg{String,7}}}(
    ' ' => ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    '-' => ("00000", "00000", "00000", "11110", "00000", "00000", "00000"),
    '.' => ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ',' => ("00000", "00000", "00000", "00000", "00000", "01100", "01000"),
    ':' => ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    '=' => ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    '(' => ("00110", "01000", "10000", "10000", "10000", "01000", "00110"),
    ')' => ("11000", "00100", "00010", "00010", "00010", "00100", "11000"),
    '/' => ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    '%' => ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    '0' => ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    '1' => ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    '2' => ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    '3' => ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    '4' => ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    '5' => ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    '6' => ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    '7' => ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    '8' => ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    '9' => ("01110", "10001", "10001", "01111", "00001", "00010", "11100"),
    'A' => ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    'N' => ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    'S' => ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    'T' => ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    'a' => ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    'c' => ("00000", "00000", "01110", "10000", "10000", "10001", "01110"),
    'd' => ("00001", "00001", "01111", "10001", "10001", "10001", "01111"),
    'e' => ("00000", "00000", "01110", "10001", "11111", "10000", "01110"),
    'f' => ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    'h' => ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    'i' => ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    'l' => ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    'm' => ("00000", "00000", "11010", "10101", "10101", "10101", "10101"),
    'n' => ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    'o' => ("00000", "00000", "01110", "10001", "10001", "10001", "01110"),
    'p' => ("00000", "00000", "11110", "10001", "11110", "10000", "10000"),
    'r' => ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    's' => ("00000", "00000", "01111", "10000", "01110", "00001", "11110"),
    't' => ("01000", "01000", "11100", "01000", "01000", "01001", "00110"),
    'u' => ("00000", "00000", "10001", "10001", "10001", "10011", "01101"),
    'v' => ("00000", "00000", "10001", "10001", "10001", "01010", "00100"),
    'w' => ("00000", "00000", "10001", "10001", "10101", "10101", "01010"),
    'x' => ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
    'y' => ("00000", "00000", "10001", "10001", "01111", "00001", "01110")
)

function _range_values(start, stop, step)
    step > 0 || throw(ArgumentError("range step must be positive."))
    values = Float64[]
    value = Float64(start)
    stop_value = Float64(stop)
    while value <= stop_value + step / 2
        push!(values, round(value; digits=10))
        value += step
    end
    return values
end

function _compact_number(value; digits=3)
    rounded = round(Float64(value); digits=digits)
    if isapprox(rounded, round(Int, rounded); atol=10.0^(-digits))
        return string(round(Int, rounded))
    end
    text = string(rounded)
    text = replace(text, r"0+$" => "")
    return replace(text, r"\.$" => "")
end

function _parse_csv_numbers(text)
    values = Float64[]
    for part in split(text, ",")
        stripped = strip(part)
        isempty(stripped) && continue
        push!(values, parse(Float64, stripped))
    end
    isempty(values) && throw(ArgumentError("Expected at least one numeric value."))
    return values
end

function _parse_times_ms(text)
    times = [round(Int, value * 1000) for value in _parse_csv_numbers(text)]
    all(>(0), times) || throw(ArgumentError("All sample times must be positive seconds."))
    return sort(unique(times))
end

function _validate_search_config(config::ParameterSearchConfig)
    N = odd_positive_int(config.N)
    isempty(config.A_values) && throw(ArgumentError("A sweep cannot be empty."))
    isempty(config.period_values) && throw(ArgumentError("T sweep cannot be empty."))
    isempty(config.times_ms) && throw(ArgumentError("sample times cannot be empty."))
    all(>(0), config.period_values) || throw(ArgumentError("T values must be positive."))
    all(>(0), config.times_ms) || throw(ArgumentError("sample times must be positive."))
    config.Se > 0 || throw(ArgumentError("Se must be positive."))
    config.Si > 0 || throw(ArgumentError("Si must be positive."))
    config.kernel_cutoff > 0 || throw(ArgumentError("kernel_cutoff must be positive."))
    if config.duty_cycle_percent !== nothing
        0 <= config.duty_cycle_percent <= 100 ||
            throw(ArgumentError("duty_cycle_percent must be between 0 and 100."))
    end
    config.fftw_threads > 0 || throw(ArgumentError("fftw_threads must be positive."))
    config.gpu_threads > 0 || throw(ArgumentError("gpu_threads must be positive."))
    config.workers > 0 || throw(ArgumentError("workers must be positive."))
    config.backend in (:cpu, :metal) || throw(ArgumentError("backend must be :cpu or :metal."))
    config.view in (:retinal, :cortical) || throw(ArgumentError("view must be :retinal or :cortical."))
    config.seed_mode in (:same, :increment) || throw(ArgumentError("seed_mode must be :same or :increment."))

    convolution = if config.convolution == :auto
        config.backend == :metal ? :separable : :fft
    else
        config.convolution
    end
    convolution in (:fft, :separable) || throw(ArgumentError("convolution must be :auto, :fft, or :separable."))
    config.backend == :metal || convolution == :fft ||
        throw(ArgumentError("CPU parameter searches currently support FFT convolution only."))
    workers = config.workers
    if config.backend == :metal && workers > 1
        @warn "Metal parameter searches use one GPU process; ignoring extra workers. Use --backend cpu --workers N to parallelize across CPU processes." workers
        workers = 1
    end

    return ParameterSearchConfig(
        N=N,
        A_values=sort(config.A_values),
        period_values=sort(config.period_values),
        times_ms=sort(unique(config.times_ms)),
        Se=config.Se,
        Si=config.Si,
        backend=config.backend,
        convolution=convolution,
        kernel_cutoff=config.kernel_cutoff,
        duty_cycle_percent=config.duty_cycle_percent,
        seed=config.seed,
        seed_mode=config.seed_mode,
        view=config.view,
        cmap=config.cmap,
        out_path=config.out_path,
        overwrite=config.overwrite,
        fft_flags=config.fft_flags,
        fftw_threads=config.fftw_threads,
        gpu_threads=config.gpu_threads,
        workers=workers,
        dry_run=config.dry_run,
    )
end

function _sample_interval_ms(times_ms)
    return foldl(gcd, times_ms)
end

function _format_duration(seconds)
    seconds = max(0, round(Int, seconds))
    hours = div(seconds, 3600)
    minutes = div(seconds % 3600, 60)
    secs = seconds % 60
    hours > 0 && return string(hours, "h ", minutes, "m ", secs, "s")
    minutes > 0 && return string(minutes, "m ", secs, "s")
    return string(secs, "s")
end

function _seed_for_sim(config::ParameterSearchConfig, index::Integer)
    config.seed < 0 && return nothing
    config.seed_mode == :increment && return config.seed + index - 1
    return config.seed
end

function _search_cell_rgb(snapshot::Snapshot, view::Symbol, cmap::AbstractString)
    img = view == :retinal ? retinal_transform(snapshot.cortical_activity) : snapshot.cortical_activity
    return _heatmap_rgb(img; cmap=cmap)
end

function _text_width(text, scale)
    return isempty(text) ? 0 : length(collect(text)) * 6 * scale - scale
end

function _draw_text!(canvas, row, col, text; scale::Integer=2, color=(UInt8(13), UInt8(38), UInt8(56)))
    cursor = col
    for ch in text
        glyph = get(SEARCH_FONT, ch, SEARCH_FONT[' '])
        for (glyph_row, line) in enumerate(glyph)
            for (glyph_col, bit) in enumerate(line)
                bit == '1' || continue
                top = row + (glyph_row - 1) * scale
                left = cursor + (glyph_col - 1) * scale
                for rr in top:(top + scale - 1), cc in left:(left + scale - 1)
                    1 <= rr <= size(canvas, 1) || continue
                    1 <= cc <= size(canvas, 2) || continue
                    canvas[rr, cc, 1] = color[1]
                    canvas[rr, cc, 2] = color[2]
                    canvas[rr, cc, 3] = color[3]
                end
            end
        end
        cursor += 6 * scale
    end
    return canvas
end

function _paste_rgb!(canvas, rgb, top, left)
    rows, cols, _channels = size(rgb)
    canvas[top:(top + rows - 1), left:(left + cols - 1), :] .= rgb
    return canvas
end

function _make_montage_canvas(config::ParameterSearchConfig, time_ms::Integer)
    cell = config.N
    gap = 2
    left_margin = 70
    top_margin = 58
    right_margin = 12
    bottom_margin = 20
    rows = top_margin + length(config.A_values) * cell + (length(config.A_values) - 1) * gap + bottom_margin
    cols = left_margin + length(config.period_values) * cell + (length(config.period_values) - 1) * gap + right_margin
    canvas = fill(UInt8(255), rows, cols, 3)

    title = string(
        "t=", _compact_number(time_ms / 1000), "s ",
        config.view, " N=", config.N,
        " Se=", _compact_number(config.Se),
        " Si=", _compact_number(config.Si),
        " duty=", _duty_text(config),
    )
    _draw_text!(canvas, 8, left_margin, title; scale=2)
    _draw_text!(canvas, 34, left_margin, "T (ms)"; scale=2)
    _draw_text!(canvas, top_margin + 6, 10, "A"; scale=2)

    for (t_idx, period) in enumerate(config.period_values)
        label = _compact_number(period)
        x = left_margin + (t_idx - 1) * (cell + gap) + max(1, div(cell - _text_width(label, 1), 2))
        _draw_text!(canvas, top_margin - 15, x, label; scale=1, color=(UInt8(96), UInt8(114), UInt8(132)))
    end

    for (a_idx, amplitude) in enumerate(config.A_values)
        plot_row = length(config.A_values) - a_idx + 1
        label = _compact_number(amplitude)
        y = top_margin + (plot_row - 1) * (cell + gap) + max(1, div(cell - 7 * 2, 2))
        x = max(1, left_margin - 10 - _text_width(label, 2))
        _draw_text!(canvas, y, x, label; scale=2, color=(UInt8(96), UInt8(114), UInt8(132)))
    end

    return canvas
end

function _cell_origin(config::ParameterSearchConfig, a_idx::Integer, t_idx::Integer)
    cell = config.N
    gap = 2
    left_margin = 70
    top_margin = 58
    plot_row = length(config.A_values) - a_idx + 1
    top = top_margin + (plot_row - 1) * (cell + gap)
    left = left_margin + (t_idx - 1) * (cell + gap)
    return top, left
end

function _prepare_search_output_dir(config::ParameterSearchConfig)
    out_path = config.overwrite ? config.out_path : ensure_unique_path(config.out_path)
    mkpath(out_path)
    return out_path
end

function _duty_text(config::ParameterSearchConfig)
    config.duty_cycle_percent === nothing && return "model"
    return string(_compact_number(config.duty_cycle_percent), "%")
end

function _search_jobs(config::ParameterSearchConfig)
    total = length(config.A_values) * length(config.period_values)
    jobs = NamedTuple[]
    sim_index = 0

    for (a_idx, amplitude) in enumerate(config.A_values)
        for (t_idx, period) in enumerate(config.period_values)
            sim_index += 1
            push!(jobs, (
                index=sim_index,
                total=total,
                a_idx=a_idx,
                t_idx=t_idx,
                amplitude=amplitude,
                period=period,
                seed=_seed_for_sim(config, sim_index),
            ))
        end
    end

    return jobs
end

function _run_parameter_search_job(job, config::ParameterSearchConfig, interval_ms::Integer, max_time_ms::Integer)
    if config.backend == :cpu
        FFTW.set_num_threads(config.fftw_threads)
    end

    sim_start = time_ns()
    data = run_simulation(
        N=config.N,
        A=job.amplitude,
        T=job.period,
        Se=config.Se,
        Si=config.Si,
        start_time=minimum(config.times_ms),
        end_time=max_time_ms,
        seed=job.seed,
        plot=false,
        gif=false,
        interval=interval_ms,
        p=ModelParams(),
        fps=50,
        fft_flags=config.fft_flags,
        backend=config.backend,
        gpu_threads=config.gpu_threads,
        convolution=config.convolution,
        kernel_cutoff=config.kernel_cutoff,
        boundary=:periodic,
        duty_cycle_percent=config.duty_cycle_percent,
    )
    elapsed = (time_ns() - sim_start) / 1e9
    snapshots = Dict(round(Int, snapshot.t) => snapshot for snapshot in data.images)
    images = Dict{Int,Array{UInt8,3}}()

    for time_ms in config.times_ms
        snapshot = get(snapshots, time_ms, nothing)
        snapshot === nothing && continue
        images[time_ms] = _search_cell_rgb(snapshot, config.view, config.cmap)
    end

    return (
        index=job.index,
        total=job.total,
        a_idx=job.a_idx,
        t_idx=job.t_idx,
        amplitude=job.amplitude,
        period=job.period,
        seed=job.seed,
        compute_seconds=data.compute_seconds,
        elapsed_seconds=elapsed,
        images=images,
        worker=myid(),
    )
end

function _apply_search_result!(montages, config::ParameterSearchConfig, result)
    for time_ms in config.times_ms
        rgb = get(result.images, time_ms, nothing)
        if rgb === nothing
            @warn "Missing requested snapshot." A=result.amplitude T=result.period time_ms=time_ms
            continue
        end
        top, left = _cell_origin(config, result.a_idx, result.t_idx)
        _paste_rgb!(montages[time_ms], rgb, top, left)
    end

    return montages
end

function _append_search_summary(summary_path::AbstractString, config::ParameterSearchConfig, result)
    open(summary_path, "a") do io
        println(io, _summary_line(config, result))
    end
end

function _summary_header()
    return "index,total,a_idx,t_idx,A,T_ms,seed,compute_seconds,elapsed_seconds,tile_top,tile_left"
end

function _summary_line(config::ParameterSearchConfig, result)
    tile_top, tile_left = _cell_origin(config, result.a_idx, result.t_idx)
    return join((
        result.index,
        result.total,
        result.a_idx,
        result.t_idx,
        result.amplitude,
        result.period,
        result.seed === nothing ? "" : result.seed,
        result.compute_seconds,
        result.elapsed_seconds,
        tile_top,
        tile_left,
    ), ",")
end

function _record_search_result!(results::Vector, result)
    results[result.index] = result
    return results
end

function _write_search_summary(summary_path::AbstractString, config::ParameterSearchConfig, results::Vector)
    open(summary_path, "w") do io
        println(io, _summary_header())
        for result in results
            println(io, _summary_line(config, result))
        end
    end
    return summary_path
end

function _format_vector(values)
    return join(_compact_number.(values), ",")
end

function _write_search_metadata(out_path::AbstractString, config::ParameterSearchConfig, total::Integer, interval_ms::Integer)
    path = joinpath(out_path, "config.txt")
    open(path, "w") do io
        println(io, "RSEModel parameter search")
        println(io, "N=", config.N)
        println(io, "Se=", config.Se)
        println(io, "Si=", config.Si)
        println(io, "A_values=", _format_vector(config.A_values))
        println(io, "T_ms_values=", _format_vector(config.period_values))
        println(io, "times_ms=", join(config.times_ms, ","))
        println(io, "simulation_start_time_ms=0")
        println(io, "simulation_end_time_ms=", maximum(config.times_ms))
        println(io, "start_time_ms=", minimum(config.times_ms))
        println(io, "end_time_ms=", maximum(config.times_ms))
        println(io, "interval_ms=", interval_ms)
        println(io, "view=", config.view)
        println(io, "backend=", config.backend)
        println(io, "convolution=", config.convolution)
        println(io, "boundary=periodic")
        println(io, "duty_cycle_percent=", config.duty_cycle_percent === nothing ? "model" : _compact_number(config.duty_cycle_percent))
        println(io, "seed=", config.seed)
        println(io, "seed_mode=", config.seed_mode)
        println(io, "workers=", config.workers)
        println(io, "total_simulations=", total)
        println(io, "summary=summary.csv")
        println(io, "grid_map=grid_map.csv")
        println(io, "snapshot_manifest=snapshot_manifest.csv")
        println(io, "file_pattern=parameter_search_", config.view, "_{time_ms}ms.png")
    end
    return path
end

function _write_grid_map(out_path::AbstractString, config::ParameterSearchConfig, jobs)
    path = joinpath(out_path, "grid_map.csv")
    open(path, "w") do io
        println(io, "index,total,a_idx,t_idx,A,T_ms,tile_top,tile_left")
        for job in jobs
            tile_top, tile_left = _cell_origin(config, job.a_idx, job.t_idx)
            println(io, join((
                job.index,
                job.total,
                job.a_idx,
                job.t_idx,
                job.amplitude,
                job.period,
                tile_top,
                tile_left,
            ), ","))
        end
    end
    return path
end

function _write_snapshot_manifest(out_path::AbstractString, config::ParameterSearchConfig, jobs)
    path = joinpath(out_path, "snapshot_manifest.csv")
    open(path, "w") do io
        println(io, "file,time_ms,index,total,a_idx,t_idx,A,T_ms,tile_top,tile_left")
        for time_ms in config.times_ms
            filename = string("parameter_search_", config.view, "_", lpad(string(time_ms), 5, "0"), "ms.png")
            for job in jobs
                tile_top, tile_left = _cell_origin(config, job.a_idx, job.t_idx)
                println(io, join((
                    filename,
                    time_ms,
                    job.index,
                    job.total,
                    job.a_idx,
                    job.t_idx,
                    job.amplitude,
                    job.period,
                    tile_top,
                    tile_left,
                ), ","))
            end
        end
    end
    return path
end

function _print_search_progress(config::ParameterSearchConfig, result, completed::Integer, total::Integer, search_start)
    total_elapsed = (time_ns() - search_start) / 1e9
    eta = completed == 0 ? 0.0 : (total_elapsed / completed) * (total - completed)
    worker_text = result.worker == 1 ? "" : string(" worker=", result.worker)
    println(
        "[", completed, "/", total, "] ",
        "idx=", result.index, " ",
        "A=", _compact_number(result.amplitude),
        " T=", _compact_number(result.period), " ms ",
        "seed=", result.seed === nothing ? "random" : result.seed,
        worker_text,
        " compute=", round(result.compute_seconds; digits=3), "s ",
        "elapsed=", round(result.elapsed_seconds; digits=3), "s ",
        "eta=", _format_duration(eta),
    )
    flush(stdout)
    return nothing
end

function _run_parameter_search_serial!(montages, results::Vector, progress_summary_path::AbstractString, config::ParameterSearchConfig, jobs, interval_ms::Integer, max_time_ms::Integer, search_start)
    total = length(jobs)
    for (completed, job) in enumerate(jobs)
        result = _run_parameter_search_job(job, config, interval_ms, max_time_ms)
        _apply_search_result!(montages, config, result)
        _record_search_result!(results, result)
        _append_search_summary(progress_summary_path, config, result)
        _print_search_progress(config, result, completed, total, search_start)
    end

    return nothing
end

function _worker_project_flags()
    project = Base.active_project()
    project === nothing && return `--project=@.`
    return `--project=$(project)`
end

function _search_worker_ids()
    return filter(!=(myid()), procs())
end

function _ensure_search_workers!(target_workers::Integer)
    active = _search_worker_ids()
    added = Int[]

    if length(active) < target_workers
        added = addprocs(target_workers - length(active); exeflags=_worker_project_flags())
    end

    worker_ids = _search_worker_ids()[1:target_workers]
    for pid in worker_ids
        remotecall_fetch(Core.eval, pid, Main, :(using RSEModel))
    end

    return worker_ids, added
end

function _run_parameter_search_job_with_channel(args)
    job, config, interval_ms, max_time_ms, result_channel = args
    result = _run_parameter_search_job(job, config, interval_ms, max_time_ms)
    put!(result_channel, result)
    return nothing
end

function _run_parameter_search_parallel!(montages, results::Vector, progress_summary_path::AbstractString, config::ParameterSearchConfig, jobs, interval_ms::Integer, max_time_ms::Integer, search_start)
    target_workers = min(config.workers, length(jobs))
    worker_ids, added_workers = _ensure_search_workers!(target_workers)
    println("Parallel CPU workers: ", target_workers)
    flush(stdout)

    completed = 0
    result_channel = RemoteChannel(() -> Channel{Any}(min(length(jobs), target_workers * 2)))
    work_items = [(job, config, interval_ms, max_time_ms, result_channel) for job in jobs]
    worker_pool = CachingPool(worker_ids)
    scheduler_task = @async pmap(_run_parameter_search_job_with_channel, worker_pool, work_items; batch_size=1)

    try
        while completed < length(jobs)
            if !isready(result_channel)
                if istaskdone(scheduler_task)
                    wait(scheduler_task)
                end
                sleep(0.05)
                continue
            end

            result = take!(result_channel)
            completed += 1
            _apply_search_result!(montages, config, result)
            _record_search_result!(results, result)
            _append_search_summary(progress_summary_path, config, result)
            _print_search_progress(config, result, completed, length(jobs), search_start)
        end

        wait(scheduler_task)
    finally
        isempty(added_workers) || rmprocs(added_workers)
    end

    return nothing
end

function run_parameter_search(config::ParameterSearchConfig=ParameterSearchConfig())
    config = _validate_search_config(config)
    total = length(config.A_values) * length(config.period_values)
    interval_ms = _sample_interval_ms(config.times_ms)
    max_time_ms = maximum(config.times_ms)
    out_path = config.dry_run ? config.out_path : _prepare_search_output_dir(config)

    println("Parameter search output: ", out_path)
    println(
        "Grid: ", length(config.A_values), " A values x ",
        length(config.period_values), " T values = ", total, " simulations",
    )
    println(
        "N=", config.N,
        " Se=", config.Se,
        " Si=", config.Si,
        " backend=", config.backend,
        " conv=", config.convolution,
        " boundary=periodic",
        " view=", config.view,
        " duty=", _duty_text(config),
        " workers=", config.workers,
        " run_end=", max_time_ms,
        " ms",
        " snapshots=", join(string.(config.times_ms), ","),
        " ms",
    )

    if config.dry_run
        println("Dry run only. No simulations were executed.")
        return out_path
    end

    if config.backend == :cpu
        FFTW.set_num_threads(config.fftw_threads)
    end

    jobs = _search_jobs(config)
    montages = Dict(time_ms => _make_montage_canvas(config, time_ms) for time_ms in config.times_ms)
    summary_path = joinpath(out_path, "summary.csv")
    progress_summary_path = joinpath(out_path, "summary_progress.csv")
    open(progress_summary_path, "w") do io
        println(io, _summary_header())
    end
    _write_search_metadata(out_path, config, total, interval_ms)
    _write_grid_map(out_path, config, jobs)
    _write_snapshot_manifest(out_path, config, jobs)

    search_start = time_ns()
    results = Vector{Any}(undef, length(jobs))

    if config.backend == :cpu && config.workers > 1
        _run_parameter_search_parallel!(montages, results, progress_summary_path, config, jobs, interval_ms, max_time_ms, search_start)
    else
        _run_parameter_search_serial!(montages, results, progress_summary_path, config, jobs, interval_ms, max_time_ms, search_start)
    end

    _write_search_summary(summary_path, config, results)
    rm(progress_summary_path; force=true)

    for time_ms in config.times_ms
        filename = joinpath(
            out_path,
            string("parameter_search_", config.view, "_", lpad(string(time_ms), 5, "0"), "ms.png"),
        )
        _write_rgb_png(filename, montages[time_ms])
        println("Wrote ", filename)
    end

    println("Parameter search complete in ", _format_duration((time_ns() - search_start) / 1e9), ".")
    return out_path
end

function _search_parser()
    settings = ArgParseSettings(description="Run an A x T parameter search and save time-slice montage plots.")

    @add_arg_table! settings begin
        "--N"
            help = "Neural field size. Defaults to the requested 81 x 81 grid."
            arg_type = Int
            default = 81
        "--A-range"
            help = "Amplitude sweep as START STOP STEP."
            arg_type = Float64
            nargs = 3
            default = [0.2, 1.5, 0.1]
            dest_name = "A_range"
        "--T-range"
            help = "Period sweep in ms as START STOP STEP."
            arg_type = Float64
            nargs = 3
            default = [10.0, 150.0, 5.0]
            dest_name = "T_range"
        "--times-sec"
            help = "Comma-separated snapshot times in seconds."
            arg_type = String
            default = "5,10,15,20"
            dest_name = "times_sec"
        "--Se"
            help = "Excitatory kernel standard deviation."
            arg_type = Float64
            default = 2.0
        "--Si"
            help = "Inhibitory kernel standard deviation."
            arg_type = Float64
            default = 5.0
        "--backend"
            help = "Simulation backend: cpu or metal."
            arg_type = String
            default = "metal"
        "--gpu"
            help = "Shortcut for --backend metal."
            action = :store_true
        "--conv"
            help = "Convolution backend: auto, fft, or separable. Auto uses separable on Metal and FFT on CPU."
            arg_type = String
            default = "auto"
        "--kernel-cutoff"
            help = "Gaussian cutoff in sigma units for Metal separable convolution."
            arg_type = Float64
            default = 3.0
            dest_name = "kernel_cutoff"
        "--duty-cycle"
            help = "Stimulus duty cycle percentage. Defaults to 50."
            arg_type = Float64
            default = 50.0
            dest_name = "duty_cycle_percent"
        "--view"
            help = "Montage view: retinal or cortical."
            arg_type = String
            default = "cortical"
        "--cmap"
            help = "Colormap name. Supports plasma, nipy_spectral, and grayscale."
            arg_type = String
            default = "plasma"
        "--seed"
            help = "Base random seed. Use -1 for the default non-fixed RNG."
            arg_type = Int
            default = 42
        "--seed-mode"
            help = "same or increment. same reuses the seed for every A/T cell."
            arg_type = String
            default = "same"
            dest_name = "seed_mode"
        "--out-path"
            help = "Output directory."
            arg_type = String
            default = joinpath("outputs", "parameter_search")
            dest_name = "out_path"
        "--overwrite"
            help = "Write into --out-path directly instead of creating a unique suffixed directory."
            action = :store_true
        "--fftw-threads"
            help = "Number of FFTW threads for CPU searches."
            arg_type = Int
            default = 1
            dest_name = "fftw_threads"
        "--gpu-threads"
            help = "Metal kernel threadgroup size."
            arg_type = Int
            default = 256
            dest_name = "gpu_threads"
        "--workers"
            help = "CPU worker processes for parallel parameter sweeps. Metal runs serially."
            arg_type = Int
            default = 1
        "--dry-run"
            help = "Print the planned search without running simulations."
            action = :store_true
            dest_name = "dry_run"
    end

    return settings
end

function parameter_search_main(argv=ARGS)
    args = parse_args(argv, _search_parser())
    backend = args["gpu"] ? :metal : Symbol(lowercase(args["backend"]))
    config = ParameterSearchConfig(
        N=args["N"],
        A_values=_range_values(args["A_range"][1], args["A_range"][2], args["A_range"][3]),
        period_values=_range_values(args["T_range"][1], args["T_range"][2], args["T_range"][3]),
        times_ms=_parse_times_ms(args["times_sec"]),
        Se=args["Se"],
        Si=args["Si"],
        backend=backend,
        convolution=Symbol(lowercase(args["conv"])),
        kernel_cutoff=args["kernel_cutoff"],
        duty_cycle_percent=args["duty_cycle_percent"],
        seed=args["seed"],
        seed_mode=Symbol(lowercase(args["seed_mode"])),
        view=Symbol(lowercase(args["view"])),
        cmap=args["cmap"],
        out_path=args["out_path"],
        overwrite=args["overwrite"],
        fftw_threads=args["fftw_threads"],
        gpu_threads=args["gpu_threads"],
        workers=args["workers"],
        dry_run=args["dry_run"],
    )
    run_parameter_search(config)
    return nothing
end
