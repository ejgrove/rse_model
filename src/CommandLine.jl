using ArgParse
using FFTW
using Random

function odd_positive_int(value)
    n = try
        parse(Int, string(value))
    catch exc
        throw(ArgumentError("Invalid integer: $(repr(value))"))
    end

    n > 0 || throw(ArgumentError("N must be a positive integer"))
    return div(n, 2) * 2 + 1
end

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

function next_fast_odd_size(value)
    n = odd_positive_int(value)
    while !is_fast_fft_size(n)
        n += 2
    end
    return n
end

function _format_number(value)
    text = string(value)
    return replace(text, "." => "_")
end

function _format_rounded(value)
    rounded = round(value)
    return _format_number(rounded == round(Int, value) ? round(Int, value) : rounded)
end

function _uniform_between(low, high)
    return rand() * (high - low) + low
end

function _validate_images(images)
    images === nothing && return nothing
    images in ("cortical", "retinal", "both") && return images
    throw(ArgumentError("--images must be one of: cortical, retinal, both"))
end

_has_values(value) = value !== nothing && !(value isa AbstractVector && isempty(value))

function _validate_backend(value)
    key = lowercase(value)
    key in ("cpu", "metal") && return key
    throw(ArgumentError("--backend must be cpu or metal"))
end

function _validate_convolution(value, backend, boundary_x::Symbol=:periodic, boundary_y::Symbol=:periodic)
    key = lowercase(value)
    key in ("auto", "fft", "separable") || throw(ArgumentError("--conv must be auto, fft, or separable"))
    if key == "auto"
        return _default_convolution(Symbol(backend), boundary_x, boundary_y)
    elseif key == "separable" && backend != "metal"
        throw(ArgumentError("--conv separable is currently implemented for the Metal backend only"))
    else
        return Symbol(key)
    end
end

function _validate_boundary_cli(value)
    key = lowercase(value)
    key in ("periodic", "edge", "zero", "partial_reflect") && return Symbol(key)
    key in ("partial_reflective", "partially_reflective", "reflect") && return :partial_reflect
    throw(ArgumentError("--boundary must be periodic, edge, zero, or partial_reflect"))
end

function _fft_plan_flags(value)
    key = lowercase(value)
    if key == "estimate"
        return FFTW.ESTIMATE
    elseif key == "measure"
        return FFTW.MEASURE
    elseif key == "patient"
        return FFTW.PATIENT
    else
        throw(ArgumentError("--fft-plan must be one of: estimate, measure, patient"))
    end
end

function _build_parser()
    settings = ArgParseSettings(description="Run RSE model simulations.")

    @add_arg_table! settings begin
        "--N"
            help = "Neural field size. Values are coerced to the next odd positive integer."
            arg_type = Int
            default = 101
        "--fast-n"
            help = "Increase N to the next odd FFT-friendly size with only small prime factors."
            action = :store_true
            dest_name = "fast_n"
        "--backend"
            help = "Simulation backend: cpu or metal."
            arg_type = String
            default = "cpu"
        "--gpu"
            help = "Shortcut for --backend metal."
            action = :store_true
        "--conv"
            help = "Convolution backend: auto, fft, or separable. Auto uses FFT for periodic CPU/Metal runs."
            arg_type = String
            default = "auto"
        "--kernel-cutoff"
            help = "Gaussian cutoff in sigma units for Metal separable convolution."
            arg_type = Float64
            default = 3.0
            dest_name = "kernel_cutoff"
        "--boundary"
            help = "Convolution boundary mode for both axes: periodic, edge, zero, or partial_reflect. Axis-specific flags override this."
            arg_type = String
            default = nothing
        "--boundary-x"
            help = "Horizontal/left-right convolution boundary mode: periodic, edge, zero, or partial_reflect."
            arg_type = String
            default = nothing
            dest_name = "boundary_x"
        "--boundary-y"
            help = "Vertical/top-bottom convolution boundary mode: periodic, edge, zero, or partial_reflect."
            arg_type = String
            default = nothing
            dest_name = "boundary_y"
        "--partial-reflect-strength"
            help = "Reflected boundary contribution for partial_reflect mode, from 0 to 1."
            arg_type = Float64
            default = 0.5
            dest_name = "partial_reflect_strength"
        "--duty-cycle"
            help = "Stimulus duty cycle percentage. A value of 50 uses threshold 0. Defaults to the ModelParams threshold V."
            arg_type = Float64
            default = nothing
            dest_name = "duty_cycle_percent"
        "--A"
            help = "Stimulus amplitude."
            arg_type = Float64
            default = 0.7
        "--T"
            help = "Stimulus period in ms."
            arg_type = Float64
            default = 115.0
        "--Se"
            help = "Excitatory kernel standard deviation."
            arg_type = Float64
            default = 2.0
        "--Si"
            help = "Inhibitory kernel standard deviation."
            arg_type = Float64
            default = 5.0

        "--seed"
            help = "Random seed."
            arg_type = Int
            default = nothing
        "--start"
            help = "Time in ms to start saving outputs."
            arg_type = Int
            default = 0
        "--end"
            help = "Time in ms to end saving outputs."
            arg_type = Int
            default = 2000
        "--interval"
            help = "Time interval in ms for saving outputs."
            arg_type = Int
            default = 1000
        "--num-sims"
            help = "Number of simulations to run."
            arg_type = Int
            default = 1
            dest_name = "num_sims"
        "--rand-freq"
            help = "Randomize T in [T - rand_freq, T + rand_freq]."
            arg_type = Int
            default = nothing
            dest_name = "rand_freq"
        "--rand-size"
            help = "Randomize N in the inclusive integer range [low, high]."
            arg_type = Int
            nargs = 2
            default = nothing
            dest_name = "rand_size"

        "--plot"
            help = "Save a compact cortical/retinal summary plot."
            action = :store_true
        "--images"
            help = "Save images of the simulation: cortical, retinal, or both."
            arg_type = String
            default = nothing
        "--contours"
            help = "Number of contours for compatibility with the Python CLI."
            arg_type = Int
            default = 50
        "--cmap"
            help = "Colormap name. Supports plasma, nipy_spectral, and grayscale."
            arg_type = String
            default = "plasma"
        "--dpi"
            help = "Image DPI metadata placeholder for CLI compatibility."
            arg_type = Int
            default = 100
        "--out-path"
            help = "Output directory."
            arg_type = String
            default = "outputs"
            dest_name = "out_path"

        "--gif"
            help = "Save a retinal-view GIF."
            action = :store_true
        "--fps"
            help = "Frames per second for GIF sampling."
            arg_type = Int
            default = 50
        "--label"
            help = "Write label metadata alongside generated images."
            action = :store_true
        "--fft-plan"
            help = "FFTW planning mode: estimate, measure, or patient."
            arg_type = String
            default = "measure"
            dest_name = "fft_plan"
        "--fftw-threads"
            help = "Number of FFTW threads. For small grids, 1 is usually fastest."
            arg_type = Int
            default = 1
            dest_name = "fftw_threads"
        "--gpu-threads"
            help = "Metal kernel threadgroup size for the fused Euler update."
            arg_type = Int
            default = 256
            dest_name = "gpu_threads"
    end

    return settings
end

function _prepare_output_dir(args)
    if !(args["gif"] || args["images"] !== nothing || args["plot"])
        return nothing
    end

    T_str = if args["rand_freq"] !== nothing
        low = round(args["T"]) - args["rand_freq"]
        high = round(args["T"]) + args["rand_freq"]
        string(_format_number(low), "to", _format_number(high))
    else
        _format_rounded(args["T"])
    end

    N_str = if _has_values(args["rand_size"])
        low, high = args["rand_size"]
        string(_format_number(round(low)), "to", _format_number(round(high)))
    else
        string(args["N"])
    end

    duty_suffix = args["duty_cycle_percent"] === nothing ? "" :
        string("_Duty", _format_number(args["duty_cycle_percent"]))

    file_suffix = string(
        "simulation_A", _format_number(args["A"]),
        "_T", T_str,
        "_Se", _format_rounded(args["Se"]),
        "_Si", _format_rounded(args["Si"]),
        "_N", N_str,
        duty_suffix,
    )

    out_path = ensure_unique_path(joinpath(args["out_path"], file_suffix))
    mkpath(out_path)
    println("Outputs will be saved to: ", out_path)
    return out_path
end

function main(argv=ARGS)
    cli_timer_start = time_ns()
    args = parse_args(argv, _build_parser())
    args["N"] = odd_positive_int(args["N"])
    if args["fast_n"]
        requested_N = args["N"]
        args["N"] = next_fast_odd_size(requested_N)
        if args["N"] != requested_N
            println("Adjusted N from $(requested_N) to FFT-friendly size $(args["N"]).")
        end
    end
    args["images"] = _validate_images(args["images"])
    args["backend"] = args["gpu"] ? "metal" : _validate_backend(args["backend"])
    boundary_base = args["boundary"] === nothing ? :periodic : _validate_boundary_cli(args["boundary"])
    boundary_x = args["boundary_x"] === nothing ? boundary_base : _validate_boundary_cli(args["boundary_x"])
    boundary_y = args["boundary_y"] === nothing ? boundary_base : _validate_boundary_cli(args["boundary_y"])
    convolution = _validate_convolution(args["conv"], args["backend"], boundary_x, boundary_y)

    if args["interval"] <= 0
        throw(ArgumentError("--interval must be positive"))
    end
    if args["end"] < args["start"]
        throw(ArgumentError("--end must be greater than or equal to --start"))
    end
    if args["fps"] <= 0
        throw(ArgumentError("--fps must be positive"))
    end
    if args["fftw_threads"] <= 0
        throw(ArgumentError("--fftw-threads must be positive"))
    end
    if args["gpu_threads"] <= 0
        throw(ArgumentError("--gpu-threads must be positive"))
    end
    if args["kernel_cutoff"] <= 0
        throw(ArgumentError("--kernel-cutoff must be positive"))
    end
    if args["duty_cycle_percent"] !== nothing && !(0 <= args["duty_cycle_percent"] <= 100)
        throw(ArgumentError("--duty-cycle must be between 0 and 100"))
    end

    if args["backend"] == "cpu"
        FFTW.set_num_threads(args["fftw_threads"])
    end
    fft_flags = _fft_plan_flags(args["fft_plan"])
    out_path = _prepare_output_dir(args)
    seed = args["seed"]

    for _sim in 1:args["num_sims"]
        period = if args["rand_freq"] !== nothing
            _uniform_between(args["T"] - args["rand_freq"], args["T"] + args["rand_freq"])
        else
            args["T"]
        end

        N = if _has_values(args["rand_size"])
            low, high = args["rand_size"]
            randomized_N = args["fast_n"] ? next_fast_odd_size(rand(low:high)) : odd_positive_int(rand(low:high))
            println(randomized_N)
            randomized_N
        else
            args["N"]
        end

        println(N)

        params = ModelParams()
        data = run_simulation(
            N=N,
            A=args["A"],
            T=period,
            Se=args["Se"],
            Si=args["Si"],
            start_time=args["start"],
            end_time=args["end"],
            seed=seed,
            plot=args["plot"],
            gif=args["gif"],
            interval=args["interval"],
            p=params,
            fps=args["fps"],
            fft_flags=fft_flags,
            backend=Symbol(args["backend"]),
            gpu_threads=args["gpu_threads"],
            convolution=convolution,
            kernel_cutoff=args["kernel_cutoff"],
            boundary_x=boundary_x,
            boundary_y=boundary_y,
            partial_reflect_strength=args["partial_reflect_strength"],
            duty_cycle_percent=args["duty_cycle_percent"],
        )
        println(
            "Simulation compute duration ($(args["backend"])): ",
            round(data.compute_seconds; digits=4),
            " s",
        )

        if args["gif"]
            make_gif(
                data.gif;
                label=args["label"],
                out_path=out_path,
                fps=args["fps"],
                dpi=args["dpi"],
                N=N,
                A=args["A"],
                T=period,
                Se=args["Se"],
                Si=args["Si"],
                contours=args["contours"],
                cmap=args["cmap"],
                p=params,
            )
        end

        if args["images"] !== nothing || args["plot"]
            println("Generating images and/or plots...")
            for snapshot in data.images
                if args["plot"]
                    plot_dir = joinpath(out_path, "plots")
                    mkpath(plot_dir)
                    plot_file = joinpath(plot_dir, "plot_$(round(Int, snapshot.t))ms.png")
                    make_plot(
                        snapshot;
                        out_file=plot_file,
                        N=N,
                        A=args["A"],
                        T=period,
                        Se=args["Se"],
                        Si=args["Si"],
                        contours=args["contours"],
                        cmap=args["cmap"],
                        p=params,
                    )
                end

                if args["images"] !== nothing
                    image_dir = joinpath(out_path, "images")
                    mkpath(image_dir)
                    println(snapshot.t)
                    make_images(
                        snapshot;
                        images=args["images"],
                        label=args["label"],
                        out_path=image_dir,
                        dpi=args["dpi"],
                        N=N,
                        A=args["A"],
                        T=period,
                        Se=args["Se"],
                        Si=args["Si"],
                        contours=args["contours"],
                        cmap=args["cmap"],
                        p=params,
                    )
                end
            end
        end

        if seed !== nothing && seed != 0
            seed += 1
        end
    end

    println("Total command duration: ", round((time_ns() - cli_timer_start) / 1e9; digits=4), " s")
    return nothing
end
