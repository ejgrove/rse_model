#!/usr/bin/env julia

using ArgParse
using Printf
using RSEModel

function _parse_csv_floats(text)
    values = Float64[]
    for part in split(text, ",")
        stripped = strip(part)
        isempty(stripped) && continue
        push!(values, parse(Float64, stripped))
    end
    isempty(values) && throw(ArgumentError("Expected at least one comma-separated value."))
    return values
end

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

function _linspace_values(start, stop, count::Integer)
    count > 1 || return [Float64(start)]
    return [round(Float64(start) + (Float64(stop) - Float64(start)) * (idx - 1) / (count - 1); digits=10) for idx in 1:count]
end

function _parse_times_ms(text)
    times = [round(Int, value * 1000) for value in _parse_csv_floats(text)]
    all(>(0), times) || throw(ArgumentError("All sample times must be positive seconds."))
    return sort(unique(times))
end

function _label_number(value)
    text = @sprintf("%.4f", Float64(value))
    text = replace(text, r"0+$" => "")
    text = replace(text, r"\.$" => "")
    return replace(text, "." => "p", "-" => "m")
end

function _parser()
    settings = ArgParseSettings(description="Run an Se x Si kernel sweep around the A x T parameter search.")

    @add_arg_table! settings begin
        "--N"
            help = "Neural field size."
            arg_type = Int
            default = 81
        "--Se-values"
            help = "Comma-separated excitatory kernel widths."
            arg_type = String
            default = "1.5,1.75,2,2.25,2.5"
            dest_name = "Se_values"
        "--Si-factor-start"
            help = "Start factor for Si = factor * Se."
            arg_type = Float64
            default = 2.0
            dest_name = "Si_factor_start"
        "--Si-factor-stop"
            help = "Stop factor for Si = factor * Se."
            arg_type = Float64
            default = 3.0
            dest_name = "Si_factor_stop"
        "--Si-count"
            help = "Number of Si factor values, including endpoints."
            arg_type = Int
            default = 5
            dest_name = "Si_count"
        "--A-range"
            help = "Amplitude sweep as START STOP STEP."
            arg_type = Float64
            nargs = 3
            default = [0.25, 1.25, 0.25]
            dest_name = "A_range"
        "--T-range"
            help = "Period sweep in ms as START STOP STEP."
            arg_type = Float64
            nargs = 3
            default = [35.0, 125.0, 10.0]
            dest_name = "T_range"
        "--times-sec"
            help = "Comma-separated snapshot times in seconds."
            arg_type = String
            default = "5"
            dest_name = "times_sec"
        "--backend"
            help = "Simulation backend: cpu or metal."
            arg_type = String
            default = "metal"
        "--gpu"
            help = "Shortcut for --backend metal."
            action = :store_true
        "--conv"
            help = "Convolution backend: auto, fft, or separable."
            arg_type = String
            default = "separable"
        "--kernel-cutoff"
            help = "Gaussian cutoff in sigma units for Metal separable convolution."
            arg_type = Float64
            default = 4.0
            dest_name = "kernel_cutoff"
        "--duty-cycle"
            help = "Stimulus duty cycle percentage."
            arg_type = Float64
            default = 50.0
            dest_name = "duty_cycle_percent"
        "--view"
            help = "Montage view: retinal or cortical."
            arg_type = String
            default = "cortical"
        "--cmap"
            help = "Colormap name."
            arg_type = String
            default = "plasma"
        "--seed"
            help = "Base random seed. Use -1 for the default non-fixed RNG."
            arg_type = Int
            default = 42
        "--seed-mode"
            help = "same or increment."
            arg_type = String
            default = "same"
            dest_name = "seed_mode"
        "--workers"
            help = "CPU worker processes. Metal runs serially."
            arg_type = Int
            default = 1
        "--out-path"
            help = "Root output directory."
            arg_type = String
            default = joinpath("outputs", "kernel_parameter_search_N81_metal_sep_t5")
            dest_name = "out_path"
        "--overwrite"
            help = "Write into --out-path directly instead of creating a unique suffixed root directory."
            action = :store_true
        "--dry-run"
            help = "Print the planned kernel sweep without running simulations."
            action = :store_true
            dest_name = "dry_run"
    end

    return settings
end

function _kernel_jobs(se_values, si_factors)
    jobs = NamedTuple[]
    index = 0
    for Se in se_values
        for factor in si_factors
            index += 1
            push!(jobs, (index=index, Se=Se, Si=Se * factor, Si_factor=factor))
        end
    end
    return jobs
end

function _write_kernel_grid(root_out, jobs, sims_per_kernel)
    mkpath(root_out)
    path = joinpath(root_out, "kernel_grid.csv")
    open(path, "w") do io
        println(io, "index,Se,Si,Si_factor,simulations,out_path")
        for job in jobs
            subdir = string("Se", _label_number(job.Se), "_Si", _label_number(job.Si))
            println(io, join((job.index, job.Se, job.Si, job.Si_factor, sims_per_kernel, joinpath(root_out, subdir)), ","))
        end
    end
    return path
end

function main(argv=ARGS)
    args = parse_args(argv, _parser())
    se_values = _parse_csv_floats(args["Se_values"])
    si_factors = _linspace_values(args["Si_factor_start"], args["Si_factor_stop"], args["Si_count"])
    a_values = _range_values(args["A_range"][1], args["A_range"][2], args["A_range"][3])
    period_values = _range_values(args["T_range"][1], args["T_range"][2], args["T_range"][3])
    times_ms = _parse_times_ms(args["times_sec"])
    backend = args["gpu"] ? :metal : Symbol(lowercase(args["backend"]))
    root_out = args["overwrite"] ? args["out_path"] : ensure_unique_path(args["out_path"])
    jobs = _kernel_jobs(se_values, si_factors)
    sims_per_kernel = length(a_values) * length(period_values)
    total_sims = length(jobs) * sims_per_kernel

    println("Kernel parameter search root: ", root_out)
    println(
        "Kernel grid: ", length(se_values), " Se values x ", length(si_factors),
        " Si factors = ", length(jobs), " kernel pairs",
    )
    println(
        "A x T per kernel: ", length(a_values), " x ", length(period_values),
        " = ", sims_per_kernel, " simulations",
    )
    println("Total simulations: ", total_sims)
    println("Se values: ", join(se_values, ", "))
    println("Si factors: ", join(si_factors, ", "), " => Si = factor * Se")

    if args["dry_run"]
        for job in jobs
            subdir = string("Se", _label_number(job.Se), "_Si", _label_number(job.Si))
            println(
                "[", job.index, "/", length(jobs), "] ",
                "Se=", job.Se, " Si=", job.Si, " out=", joinpath(root_out, subdir),
            )
        end
        println("Dry run only. No simulations were executed.")
        return nothing
    end

    _write_kernel_grid(root_out, jobs, sims_per_kernel)
    search_start = time_ns()

    for job in jobs
        subdir = string("Se", _label_number(job.Se), "_Si", _label_number(job.Si))
        out_path = joinpath(root_out, subdir)
        println()
        println(
            "Kernel pair [", job.index, "/", length(jobs), "] ",
            "Se=", job.Se, " Si=", job.Si, " Si_factor=", job.Si_factor,
        )

        run_parameter_search(ParameterSearchConfig(
            N=args["N"],
            A_values=a_values,
            period_values=period_values,
            times_ms=times_ms,
            Se=job.Se,
            Si=job.Si,
            backend=backend,
            convolution=Symbol(lowercase(args["conv"])),
            kernel_cutoff=args["kernel_cutoff"],
            duty_cycle_percent=args["duty_cycle_percent"],
            seed=args["seed"],
            seed_mode=Symbol(lowercase(args["seed_mode"])),
            view=Symbol(lowercase(args["view"])),
            cmap=args["cmap"],
            out_path=out_path,
            overwrite=true,
            workers=args["workers"],
        ))
    end

    elapsed = round((time_ns() - search_start) / 1e9; digits=1)
    println()
    println("Kernel parameter search complete in ", elapsed, "s.")
    println("Root output: ", root_out)
    return nothing
end

main()
