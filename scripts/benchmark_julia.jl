#!/usr/bin/env julia

using ArgParse
using FFTW
using Metal
using RSEModel

function _parse_sizes(text)
    return [parse(Int, strip(part)) for part in split(text, ",") if !isempty(strip(part))]
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

function _convolution(value, backend)
    key = lowercase(value)
    key in ("auto", "fft", "separable") || throw(ArgumentError("--conv must be auto, fft, or separable"))
    if key == "auto"
        return backend == "metal" ? :separable : :fft
    elseif key == "separable" && backend != "metal"
        throw(ArgumentError("--conv separable is currently implemented for the Metal backend only"))
    else
        return Symbol(key)
    end
end

function _settings()
    settings = ArgParseSettings(description="Benchmark the Julia RSE simulation loop.")

    @add_arg_table! settings begin
        "--sizes"
            help = "Comma-separated N values to benchmark."
            arg_type = String
            default = "101,105,135,201,225"
        "--end"
            help = "Simulation duration in ms."
            arg_type = Float64
            default = 100.0
        "--passes"
            help = "Benchmark passes per size."
            arg_type = Int
            default = 2
        "--backend"
            help = "Simulation backend: cpu or metal."
            arg_type = String
            default = "cpu"
        "--gpu"
            help = "Shortcut for --backend metal."
            action = :store_true
        "--conv"
            help = "Convolution backend: auto, fft, or separable. Auto uses separable on Metal."
            arg_type = String
            default = "auto"
        "--kernel-cutoff"
            help = "Gaussian cutoff in sigma units for Metal separable convolution."
            arg_type = Float64
            default = 2.0
            dest_name = "kernel_cutoff"
        "--fft-plan"
            help = "FFTW planning mode: estimate, measure, or patient."
            arg_type = String
            default = "measure"
            dest_name = "fft_plan"
        "--fftw-threads"
            help = "Number of FFTW threads."
            arg_type = Int
            default = 1
            dest_name = "fftw_threads"
        "--fast-n"
            help = "Map each requested N to the next odd FFT-friendly size."
            action = :store_true
            dest_name = "fast_n"
        "--gpu-threads"
            help = "Metal kernel threadgroup size for the fused Euler update."
            arg_type = Int
            default = 256
            dest_name = "gpu_threads"
    end

    return settings
end

function main(argv=ARGS)
    args = parse_args(argv, _settings())
    sizes = _parse_sizes(args["sizes"])
    backend = args["gpu"] ? "metal" : lowercase(args["backend"])
    backend in ("cpu", "metal") || throw(ArgumentError("--backend must be cpu or metal"))
    convolution = _convolution(args["conv"], backend)
    args["passes"] > 0 || throw(ArgumentError("--passes must be positive"))
    args["fftw_threads"] > 0 || throw(ArgumentError("--fftw-threads must be positive"))
    args["gpu_threads"] > 0 || throw(ArgumentError("--gpu-threads must be positive"))
    args["kernel_cutoff"] > 0 || throw(ArgumentError("--kernel-cutoff must be positive"))

    if backend == "cpu"
        FFTW.set_num_threads(args["fftw_threads"])
    elseif !Metal.functional()
        throw(ErrorException("Metal.jl is not functional on this machine."))
    end

    fft_flags = _fft_plan_flags(args["fft_plan"])
    params = ModelParams()
    steps = round(Int, args["end"] / params.dt)

    println(
        "backend=$(backend) fft_plan=$(args["fft_plan"]) fftw_threads=$(args["fftw_threads"]) ",
        "gpu_threads=$(args["gpu_threads"]) conv=$(convolution) kernel_cutoff=$(args["kernel_cutoff"]) steps=$(steps)",
    )
    for requested_N in sizes
        N = args["fast_n"] ? next_fast_odd_size(requested_N) : odd_positive_int(requested_N)
        for pass in 1:args["passes"]
            GC.gc()
            data = nothing
            elapsed = @elapsed data = run_simulation(
                N=N,
                A=0.7,
                T=115.0,
                Se=2.0,
                Si=5.0,
                start_time=0,
                end_time=round(Int, args["end"]),
                seed=42,
                plot=false,
                gif=false,
                interval=1000,
                p=params,
                fft_flags=fft_flags,
                backend=Symbol(backend),
                gpu_threads=args["gpu_threads"],
                convolution=convolution,
                kernel_cutoff=args["kernel_cutoff"],
            )
            compute_seconds = data.compute_seconds
            ms_per_step = 1000 * compute_seconds / steps
            realtime = params.dt / ms_per_step
            println(
                "requested_N=$(requested_N) N=$(N) pass=$(pass) elapsed=$(round(elapsed; digits=4))s ",
                "compute=$(round(compute_seconds; digits=4))s ",
                "ms_per_step=$(round(ms_per_step; digits=4)) realtime_x=$(round(realtime; digits=3))",
            )
        end
    end

    return nothing
end

main()
