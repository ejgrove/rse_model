#!/usr/bin/env julia

using ArgParse
using FFTW
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

function _settings()
    settings = ArgParseSettings(description="Benchmark the Julia RSE CPU simulation loop.")

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
    end

    return settings
end

function main(argv=ARGS)
    args = parse_args(argv, _settings())
    sizes = _parse_sizes(args["sizes"])
    args["passes"] > 0 || throw(ArgumentError("--passes must be positive"))
    args["fftw_threads"] > 0 || throw(ArgumentError("--fftw-threads must be positive"))

    FFTW.set_num_threads(args["fftw_threads"])
    fft_flags = _fft_plan_flags(args["fft_plan"])
    params = ModelParams()
    steps = round(Int, args["end"] / params.dt)

    println("backend=cpu fft_plan=$(args["fft_plan"]) fftw_threads=$(args["fftw_threads"]) steps=$(steps)")
    for requested_N in sizes
        N = args["fast_n"] ? next_fast_odd_size(requested_N) : odd_positive_int(requested_N)
        for pass in 1:args["passes"]
            GC.gc()
            elapsed = @elapsed run_simulation(
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
            )
            ms_per_step = 1000 * elapsed / steps
            realtime = params.dt / ms_per_step
            println(
                "requested_N=$(requested_N) N=$(N) pass=$(pass) elapsed=$(round(elapsed; digits=4))s ",
                "ms_per_step=$(round(ms_per_step; digits=4)) realtime_x=$(round(realtime; digits=3))",
            )
        end
    end

    return nothing
end

main()
