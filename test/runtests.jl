using Test
using FFTW
using HTTP
using Metal
using RSEModel

@testset "kernel shape" begin
    kernel = generate_gaussian_kernel(2.0, 5)
    @test size(kernel) == (5, 5)
    @test kernel[3, 3] == maximum(kernel)
    kernel_1d = generate_gaussian_kernel_1d(2.0, 2)
    @test length(kernel_1d) == 5
    @test kernel_1d[3] == maximum(kernel_1d)
end

@testset "odd positive int" begin
    @test odd_positive_int(100) == 101
    @test odd_positive_int(101) == 101
    @test_throws ArgumentError odd_positive_int(0)
end

@testset "fast FFT sizes" begin
    @test is_fast_fft_size(105)
    @test !is_fast_fft_size(101)
    @test next_fast_odd_size(101) == 105
    @test next_fast_odd_size(201) == 225
end

@testset "model params constructors" begin
    @test ModelParams() isa ModelParams{Float64}
    @test ModelParams{Float32}().dt isa Float32
end

@testset "stimulus duty cycle" begin
    default_duty = duty_cycle_percent_from_threshold(ModelParams{Float32}().V)
    @test 20 < default_duty < 21
    @test stimulus_threshold_from_duty_cycle_percent(default_duty) ≈ ModelParams{Float32}().V atol=1e-6
    @test stimulus_threshold_from_duty_cycle_percent(50.0) ≈ 0.0 atol=1e-12
    @test RSEModel.strobe_stimulus(0.0f0, 0.7f0, 115.0f0, ModelParams{Float32}(), 50.0f0) == 0.0f0
    @test RSEModel.strobe_stimulus(30.0f0, 0.7f0, 115.0f0, ModelParams{Float32}(), 50.0f0) == 0.7f0
end

@testset "midline coupling" begin
    left_e = fill(1.0f0, 8, 4)
    left_i = fill(2.0f0, 8, 4)
    right_e = fill(3.0f0, 8, 4)
    right_i = fill(4.0f0, 8, 4)
    RSEModel._apply_midline_coupling!(left_e, left_i, right_e, right_i, 0.25f0, 4)

    @test left_e[1, 1] == 1.5f0
    @test right_e[2, 1] == 2.5f0
    @test left_i[8, 1] == 2.5f0
    @test right_i[7, 1] == 3.5f0
    @test left_e[4, 1] == 1.0f0
    @test right_i[4, 1] == 4.0f0
end

@testset "short simulation" begin
    data = run_simulation(
        N=25,
        A=0.7,
        T=115,
        Se=2.0,
        Si=5.0,
        start_time=0,
        end_time=1,
        seed=42,
        plot=true,
        gif=true,
        interval=1,
        p=ModelParams{Float32}(),
        fps=50,
    )
    @test !isempty(data.images)
    @test !isempty(data.gif)
    @test size(first(data.images).cortical_activity) == (25, 25)
    @test isfinite(data.compute_seconds)
    @test data.compute_seconds > 0
end

@testset "short metal simulation" begin
    if Metal.functional()
        data = run_simulation(
            N=25,
            A=0.7,
            T=115,
            Se=2.0,
            Si=5.0,
            start_time=0,
            end_time=1,
            seed=42,
            plot=false,
            gif=false,
            interval=1,
            p=ModelParams{Float32}(),
            backend=:metal,
            gpu_threads=128,
        )
        @test !isempty(data.images)
        @test isempty(data.gif)
        @test size(first(data.images).cortical_activity) == (25, 25)
        @test isfinite(data.compute_seconds)
        @test data.compute_seconds > 0
    else
        @test_skip "Metal.jl is not functional on this machine."
    end
end

@testset "paired metal separable convolution" begin
    if Metal.functional()
        Ue = Metal.rand(Float32, 25, 25)
        Ui = Metal.rand(Float32, 25, 25)
        out_e_separate = similar(Ue)
        out_i_separate = similar(Ui)
        out_e_pair = similar(Ue)
        out_i_pair = similar(Ui)

        ce_separate = RSEModel.MetalSeparableConvolver(2.0, Ue; cutoff=3.0)
        ci_separate = RSEModel.MetalSeparableConvolver(5.0, Ui; cutoff=3.0)
        ce_pair = RSEModel.MetalSeparableConvolver(2.0, Ue; cutoff=3.0)
        ci_pair = RSEModel.MetalSeparableConvolver(5.0, Ui; cutoff=3.0)

        for (boundary_x, boundary_y) in ((:periodic, :periodic), (:edge, :zero), (:zero, :edge))
            RSEModel.separable_convolution!(
                out_e_separate,
                ce_separate,
                Ue;
                gpu_threads=128,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            )
            RSEModel.separable_convolution!(
                out_i_separate,
                ci_separate,
                Ui;
                gpu_threads=128,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            )
            RSEModel.separable_convolution_pair!(
                out_e_pair,
                out_i_pair,
                ce_pair,
                ci_pair,
                Ue,
                Ui;
                gpu_threads=128,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            )
            Metal.synchronize()

            @test Array(out_e_pair) ≈ Array(out_e_separate) rtol=1e-6 atol=1e-6
            @test Array(out_i_pair) ≈ Array(out_i_separate) rtol=1e-6 atol=1e-6
        end
    else
        @test_skip "Metal.jl is not functional on this machine."
    end
end

@testset "metal fft convolution matches cpu fft" begin
    if Metal.functional()
        N = 17
        U = reshape(Float32.(1:(N * N)), N, N) ./ Float32(N * N)
        K = generate_gaussian_kernel(2.0, N; dtype=Float32)
        cpu_out = similar(U)
        RSEModel.fft_convolution!(cpu_out, RSEModel.FFTConvolver(K, U; flags=FFTW.ESTIMATE), U)

        gpu_U = Metal.MtlArray(U)
        gpu_out = similar(gpu_U)
        RSEModel.fft_convolution!(gpu_out, RSEModel.MetalFFTConvolver(K, gpu_U), gpu_U)
        Metal.synchronize()

        @test Array(gpu_out) ≈ cpu_out rtol=1e-5 atol=1e-5
    else
        @test_skip "Metal.jl is not functional on this machine."
    end
end

@testset "retinal transform" begin
    img = reshape(collect(Float32, 1:25), 5, 5)
    ret = retinal_transform(img)
    @test size(ret) == size(img)
end

@testset "parameter search smoke" begin
    out_path = mktempdir()
    result_path = run_parameter_search(ParameterSearchConfig(
        N=9,
        A_values=[0.2],
        period_values=[10.0],
        times_ms=[10],
        Se=2.0,
        Si=5.0,
        backend=:cpu,
        convolution=:fft,
        seed=1,
        view=:cortical,
        out_path=out_path,
        overwrite=true,
    ))
    @test result_path == out_path
    @test isfile(joinpath(out_path, "summary.csv"))
    @test isfile(joinpath(out_path, "config.txt"))
    @test isfile(joinpath(out_path, "grid_map.csv"))
    @test isfile(joinpath(out_path, "snapshot_manifest.csv"))
    @test !isfile(joinpath(out_path, "summary_progress.csv"))
    @test isfile(joinpath(out_path, "parameter_search_cortical_00010ms.png"))
    @test occursin("simulation_end_time_ms=10", read(joinpath(out_path, "config.txt"), String))
    @test occursin("stimulus_threshold=0", read(joinpath(out_path, "config.txt"), String))
    @test occursin("index,total,a_idx,t_idx,A,T_ms", first(readlines(joinpath(out_path, "summary.csv"))))
end

@testset "live applet config" begin
    periodic_config = live_config_from_query(Dict(
        "backend" => "metal",
        "conv" => "auto",
    ))
    @test periodic_config.backend == :metal
    @test periodic_config.convolution == :fft

    config = live_config_from_query(Dict(
        "backend" => "gpu",
        "conv" => "auto",
        "N" => "101",
        "fast_n" => "true",
        "fps" => "20",
        "speed" => "0",
        "seed" => "42",
        "duty_cycle" => "25",
        "boundary_x" => "edge",
        "boundary_y" => "zero",
        "coupling" => "midline",
        "coupling_strength" => "0.03",
        "overlap_rows" => "5",
    ))
    @test config.backend == :metal
    @test config.convolution == :separable
    @test config.N == 105
    @test config.target_fps == 20
    @test config.speed == 0
    @test config.seed == 42
    @test config.duty_cycle_percent == 25.0f0
    @test config.boundary_x == :edge
    @test config.boundary_y == :zero
    @test config.coupling == :midline
    @test config.coupling_strength == 0.03f0
    @test config.overlap_rows == 6

    legacy_config = live_config_from_query(Dict(
        "backend" => "metal",
        "conv" => "separable",
        "boundary" => "edge",
    ))
    @test legacy_config.boundary_x == :edge
    @test legacy_config.boundary_y == :edge
end

@testset "live applet server" begin
    server = serve_applet_async(host="127.0.0.1", port=0, verbose=false)
    try
        url = applet_url(server, "127.0.0.1")
        response = HTTP.get(url; status_exception=false)
        @test response.status == 200
        @test occursin("RSE Real-Time Viewer", String(response.body))

        address = "127.0.0.1:$(HTTP.port(server))"
        HTTP.WebSockets.open("ws://$address/stream?backend=cpu&N=25&fps=10&speed=0&max_frames=1&coupling=midline&overlap_rows=6") do ws
            hello = String(HTTP.WebSockets.receive(ws))
            frame = String(HTTP.WebSockets.receive(ws))
            @test occursin("\"type\":\"hello\"", hello)
            @test occursin("\"type\":\"frame\"", frame)
            @test occursin("\"coupling\":\"midline\"", hello)
            @test occursin("\"cols\":50", frame)
        end
    finally
        close(server)
    end
end
