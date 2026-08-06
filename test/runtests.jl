using Test
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

        for boundary in (:periodic, :edge, :zero)
            RSEModel.separable_convolution!(out_e_separate, ce_separate, Ue; gpu_threads=128, boundary=boundary)
            RSEModel.separable_convolution!(out_i_separate, ci_separate, Ui; gpu_threads=128, boundary=boundary)
            RSEModel.separable_convolution_pair!(
                out_e_pair,
                out_i_pair,
                ce_pair,
                ci_pair,
                Ue,
                Ui;
                gpu_threads=128,
                boundary=boundary,
            )
            Metal.synchronize()

            @test Array(out_e_pair) ≈ Array(out_e_separate) rtol=1e-6 atol=1e-6
            @test Array(out_i_pair) ≈ Array(out_i_separate) rtol=1e-6 atol=1e-6
        end
    else
        @test_skip "Metal.jl is not functional on this machine."
    end
end

@testset "retinal transform" begin
    img = reshape(collect(Float32, 1:25), 5, 5)
    ret = retinal_transform(img)
    @test size(ret) == size(img)
end

@testset "live applet config" begin
    config = live_config_from_query(Dict(
        "backend" => "gpu",
        "conv" => "auto",
        "N" => "101",
        "fast_n" => "true",
        "fps" => "20",
        "speed" => "0",
        "seed" => "42",
        "duty_cycle" => "25",
        "boundary" => "edge",
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
    @test config.boundary == :edge
    @test config.coupling == :midline
    @test config.coupling_strength == 0.03f0
    @test config.overlap_rows == 6
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
