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
    ))
    @test config.backend == :metal
    @test config.convolution == :separable
    @test config.N == 105
    @test config.target_fps == 20
    @test config.speed == 0
    @test config.seed == 42
end

@testset "live applet server" begin
    server = serve_applet_async(host="127.0.0.1", port=0, verbose=false)
    try
        url = applet_url(server, "127.0.0.1")
        response = HTTP.get(url; status_exception=false)
        @test response.status == 200
        @test occursin("RSE Real-Time Viewer", String(response.body))

        address = "127.0.0.1:$(HTTP.port(server))"
        HTTP.WebSockets.open("ws://$address/stream?backend=cpu&N=25&fps=10&speed=0&max_frames=1") do ws
            hello = String(HTTP.WebSockets.receive(ws))
            frame = String(HTTP.WebSockets.receive(ws))
            @test occursin("\"type\":\"hello\"", hello)
            @test occursin("\"type\":\"frame\"", frame)
        end
    finally
        close(server)
    end
end
