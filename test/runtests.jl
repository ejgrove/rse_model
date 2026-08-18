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

@testset "field geometry" begin
    square = field_geometry(:square, 25)
    @test square.kind == :square
    @test size(square.mask) == (25, 25)
    @test all(square.mask)

    v1 = field_geometry(:double_sech)
    @test v1.kind == :double_sech
    @test v1.rows == 81
    @test v1.cols > v1.rows
    @test 0 < count(v1.mask) < length(v1.mask)
    col_counts = [count(@view v1.mask[:, col]) for col in axes(v1.mask, 2)]
    @test col_counts[findlast(>(0), col_counts)] <= 3
    border = RSEModel.field_border_mask(v1.mask, 3)
    @test size(border) == size(v1.mask)
    @test 0 < count(border) < count(v1.mask)
    @test all(border .<= v1.mask)

    dense = field_geometry(:double_sech; density=1.5)
    @test dense.rows > v1.rows
    @test dense.cols > v1.cols
    @test field_geometry(:double_sech, 25; density=1.0).rows == v1.rows

    @test double_sech_shear(1.05, 0.0, 1.05) ≈ 1.0
    @test dipole_double_sech_map(0.0, 0.0) isa ComplexF64

    left = reshape(collect(Float32, 1:length(v1.mask)), size(v1.mask))
    right = fill(2.0f0, size(v1.mask))
    ret = double_sech_retinal_transform(left, right, v1; output_size=(31, 31))
    @test size(ret) == (31, 31)
    @test all(isfinite, ret)

    left_const = fill(10.0f0, size(v1.mask))
    right_const = fill(20.0f0, size(v1.mask))
    split_ret = double_sech_retinal_transform(left_const, right_const, v1; output_size=(31, 31), seam_blend_pixels=0)
    @test split_ret[16, 28] ≈ 10.0f0
    @test split_ret[16, 4] ≈ 20.0f0
    blended_ret = double_sech_retinal_transform(left_const, right_const, v1; output_size=(31, 31))
    @test blended_ret[16, 16] ≈ 15.0f0
    @test blended_ret[16, 28] ≈ 10.0f0
    @test blended_ret[16, 4] ≈ 20.0f0

    row_values = repeat(reshape(collect(Float32, 1:v1.rows), v1.rows, 1), 1, v1.cols)
    left_row_ret = double_sech_retinal_transform(row_values, zeros(Float32, size(v1.mask)), v1; output_size=(31, 31), seam_blend_pixels=0)
    right_row_ret = double_sech_retinal_transform(zeros(Float32, size(v1.mask)), row_values, v1; output_size=(31, 31), seam_blend_pixels=0)
    @test left_row_ret[8, 24] < left_row_ret[24, 24]
    @test right_row_ret[8, 8] > right_row_ret[24, 8]

    U = ones(Float32, size(v1.mask))
    apply_field_mask!(U, v1)
    @test all(U[.!v1.mask] .== 0)
    @test all(U[v1.mask] .== 1)
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

    masked_left_e = fill(1.0f0, 8, 4)
    masked_left_i = fill(2.0f0, 8, 4)
    masked_right_e = fill(3.0f0, 8, 4)
    masked_right_i = fill(4.0f0, 8, 4)
    mask = trues(8, 4)
    mask[1, 1] = false
    RSEModel._apply_masked_midline_coupling!(
        masked_left_e,
        masked_left_i,
        masked_right_e,
        masked_right_i,
        mask,
        0.25f0,
        4,
    )
    @test masked_left_e[1, 1] == 1.0f0
    @test masked_right_e[2, 1] == 3.0f0
    @test masked_left_e[1, 2] == 1.5f0
    @test masked_right_i[2, 2] == 3.5f0

    border_left_e = fill(1.0f0, 5, 3)
    border_left_i = fill(2.0f0, 5, 3)
    border_right_e = fill(3.0f0, 5, 3)
    border_right_i = fill(4.0f0, 5, 3)
    border_mask = falses(5, 3)
    border_mask[1, 2] = true
    border_mask[5, 2] = true
    border_mask[3, 1] = true
    border_mask[3, 3] = true
    RSEModel._apply_border_coupling!(border_left_e, border_left_i, border_right_e, border_right_i, border_mask, 0.25f0)
    @test border_left_e[1, 2] == 1.5f0
    @test border_right_e[5, 2] == 2.5f0
    @test border_left_i[5, 2] == 2.5f0
    @test border_right_i[1, 2] == 3.5f0
    @test border_left_e[2, 2] == 1.0f0
end

@testset "coupled live view orientation" begin
    left = reshape(collect(Float32, 1:9), 3, 3)
    right = reshape(collect(Float32, 10:18), 3, 3)
    display = Matrix{Float32}(undef, 3, 6)
    retinal_source = Matrix{Float32}(undef, 6, 3)

    RSEModel._fill_coupled_views!(display, left, right)
    RSEModel._fill_coupled_retinal_source!(retinal_source, left, right)

    @test display[:, 1:3] == left
    @test display[:, 4:6] == right
    @test retinal_source[1:3, :] == left
    @test retinal_source[4:6, :] == right
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

        for (boundary_x, boundary_y) in ((:periodic, :periodic), (:edge, :zero), (:zero, :edge), (:partial_reflect, :edge))
            RSEModel.separable_convolution!(
                out_e_separate,
                ce_separate,
                Ue;
                gpu_threads=128,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
                partial_reflect_strength=0.35,
            )
            RSEModel.separable_convolution!(
                out_i_separate,
                ci_separate,
                Ui;
                gpu_threads=128,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
                partial_reflect_strength=0.35,
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
                partial_reflect_strength=0.35,
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

@testset "metal separable full kernel matches fft alignment" begin
    if Metal.functional()
        N = 17
        U = reshape(Float32.(1:(N * N)), N, N) ./ Float32(N * N)
        K = generate_gaussian_kernel(2.0, N; dtype=Float32)
        cpu_out = similar(U)
        RSEModel.fft_convolution!(cpu_out, RSEModel.FFTConvolver(K, U; flags=FFTW.ESTIMATE), U)

        gpu_U = Metal.MtlArray(U)
        gpu_out = similar(gpu_U)
        RSEModel.separable_convolution!(
            gpu_out,
            RSEModel.MetalSeparableConvolver(2.0, gpu_U; cutoff=100.0),
            gpu_U;
            gpu_threads=128,
            boundary=:periodic,
        )
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

    wide_img = reshape(collect(Float32, 1:50), 5, 10)
    wide_ret = retinal_transform(wide_img)
    @test size(wide_ret) == size(wide_img)
    square_from_wide = retinal_transform(wide_img; output_size=(5, 5))
    @test size(square_from_wide) == (5, 5)

    angle_by_row = repeat(reshape(collect(Float32, 1:8), 8, 1), 1, 5)
    angle_ret = retinal_transform(angle_by_row; output_size=(3, 3))
    @test angle_ret[2, 3] ≈ 1.0f0
    @test angle_ret[2, 1] ≈ 5.0f0

    offset_ret = retinal_transform(angle_by_row; output_size=(3, 3), angle_origin=Float32(pi / 2))
    @test offset_ret[1, 2] ≈ 1.0f0
    @test offset_ret[3, 2] ≈ 5.0f0
end

@testset "parameter search smoke" begin
    metal_auto = RSEModel._validate_search_config(ParameterSearchConfig(backend=:metal, convolution=:auto))
    @test metal_auto.convolution == :separable
    @test metal_auto.kernel_cutoff == 4.0

    equivalence_config = RSEModel._validate_search_config(ParameterSearchConfig(
        N=9,
        A_values=[0.2],
        period_values=[10.0],
        times_ms=[2, 4],
        backend=:cpu,
        convolution=:fft,
        seed=3,
        view=:cortical,
    ))
    equivalence_job = first(RSEModel._search_jobs(equivalence_config))
    equivalence_result = RSEModel._run_parameter_search_job(
        equivalence_job,
        equivalence_config,
        RSEModel._sample_interval_ms(equivalence_config.times_ms),
        maximum(equivalence_config.times_ms),
    )
    reference = run_simulation(
        N=equivalence_config.N,
        A=equivalence_job.amplitude,
        T=equivalence_job.period,
        Se=equivalence_config.Se,
        Si=equivalence_config.Si,
        start_time=minimum(equivalence_config.times_ms),
        end_time=maximum(equivalence_config.times_ms),
        seed=equivalence_job.seed,
        plot=false,
        gif=false,
        interval=RSEModel._sample_interval_ms(equivalence_config.times_ms),
        p=ModelParams(),
        backend=:cpu,
        convolution=:fft,
        duty_cycle_percent=equivalence_config.duty_cycle_percent,
    )
    reference_images = Dict(
        round(Int, snapshot.t) => RSEModel._search_cell_rgb(snapshot, equivalence_config.view, equivalence_config.cmap)
        for snapshot in reference.images
    )
    @test equivalence_result.images == reference_images

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
    @test occursin("kernel_cutoff=4.0", read(joinpath(out_path, "config.txt"), String))
    @test occursin("stimulus_threshold=0", read(joinpath(out_path, "config.txt"), String))
    @test occursin("index,total,a_idx,t_idx,A,T_ms", first(readlines(joinpath(out_path, "summary.csv"))))
end

@testset "live applet config" begin
    periodic_config = live_config_from_query(Dict(
        "backend" => "metal",
    ))
    @test periodic_config.backend == :metal
    @test periodic_config.convolution == :separable

    auto_config = live_config_from_query(Dict(
        "backend" => "metal",
        "conv" => "auto",
    ))
    @test auto_config.backend == :metal
    @test auto_config.convolution == :fft

    cpu_default_config = live_config_from_query(Dict(
        "backend" => "cpu",
    ))
    @test cpu_default_config.backend == :cpu
    @test cpu_default_config.convolution == :fft

    config = live_config_from_query(Dict(
        "backend" => "gpu",
        "conv" => "auto",
        "N" => "101",
        "fast_n" => "true",
        "fps" => "20",
        "speed" => "0",
        "seed" => "42",
        "duty_cycle" => "25",
        "Se" => "1.75",
        "Si" => "4.5",
        "dt" => "0.1",
        "boundary_x" => "edge",
        "boundary_y" => "zero",
        "coupling" => "midline",
        "coupling_strength" => "0.03",
        "overlap_rows" => "5",
        "activity_scale" => "simulation",
    ))
    @test config.backend == :metal
    @test config.convolution == :separable
    @test config.N == 105
    @test config.target_fps == 20
    @test config.speed == 0
    @test config.seed == 42
    @test config.duty_cycle_percent == 25.0f0
    @test config.Se == 1.75f0
    @test config.Si == 4.5f0
    @test config.dt == 0.1f0
    @test config.boundary_x == :edge
    @test config.boundary_y == :zero
    @test config.coupling == :overlap
    @test config.coupling_strength == 0.03f0
    @test config.overlap_rows == 6
    @test config.activity_scale == :simulation

    disconnected_config = live_config_from_query(Dict(
        "backend" => "metal",
        "coupling" => "no_connection",
    ))
    @test disconnected_config.coupling == :no_connection

    legacy_config = live_config_from_query(Dict(
        "backend" => "metal",
        "conv" => "separable",
        "boundary" => "edge",
    ))
    @test legacy_config.boundary_x == :edge
    @test legacy_config.boundary_y == :edge

    runtime = RSEModel.LiveRuntime()
    @test RSEModel._apply_visual_control!(runtime, "visual:fps=12&speed=0.5&activity_scale=simulation")
    @test runtime.target_fps == 12
    @test runtime.speed == 0.5
    @test runtime.activity_scale == :simulation
    @test RSEModel._apply_visual_control!(runtime, "visual:activity_scale=frame")
    @test runtime.activity_scale == :frame
    @test !RSEModel._apply_visual_control!(runtime, "pause")

    p = ModelParams{Float32}()
    scale_runtime = RSEModel.LiveRuntime(activity_scale=:simulation)
    first_frame = RSEModel._make_live_frame(
        Float32[1 2; 3 4],
        1,
        0.0f0,
        1.0,
        time_ns(),
        1,
        p;
        runtime=scale_runtime,
    )
    second_frame = RSEModel._make_live_frame(
        Float32[2 3; 5 6],
        2,
        1.0f0,
        1.0,
        time_ns(),
        1,
        p;
        runtime=scale_runtime,
    )
    @test first_frame.lo == 1.0f0
    @test first_frame.hi == 4.0f0
    @test second_frame.lo == 1.0f0
    @test second_frame.hi == 6.0f0

    local_frame = RSEModel._make_live_frame(
        Float32[2 3; 5 6],
        1,
        0.0f0,
        1.0,
        time_ns(),
        1,
        p;
        runtime=RSEModel.LiveRuntime(activity_scale=:frame),
    )
    @test local_frame.lo == 2.0f0
    @test local_frame.hi == 6.0f0

    double_sech_config = live_config_from_query(Dict(
        "backend" => "metal",
        "conv" => "auto",
        "N" => "25",
        "field_geometry" => "double_sech",
        "field_density" => "1.5",
        "boundary" => "partial_reflective",
        "partial_reflect_strength" => "0.35",
        "coupling" => "off",
    ))
    @test double_sech_config.field_geometry == :double_sech
    @test double_sech_config.field_density == 1.5
    @test double_sech_config.N == 123
    @test double_sech_config.convolution == :separable
    @test double_sech_config.boundary_x == :partial_reflect
    @test double_sech_config.boundary_y == :partial_reflect
    @test double_sech_config.partial_reflect_strength == 0.35f0
    @test RSEModel._uses_two_hemispheres(double_sech_config)

    @test_throws ArgumentError live_config_from_query(Dict(
        "backend" => "cpu",
        "field_geometry" => "double_sech",
    ))
end

@testset "live applet server" begin
    server = serve_applet_async(host="127.0.0.1", port=0, verbose=false)
    try
        url = applet_url(server, "127.0.0.1")
        response = HTTP.get(url; status_exception=false)
        body = String(response.body)
        @test response.status == 200
        @test occursin("Real-time Strobe Hallucination Simulator", body)
        @test occursin("Rule-Ermentrout-Stroffegen", body)
        @test occursin("GPU (Metal)", body)
        @test occursin("Amplitude", body)
        @test occursin("Period (ms)", body)
        @test occursin("Duty cycle (%)", body)
        @test occursin("Max speed", body)
        @test occursin("id=\"maxSpeed\"", body)
        @test occursin("&sigma;<sub>e</sub>", body)
        @test occursin("id=\"dt\"", body)
        @test occursin("value=\"no_connection\"", body)
        @test occursin("value=\"overlap\"", body)
        @test occursin("id=\"corticalFrame\"", body)
        @test occursin("retinal-angle-90", body)
        @test occursin("id=\"colorMap\"", body)
        @test occursin("id=\"activityScale\"", body)
        @test occursin("id=\"frameSelect\"", body)
        @test occursin("id=\"fieldGraph\"", body)
        @test occursin("id=\"fieldInfo\"", body)
        @test occursin("value=\"phase\"", body)
        @test occursin("id=\"phaseGraph\"", body)
        @test occursin("id=\"phaseInfo\"", body)
        @test occursin("function drawPhasePlane", body)
        @test occursin("nipy_spectral", body)
        @test occursin("id=\"stimulusGraph\"", body)
        @test occursin("id=\"fieldGeometry\"", body)
        @test occursin("id=\"fieldDensityControl\"", body)
        @test occursin("id=\"nControl\"", body)
        @test occursin("id=\"legendLow\"", body)
        @test occursin("id=\"legendHigh\"", body)
        @test occursin("id=\"boundaryControl\"", body)
        @test occursin("id=\"boundaryXControl\"", body)
        @test occursin("id=\"partialReflectControl\"", body)
        @test occursin("id=\"partialReflectStrength\"", body)
        @test occursin("Selected Parameters", body)
        @test occursin("data-preset=\"p1\"", body)
        @test occursin("id=\"printParams\"", body)
        @test occursin("id=\"paramOutput\"", body)
        @test occursin("value=\"double_sech\"", body)
        @test occursin("value=\"partial_reflect\"", body)
        @test occursin("function formatSimTime", body)
        @test occursin("function drawFieldGraph", body)
        @test occursin("fovea", body)
        @test occursin("periphery", body)
        @test occursin("cortical-fovea-left", body)
        @test occursin("cortical-fovea-right", body)
        @test !occursin("retinal-fovea", body)
        @test !occursin("retinal-periphery", body)
        @test !occursin("value=\"auto\"", body)
        @test !occursin("Boundary / Coupling", body)
        @test !occursin("server-side log-polar map", body)
        @test occursin("event.code === \"Space\"", body)
        @test occursin("event.code === \"Enter\"", body)
        @test !occursin("id=\"dtMetric\"", body)
        @test !occursin("id=\"gridN\"", body)

        address = "127.0.0.1:$(HTTP.port(server))"
        HTTP.WebSockets.open("ws://$address/stream?backend=cpu&N=25&fps=10&speed=0&max_frames=1&coupling=overlap&overlap_rows=6&Se=1.5&Si=4.5&dt=0.1&activity_scale=simulation") do ws
            hello = String(HTTP.WebSockets.receive(ws))
            frame = String(HTTP.WebSockets.receive(ws))
            @test occursin("\"type\":\"hello\"", hello)
            @test occursin("\"type\":\"frame\"", frame)
            @test occursin("\"coupling\":\"overlap\"", hello)
            @test occursin("\"Se\":1.5", hello)
            @test occursin("\"Si\":4.5", hello)
            @test occursin("\"dt\":0.1", hello)
            @test occursin("\"activityScale\":\"simulation\"", hello)
            @test occursin("\"cols\":50", frame)
            @test occursin("\"retinalRows\":25", frame)
            @test occursin("\"retinalCols\":25", frame)
            @test occursin("\"phaseCount\":1250", frame)
            @test occursin("\"phaseEData\":", frame)
            @test occursin("\"phaseIData\":", frame)
        end
    finally
        close(server)
    end
end
