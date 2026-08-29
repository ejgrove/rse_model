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
    ret_plan = double_sech_retinal_plan(v1; output_size=(31, 31))
    cached_ret = similar(ret)
    double_sech_retinal_transform!(cached_ret, left, right, ret_plan)
    @test cached_ret == ret

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

@testset "paired metal separable convolution" begin
    if Metal.functional()
        Ue = Metal.rand(Float32, 25, 25)
        Ui = Metal.rand(Float32, 25, 25)
        out_e_pair = similar(Ue)
        out_i_pair = similar(Ui)

        ce_pair = RSEModel.MetalSeparableConvolver(2.0, Ue; cutoff=3.0)
        ci_pair = RSEModel.MetalSeparableConvolver(5.0, Ui; cutoff=3.0)

        for (boundary_x, boundary_y) in ((:periodic, :periodic), (:edge, :zero), (:zero, :edge), (:partial_reflect, :edge))
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

            @test size(out_e_pair) == size(Ue)
            @test size(out_i_pair) == size(Ui)
            @test all(isfinite, Array(out_e_pair))
            @test all(isfinite, Array(out_i_pair))
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
        gpu_out_e = similar(gpu_U)
        gpu_out_i = similar(gpu_U)
        RSEModel.separable_convolution_pair!(
            gpu_out_e,
            gpu_out_i,
            RSEModel.MetalSeparableConvolver(2.0, gpu_U; cutoff=100.0),
            RSEModel.MetalSeparableConvolver(2.0, gpu_U; cutoff=100.0),
            gpu_U,
            gpu_U;
            gpu_threads=128,
            boundary=:periodic,
        )
        Metal.synchronize()

        @test Array(gpu_out_e) ≈ cpu_out rtol=1e-5 atol=1e-5
        @test Array(gpu_out_i) ≈ cpu_out rtol=1e-5 atol=1e-5
    else
        @test_skip "Metal.jl is not functional on this machine."
    end
end

@testset "retinal transform" begin
    img = reshape(collect(Float32, 1:25), 5, 5)
    ret = retinal_transform(img)
    @test size(ret) == size(img)
    plan = retinal_map_plan(size(img); output_size=(9, 9))
    cached_ret = zeros(Float32, 9, 9)
    retinal_transform!(cached_ret, img, plan)
    @test cached_ret == retinal_transform(img; output_size=(9, 9))

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
        "Ge" => "1.25",
        "Gi" => "0.35",
        "Se" => "1.75",
        "Si" => "4.5",
        "Aee" => "9.5",
        "Aei" => "11.5",
        "Aie" => "7.5",
        "Aii" => "2.5",
        "dt" => "0.1",
        "boundary_x" => "edge",
        "boundary_y" => "zero",
        "coupling" => "midline",
        "coupling_strength" => "0.03",
        "overlap_rows" => "5",
        "retinal_resolution" => "320",
        "retinal_rendering" => "mapped",
        "activity_scale" => "simulation",
    ))
    @test config.backend == :metal
    @test config.convolution == :separable
    @test config.N == 105
    @test config.target_fps == 20
    @test config.speed == 0
    @test config.seed == 42
    @test config.duty_cycle_percent == 25.0f0
    @test config.Ge == 1.25f0
    @test config.Gi == 0.35f0
    @test config.Se == 1.75f0
    @test config.Si == 4.5f0
    @test config.Aee == 9.5f0
    @test config.Aei == 11.5f0
    @test config.Aie == 7.5f0
    @test config.Aii == 2.5f0
    @test config.dt == 0.1f0
    @test config.boundary_x == :edge
    @test config.boundary_y == :zero
    @test config.coupling == :overlap
    @test config.coupling_strength == 0.03f0
    @test config.overlap_rows == 6
    @test config.retinal_resolution == 321
    @test config.retinal_rendering == :mapped
    @test RSEModel._retinal_output_size(config) == (321, 321)
    @test config.activity_scale == :simulation
    model_params = RSEModel._live_model_params(config)
    @test model_params.Ge == config.Ge
    @test model_params.Gi == config.Gi
    @test model_params.Aee == config.Aee
    @test model_params.Aei == config.Aei
    @test model_params.Aie == config.Aie
    @test model_params.Aii == config.Aii
    @test live_config_from_query(Dict("backend" => "cpu", "seed" => "1")).seed == 1
    @test live_config_from_query(Dict("backend" => "cpu", "seed" => "999")).seed == 999
    @test_throws ArgumentError live_config_from_query(Dict("backend" => "cpu", "seed" => "0"))
    @test_throws ArgumentError live_config_from_query(Dict("backend" => "cpu", "seed" => "1000"))
    @test_throws ArgumentError live_config_from_query(Dict("backend" => "cpu", "Aee" => "-1"))
    @test_throws ArgumentError live_config_from_query(Dict("backend" => "cpu", "Gi" => "-0.1"))

    interpolated_config = live_config_from_query(Dict(
        "backend" => "cpu",
        "N" => "25",
        "retinal_resolution" => "321",
    ))
    @test interpolated_config.retinal_rendering == :interpolated
    @test RSEModel._retinal_output_size(interpolated_config) == (25, 25)

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
    @test RSEModel._steps_per_frame(30, 1.0, p) == 167
    @test RSEModel._steps_per_frame(30, 0.1, p) == 17
    @test RSEModel._steps_per_frame(30, 0.01, p) == 2
    @test RSEModel._steps_per_frame(30, 0.001, p) == 1
    @test RSEModel._steps_per_frame(30, 0.006, p) == 1
    @test RSEModel._steps_per_frame(30, 0.0, p) == 1

    max_runtime = RSEModel._live_runtime(RSEModel.LiveConfig(
        target_fps=30,
        speed=0.0,
        dt=0.2f0,
    ))
    @test max_runtime.max_steps_per_frame == 167
    @test RSEModel._steps_per_frame(max_runtime, p) == 167
    RSEModel._update_max_steps_per_frame!(max_runtime, 167, 8.0, 12.0)
    @test max_runtime.max_steps_per_frame > 167
    @test RSEModel._steps_per_frame(max_runtime, p) == max_runtime.max_steps_per_frame

    scale_runtime = RSEModel.LiveRuntime(activity_scale=:simulation)
    warmup_frame = RSEModel._make_live_frame(
        Float32[1 2; 3 100],
        1,
        499.9f0,
        1.0,
        time_ns(),
        1,
        p;
        runtime=scale_runtime,
    )
    warmup_scale_lo = scale_runtime.scale_lo
    warmup_scale_hi = scale_runtime.scale_hi
    first_scaled_frame = RSEModel._make_live_frame(
        Float32[2 3; 5 6],
        2,
        500.0f0,
        1.0,
        time_ns(),
        1,
        p;
        runtime=scale_runtime,
    )
    later_scaled_frame = RSEModel._make_live_frame(
        Float32[1 4; 7 8],
        3,
        501.0f0,
        1.0,
        time_ns(),
        1,
        p;
        runtime=scale_runtime,
    )
    @test RSEModel.ACTIVITY_SCALE_WARMUP_MS == 500.0
    @test warmup_frame.lo == 1.0f0
    @test warmup_frame.hi == 100.0f0
    @test warmup_frame.steps_per_frame == 1
    @test warmup_scale_lo == Float32(Inf)
    @test warmup_scale_hi == -Float32(Inf)
    @test first_scaled_frame.lo == 2.0f0
    @test first_scaled_frame.hi == 6.0f0
    @test later_scaled_frame.lo == 1.0f0
    @test later_scaled_frame.hi == 8.0f0

    frame_runtime = RSEModel.LiveRuntime(activity_scale=:frame)
    local_frame = RSEModel._make_live_frame(
        Float32[2 3; 5 6],
        1,
        0.0f0,
        1.0,
        time_ns(),
        1,
        p;
        runtime=frame_runtime,
    )
    @test local_frame.lo == 2.0f0
    @test local_frame.hi == 6.0f0
    @test frame_runtime.scale_lo == Float32(Inf)
    @test frame_runtime.scale_hi == -Float32(Inf)

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
        css_response = HTTP.get(url * "styles.css"; status_exception=false)
        js_response = HTTP.get(url * "app.js"; status_exception=false)
        index_html = String(response.body)
        body = string(index_html, String(css_response.body), String(js_response.body))
        @test response.status == 200
        @test css_response.status == 200
        @test js_response.status == 200
        @test occursin("text/css", HTTP.header(css_response, "Content-Type"))
        @test occursin("text/javascript", HTTP.header(js_response, "Content-Type"))
        @test occursin("href=\"/styles.css\"", index_html)
        @test occursin("src=\"/app.js\"", index_html)
        @test !occursin("<style>", index_html)
        @test !occursin("<script>\n", index_html)
        @test occursin("Real-time Strobe Hallucination Simulator", body)
        @test occursin("Rule-Ermentrout-Stroffegen", body)
        @test occursin("--text-title: clamp(24px, 2.3vw, 33px)", body)
        @test occursin("--app-shell-min: 1400px", body)
        @test occursin("--app-shell-max: 2000px", body)
        @test occursin("--parameter-pane-width: 490px", body)
        @test occursin("overflow-x: auto", body)
        @test occursin("width: clamp(var(--app-shell-min), calc(100vw - var(--page-gutter) - var(--page-gutter)), var(--app-shell-max))", body)
        @test occursin("min-width: var(--app-shell-min)", body)
        @test occursin("margin: 0 auto", body)
        @test occursin("grid-template-columns: var(--parameter-pane-width) minmax(0, 1fr)", body)
        @test occursin("width: var(--parameter-pane-width)", body)
        @test occursin("grid-template-columns: repeat(2, minmax(0, 1fr))", body)
        @test occursin("width: 100%", body)
        @test !occursin("width: min(1440px, 100%)", body)
        @test !occursin("--controls-width", body)
        @test !occursin("--plot-window-size", body)
        @test !occursin("grid-template-columns: minmax(360px, 0.42fr) minmax(760px, 1fr)", body)
        @test !occursin("width: min(100%, var(--plot-window-size))", body)
        @test occursin("GPU (Metal)", body)
        @test occursin("Amplitude", body)
        @test occursin("Period (ms)", body)
        @test occursin("Duty cycle (%)", body)
        @test occursin("Stimulus gains", body)
        @test occursin("id=\"ge\"", body)
        @test occursin("id=\"gi\"", body)
        @test occursin("Synaptic weights", body)
        @test occursin("id=\"aee\"", body)
        @test occursin("id=\"aei\"", body)
        @test occursin("id=\"aie\"", body)
        @test occursin("id=\"aii\"", body)
        @test occursin("<label>FPS<input id=\"fps\"", body)
        @test occursin("<div class=\"metric\"><span>FPS</span>", body)
        @test occursin("Visualization speed", body)
        @test occursin("Time step (ms)", body)
        @test occursin("Sheet size", body)
        @test occursin("Step interval", body)
        @test occursin("function clampSpeedInputToMinimum", body)
        @test occursin("return clampSpeedInputToMinimum(false)", body)
        @test occursin("syncSpeedControls(true)", body)
        @test occursin("syncSpeedControls(false)", body)
        @test occursin("1 is real time, 0.5 is 50%, and 2 is 200%", body)
        @test occursin("function updateRateMetrics", body)
        @test occursin("const actualFps = (rateSamples.length - 1) * 1000 / elapsedWallMs", body)
        @test occursin("const actualRealtimeX = elapsedSimulationMs / elapsedWallMs", body)
        @test occursin("updateRateMetrics(performance.now(), msg.t)", body)
        @test !occursin("const observedFps = 1000 /", body)
        @test occursin("Simulation time", body)
        @test occursin("Real-time (x)", body)
        @test occursin("Max speed", body)
        @test occursin("id=\"maxSpeed\"", body)
        @test occursin("id=\"stepInterval\"", body)
        @test occursin("&sigma;<sub>e</sub>", body)
        @test occursin("id=\"dt\"", body)
        @test occursin("value=\"no_connection\"", body)
        @test occursin("value=\"overlap\"", body)
        @test occursin("id=\"corticalFrame\"", body)
        @test occursin("retinal-angle-90", body)
        @test occursin("id=\"colorMap\"", body)
        @test occursin("id=\"activityScale\"", body)
        @test occursin("simulation min/max (after 500 ms)", body)
        @test occursin("id=\"plotContours\"", body)
        @test occursin("Resolution (contours)", body)
        @test occursin("function activeContourCount", body)
        @test occursin("function legendColorString", body)
        @test occursin("id=\"frameSelect\"", body)
        @test occursin("E - I", body)
        @test occursin("pointwise E - I", body)
        @test occursin("inhibitory curve is negative", body)
        @test !occursin("pointwise E + I", body)
        @test occursin("id=\"fieldGraph\"", body)
        @test occursin("id=\"fieldInfo\"", body)
        @test occursin("value=\"phase\"", body)
        @test occursin("id=\"phaseGraph\"", body)
        @test occursin("id=\"phaseInfo\"", body)
        @test occursin("id=\"phaseIncludeAverage\"", body)
        @test occursin("meanFieldParams", body)
        @test occursin("function activeMeanFieldParams", body)
        @test occursin("drawMeanFieldNullclines", body)
        @test occursin("dUe/dt=0", body)
        @test occursin("dUi/dt=0", body)
        @test occursin("function drawPhasePlane", body)
        @test !occursin("id=\"phaseColoredNodes\"", body)
        @test !occursin("hsvToRgb", body)
        @test occursin("nipy_spectral", body)
        @test occursin("id=\"stimulusGraph\"", body)
        @test occursin("id=\"fieldGeometry\"", body)
        @test occursin("id=\"fieldDensityControl\"", body)
        @test occursin("id=\"nControl\"", body)
        @test occursin("id=\"legendLow\"", body)
        @test occursin("id=\"legendHigh\"", body)
        @test occursin("id=\"retinalResolution\"", body)
        @test occursin("id=\"retinalRendering\"", body)
        @test occursin("imageSmoothingQuality = \"high\"", body)
        @test occursin("visual-field-wrap", body)
        @test occursin("plot-title", body)
        @test !occursin("legend-label", body)
        @test occursin("id=\"boundaryControl\"", body)
        @test occursin("id=\"boundaryXControl\"", body)
        @test occursin("id=\"partialReflectControl\"", body)
        @test occursin("id=\"partialReflectStrength\"", body)
        @test occursin("Selected Parameters", body)
        @test occursin("id=\"presetGrid\"", body)
        @test occursin("id=\"presetTitle\"", body)
        @test occursin("const presetRows = [", body)
        @test occursin("label: \"Stripes\", values: { n: 121, amp: 0.7, period: 55", body)
        @test occursin("label: \"Rectangular checkerboard\"", body)
        @test occursin("seed: 11", body)
        @test occursin("fastN: false", body)
        @test occursin("el.checked = Boolean(value)", body)
        @test occursin("function renderPresetButtons", body)
        @test occursin("id=\"seed\" type=\"number\" min=\"1\" max=\"999\"", body)
        @test occursin("id=\"randomizeSeed\"", body)
        @test occursin("Randomize seed on restart", body)
        @test occursin("function randomSeedValue", body)
        @test occursin("function normalizedSeedValue", body)
        @test occursin("function resetStream({ randomizeSeed = true } = {})", body)
        @test occursin("resetStream({ randomizeSeed: false })", body)
        @test occursin("id=\"overlapRowsControl\" class=\"hidden-control\"", body)
        @test occursin("id=\"couplingStrengthControl\" class=\"hidden-control\"", body)
        @test occursin("function syncCouplingControls", body)
        @test !occursin("Boundary periodic, coupling none", body)
        @test !occursin("N64 A0.2 T55", body)
        @test occursin("id=\"printParams\"", body)
        @test occursin("id=\"paramOutput\"", body)
        @test occursin("Print settings", body)
        @test occursin("gain: Number(els.couplingStrength.value)", body)
        @test occursin("sheet_size:", body)
        @test occursin("time_step_ms:", body)
        @test !occursin("stream query:", body)
        @test !occursin("stream path:", body)
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
        @test occursin("function isTypingShortcutTarget", body)
        @test occursin("document.addEventListener(\"keyup\"", body)
        @test !occursin("id=\"dtMetric\"", body)
        @test !occursin("id=\"gridN\"", body)
        @test !occursin("id=\"msStep\"", body)
        @test !occursin("ms / step", body)
        @test !occursin("id=\"skipInterval\"", body)
        @test !occursin("Stream FPS", body)
        @test !occursin("Speed x", body)
        @test !occursin("Speed (x)", body)
        @test !occursin(">Coupling g<input", body)
        @test !occursin("Coupling gain", body)
        @test !occursin("Coupling<select", body)
        @test !occursin("Boundary x<select", body)
        @test !occursin("Boundary y<select", body)
        @test !occursin("Parameter Presets", body)
        @test !occursin("<strong>Dots</strong>", body)
        @test !occursin("mean E=", body)
        @test !occursin("mean I=", body)
        @test !occursin("average hidden", body)
        @test !occursin("E/I firing-rate state cloud", body)
        @test !occursin("Print parameters", body)

        @test isfile(joinpath(dirname(@__DIR__), "docs", "web-design-principles.md"))
        @test isfile(joinpath(dirname(@__DIR__), "data", "rse_params.xlsx"))

        address = "127.0.0.1:$(HTTP.port(server))"
        HTTP.WebSockets.open("ws://$address/stream?backend=cpu&N=25&retinal_resolution=51&retinal_rendering=mapped&fps=10&speed=0&max_frames=1&coupling=overlap&overlap_rows=6&Se=1.5&Si=4.5&Aee=9.5&Aei=11.5&Aie=7.5&Aii=2.5&Ge=1.25&Gi=0.35&dt=0.1&seed=42&activity_scale=simulation") do ws
            hello = String(HTTP.WebSockets.receive(ws))
            frame = String(HTTP.WebSockets.receive(ws))
            @test occursin("\"type\":\"hello\"", hello)
            @test occursin("\"type\":\"frame\"", frame)
            @test occursin("\"coupling\":\"overlap\"", hello)
            @test occursin("\"Se\":1.5", hello)
            @test occursin("\"Si\":4.5", hello)
            @test occursin("\"Aee\":9.5", hello)
            @test occursin("\"Aei\":11.5", hello)
            @test occursin("\"Aie\":7.5", hello)
            @test occursin("\"Aii\":2.5", hello)
            @test occursin("\"Ge\":1.25", hello)
            @test occursin("\"Gi\":0.35", hello)
            @test occursin("\"dt\":0.1", hello)
            @test occursin("\"seed\":42", hello)
            @test occursin("\"activityScale\":\"simulation\"", hello)
            @test occursin("\"retinalResolution\":51", hello)
            @test occursin("\"retinalRendering\":\"mapped\"", hello)
            @test occursin("\"cols\":50", frame)
            @test occursin("\"retinalRows\":51", frame)
            @test occursin("\"retinalCols\":51", frame)
            @test occursin("\"phaseCount\":1250", frame)
            @test occursin("\"phaseEData\":", frame)
            @test occursin("\"phaseIData\":", frame)
            @test occursin("\"stepInterval\":1", frame)
            @test !occursin("\"skipInterval\"", frame)
        end

        HTTP.WebSockets.open("ws://$address/stream?backend=cpu&N=9&retinal_resolution=51&retinal_rendering=interpolated&fps=10&speed=0&max_frames=1") do ws
            hello = String(HTTP.WebSockets.receive(ws))
            frame = String(HTTP.WebSockets.receive(ws))
            @test occursin("\"retinalResolution\":51", hello)
            @test occursin("\"retinalRendering\":\"interpolated\"", hello)
            @test occursin("\"rows\":9", frame)
            @test occursin("\"cols\":9", frame)
            @test occursin("\"retinalRows\":9", frame)
            @test occursin("\"retinalCols\":9", frame)
        end
    finally
        close(server)
    end
end
