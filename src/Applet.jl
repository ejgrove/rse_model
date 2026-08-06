using Base64
using HTTP

Base.@kwdef struct LiveConfig
    N::Int = 105
    fast_n::Bool = true
    backend::Symbol = :metal
    convolution::Symbol = :auto
    A::Float32 = 0.7f0
    period::Float32 = 115.0f0
    Se::Float32 = 2.0f0
    Si::Float32 = 5.0f0
    seed::Union{Nothing,Int} = nothing
    target_fps::Int = 30
    speed::Float64 = 1.0
    gpu_threads::Int = 256
    kernel_cutoff::Float64 = 2.0
    max_frames::Int = 0
end

Base.@kwdef struct LiveFrame
    frame::Int
    N::Int
    t::Float64
    lo::Float32
    hi::Float32
    step_ms::Float64
    frame_ms::Float64
    ms_per_step::Float64
    realtime_x::Float64
    steps_per_frame::Int
    data::Vector{UInt8}
end

function normalize_live_config(config::LiveConfig)
    backend = config.backend == :gpu ? :metal : config.backend
    backend in (:cpu, :metal) || throw(ArgumentError("backend must be :cpu or :metal."))

    convolution = if config.convolution == :auto
        backend == :metal ? :separable : :fft
    else
        config.convolution
    end
    convolution in (:fft, :separable) || throw(ArgumentError("convolution must be :auto, :fft, or :separable."))
    backend == :metal || convolution == :fft ||
        throw(ArgumentError("The CPU live backend currently supports FFT convolution only."))

    target_fps = max(1, config.target_fps)
    N = config.fast_n ? next_fast_odd_size(config.N) : odd_positive_int(config.N)
    config.speed >= 0 || throw(ArgumentError("speed must be non-negative."))
    config.gpu_threads > 0 || throw(ArgumentError("gpu_threads must be positive."))
    config.kernel_cutoff > 0 || throw(ArgumentError("kernel_cutoff must be positive."))
    config.max_frames >= 0 || throw(ArgumentError("max_frames must be non-negative."))

    return LiveConfig(
        N=N,
        fast_n=config.fast_n,
        backend=backend,
        convolution=convolution,
        A=config.A,
        period=config.period,
        Se=config.Se,
        Si=config.Si,
        seed=config.seed,
        target_fps=target_fps,
        speed=config.speed,
        gpu_threads=config.gpu_threads,
        kernel_cutoff=config.kernel_cutoff,
        max_frames=config.max_frames,
    )
end

function _parse_bool(value::AbstractString, default::Bool)
    isempty(value) && return default
    key = lowercase(value)
    key in ("1", "true", "yes", "on") && return true
    key in ("0", "false", "no", "off") && return false
    return default
end

function _get(params, key, default)
    return get(params, key, default)
end

function _parse_int(params, key, default)
    value = _get(params, key, nothing)
    value === nothing && return default
    return parse(Int, value)
end

function _parse_float(params, key, default)
    value = _get(params, key, nothing)
    value === nothing && return default
    return parse(Float64, value)
end

function _parse_float32(params, key, default)
    return Float32(_parse_float(params, key, Float64(default)))
end

function _parse_symbol(params, key, default)
    value = _get(params, key, nothing)
    value === nothing && return default
    return Symbol(lowercase(value))
end

function live_config_from_query(params::AbstractDict{String,String})
    seed_value = _get(params, "seed", "")
    seed = isempty(seed_value) ? nothing : parse(Int, seed_value)

    return normalize_live_config(LiveConfig(
        N=_parse_int(params, "N", 105),
        fast_n=_parse_bool(_get(params, "fast_n", "true"), true),
        backend=_parse_symbol(params, "backend", :metal),
        convolution=_parse_symbol(params, "conv", :auto),
        A=_parse_float32(params, "A", 0.7f0),
        period=_parse_float32(params, "T", 115.0f0),
        Se=_parse_float32(params, "Se", 2.0f0),
        Si=_parse_float32(params, "Si", 5.0f0),
        seed=seed,
        target_fps=_parse_int(params, "fps", 30),
        speed=_parse_float(params, "speed", 1.0),
        gpu_threads=_parse_int(params, "gpu_threads", 256),
        kernel_cutoff=_parse_float(params, "kernel_cutoff", 2.0),
        max_frames=_parse_int(params, "max_frames", 0),
    ))
end

function _steps_per_frame(config::LiveConfig, p::ModelParams)
    target_frame_ms = 1000 / config.target_fps
    return max(1, round(Int, target_frame_ms / p.dt))
end

function _activity_bytes(activity::AbstractMatrix)
    rows, cols = size(activity)
    lo = Float32(minimum(activity))
    hi = Float32(maximum(activity))
    scale = hi == lo ? 0.0f0 : 255.0f0 / (hi - lo)
    bytes = Vector{UInt8}(undef, rows * cols)

    @inbounds for row in 1:rows, col in 1:cols
        value = hi == lo ? 0.0f0 : (Float32(activity[row, col]) - lo) * scale
        bytes[(row - 1) * cols + col] = UInt8(round(Int, clamp(value, 0.0f0, 255.0f0)))
    end

    return bytes, lo, hi
end

function _make_live_frame(
    activity::AbstractMatrix,
    frame_idx::Integer,
    t,
    step_ms::Float64,
    frame_start_ns::UInt64,
    steps_per_frame::Integer,
    p::ModelParams,
)
    bytes, lo, hi = _activity_bytes(activity)
    frame_ms = (time_ns() - frame_start_ns) / 1e6
    sim_ms = steps_per_frame * Float64(p.dt)
    ms_per_step = step_ms / steps_per_frame
    realtime_x = step_ms == 0 ? Inf : sim_ms / step_ms

    return LiveFrame(
        frame=Int(frame_idx),
        N=size(activity, 1),
        t=Float64(t),
        lo=lo,
        hi=hi,
        step_ms=step_ms,
        frame_ms=frame_ms,
        ms_per_step=ms_per_step,
        realtime_x=realtime_x,
        steps_per_frame=steps_per_frame,
        data=bytes,
    )
end

function _throttle!(stream_start_ns::UInt64, config::LiveConfig, sim_elapsed_ms::Float64)
    config.speed <= 0 && return
    target_elapsed = sim_elapsed_ms / (1000 * config.speed)
    actual_elapsed = (time_ns() - stream_start_ns) / 1e9
    delay = target_elapsed - actual_elapsed
    delay > 0.001 && sleep(delay)
    return
end

function _stream_cpu_frames(callback::Function, config::LiveConfig)
    p = ModelParams{Float32}()
    rng = _rng(config.seed)
    Ue = rand(rng, Float32, config.N, config.N)
    Ui = rand(rng, Float32, config.N, config.N)
    Ke = generate_gaussian_kernel(config.Se, config.N; dtype=Float32)
    Ki = generate_gaussian_kernel(config.Si, config.N; dtype=Float32)
    excitatory_convolver = FFTConvolver(Ke, Ue)
    inhibitory_convolver = FFTConvolver(Ki, Ui)
    Uec = similar(Ue)
    Uic = similar(Ui)
    noise = Array{Float32}(undef, 2, config.N, config.N)

    steps_per_frame = _steps_per_frame(config, p)
    stream_start = time_ns()
    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        frame_idx += 1
        frame_start = time_ns()
        step_start = time_ns()
        for _ in 1:steps_per_frame
            t_step = Float32(step_idx) * p.dt
            randn!(rng, noise)
            _step!(
                Ue,
                Ui,
                Uec,
                Uic,
                excitatory_convolver,
                inhibitory_convolver,
                noise,
                config.A,
                config.period,
                t_step,
                p,
            )
            step_idx += 1
        end
        step_ms = (time_ns() - step_start) / 1e6
        activity = abs.(Ue .- Ui)
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(activity, frame_idx, t, step_ms, frame_start, steps_per_frame, p)
        callback(frame) === false && break
        _throttle!(stream_start, config, frame_idx * steps_per_frame * Float64(p.dt))
    end

    return frame_idx
end

function _stream_metal_frames(callback::Function, config::LiveConfig)
    Metal.functional() || throw(ErrorException("Metal.jl is not functional on this machine."))
    p = ModelParams{Float32}()
    config.seed === nothing || Metal.seed!(config.seed)

    Ue = Metal.rand(Float32, config.N, config.N)
    Ui = Metal.rand(Float32, config.N, config.N)
    excitatory_convolver = if config.convolution == :fft
        Ke = generate_gaussian_kernel(config.Se, config.N; dtype=Float32)
        MetalFFTConvolver(Ke, Ue)
    else
        MetalSeparableConvolver(config.Se, Ue; cutoff=config.kernel_cutoff)
    end
    inhibitory_convolver = if config.convolution == :fft
        Ki = generate_gaussian_kernel(config.Si, config.N; dtype=Float32)
        MetalFFTConvolver(Ki, Ui)
    else
        MetalSeparableConvolver(config.Si, Ui; cutoff=config.kernel_cutoff)
    end

    Uec = similar(Ue)
    Uic = similar(Ui)
    noise_E = similar(Ue)
    noise_I = similar(Ui)
    cortical_gpu = similar(Ue)

    steps_per_frame = _steps_per_frame(config, p)
    stream_start = time_ns()
    step_idx = 0
    frame_idx = 0

    while config.max_frames == 0 || frame_idx < config.max_frames
        frame_idx += 1
        frame_start = time_ns()
        step_start = time_ns()
        for _ in 1:steps_per_frame
            t_step = Float32(step_idx) * p.dt
            Metal.randn!(noise_E)
            Metal.randn!(noise_I)
            if config.convolution == :fft
                _metal_step!(
                    Ue,
                    Ui,
                    Uec,
                    Uic,
                    excitatory_convolver,
                    inhibitory_convolver,
                    noise_E,
                    noise_I,
                    config.A,
                    config.period,
                    t_step,
                    p,
                    config.gpu_threads,
                )
            else
                _metal_step_separable!(
                    Ue,
                    Ui,
                    Uec,
                    Uic,
                    excitatory_convolver,
                    inhibitory_convolver,
                    noise_E,
                    noise_I,
                    config.A,
                    config.period,
                    t_step,
                    p,
                    config.gpu_threads,
                )
            end
            step_idx += 1
        end
        Metal.synchronize()
        step_ms = (time_ns() - step_start) / 1e6

        cortical_gpu .= abs.(Ue .- Ui)
        Metal.synchronize()
        activity = Array(cortical_gpu)
        t = Float32(step_idx) * p.dt
        frame = _make_live_frame(activity, frame_idx, t, step_ms, frame_start, steps_per_frame, p)
        callback(frame) === false && break
        _throttle!(stream_start, config, frame_idx * steps_per_frame * Float64(p.dt))
    end

    return frame_idx
end

function stream_live_frames(callback::Function, config::LiveConfig=LiveConfig())
    normalized = normalize_live_config(config)
    if normalized.backend == :metal
        return _stream_metal_frames(callback, normalized)
    else
        return _stream_cpu_frames(callback, normalized)
    end
end

function _json_string(value)
    escaped = replace(
        string(value),
        "\\" => "\\\\",
        "\"" => "\\\"",
        "\n" => "\\n",
        "\r" => "\\r",
    )
    return "\"$(escaped)\""
end

function _json_number(value; digits=4)
    if value isa AbstractFloat
        isfinite(value) || return "null"
        return string(round(value; digits=digits))
    end
    return string(value)
end

function _hello_json(config::LiveConfig)
    return string(
        "{\"type\":\"hello\"",
        ",\"N\":", config.N,
        ",\"backend\":", _json_string(config.backend),
        ",\"conv\":", _json_string(config.convolution),
        ",\"fps\":", config.target_fps,
        ",\"speed\":", _json_number(config.speed),
        ",\"A\":", _json_number(config.A),
        ",\"T\":", _json_number(config.period),
        ",\"Se\":", _json_number(config.Se),
        ",\"Si\":", _json_number(config.Si),
        ",\"kernelCutoff\":", _json_number(config.kernel_cutoff),
        "}",
    )
end

function _frame_json(frame::LiveFrame)
    return string(
        "{\"type\":\"frame\"",
        ",\"frame\":", frame.frame,
        ",\"N\":", frame.N,
        ",\"t\":", _json_number(frame.t),
        ",\"min\":", _json_number(frame.lo; digits=6),
        ",\"max\":", _json_number(frame.hi; digits=6),
        ",\"stepMs\":", _json_number(frame.step_ms),
        ",\"frameMs\":", _json_number(frame.frame_ms),
        ",\"msPerStep\":", _json_number(frame.ms_per_step; digits=5),
        ",\"realtimeX\":", _json_number(frame.realtime_x),
        ",\"stepsPerFrame\":", frame.steps_per_frame,
        ",\"data\":", _json_string(base64encode(frame.data)),
        "}",
    )
end

function _done_json(frames)
    return "{\"type\":\"done\",\"frames\":$(frames)}"
end

function _error_json(err)
    return string("{\"type\":\"error\",\"message\":", _json_string(sprint(showerror, err)), "}")
end

function _safe_ws_send(ws, message)
    try
        HTTP.WebSockets.send(ws, message)
        return true
    catch err
        if err isa HTTP.WebSockets.WebSocketError || err isa EOFError || err isa IOError
            return false
        end
        rethrow()
    end
end

function _stream_websocket(ws, params)
    try
        config = live_config_from_query(params)
        _safe_ws_send(ws, _hello_json(config)) || return
        frames = stream_live_frames(config) do frame
            _safe_ws_send(ws, _frame_json(frame))
        end
        _safe_ws_send(ws, _done_json(frames))
    catch err
        _safe_ws_send(ws, _error_json(err))
    end
    return
end

function _request_uri(stream)
    return HTTP.URI(String(stream.message.target))
end

function _write_response(stream, status::Integer, content_type::AbstractString, body::AbstractString)
    HTTP.setstatus(stream, status)
    HTTP.setheader(stream, "Content-Type", content_type)
    HTTP.setheader(stream, "Cache-Control", "no-store")
    HTTP.startwrite(stream)
    write(stream, body)
    return
end

function _handle_applet_stream(stream)
    uri = _request_uri(stream)
    if HTTP.WebSockets.isupgrade(stream.message) && uri.path == "/stream"
        HTTP.WebSockets.upgrade(stream; check_origin=nothing) do ws
            _stream_websocket(ws, HTTP.queryparams(uri))
        end
    elseif uri.path == "/" || uri.path == "/index.html"
        _write_response(stream, 200, "text/html; charset=utf-8", APPLET_HTML)
    elseif uri.path == "/health"
        _write_response(stream, 200, "text/plain; charset=utf-8", "ok")
    else
        _write_response(stream, 404, "text/plain; charset=utf-8", "not found")
    end
    return
end

function _display_host(host)
    return host in ("0.0.0.0", "::") ? "127.0.0.1" : host
end

function applet_url(server, host)
    return "http://$(_display_host(host)):$(HTTP.port(server))/"
end

function serve_applet_async(;
    host::AbstractString="127.0.0.1",
    port::Integer=8088,
    verbose::Bool=true,
)
    server = HTTP.listen!(host, port; listenany=true) do stream
        _handle_applet_stream(stream)
    end
    if verbose
        println("RSE real-time applet: ", applet_url(server, host))
        println("Press Ctrl-C to stop the server.")
    end
    return server
end

function serve_applet(;
    host::AbstractString="127.0.0.1",
    port::Integer=8088,
    verbose::Bool=true,
)
    server = serve_applet_async(host=host, port=port, verbose=verbose)
    wait(server)
    return server
end

const APPLET_HTML = raw"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RSE Real-Time Viewer</title>
  <style>
    :root {
      --bg: #071013;
      --panel: rgba(250, 244, 225, 0.08);
      --panel-strong: rgba(250, 244, 225, 0.14);
      --ink: #f7f0d3;
      --muted: #adbea9;
      --accent: #ffb84d;
      --accent-2: #5ee6b5;
      --danger: #ff6d6d;
      --line: rgba(247, 240, 211, 0.16);
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
      font-family: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      min-height: 100vh;
      background:
        radial-gradient(circle at 15% 10%, rgba(94, 230, 181, 0.16), transparent 28rem),
        radial-gradient(circle at 90% 15%, rgba(255, 184, 77, 0.16), transparent 30rem),
        linear-gradient(135deg, #071013 0%, #122524 52%, #211b12 100%);
    }

    main {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 28px;
      display: grid;
      grid-template-columns: 330px 1fr;
      gap: 22px;
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 54px);
      letter-spacing: -0.06em;
      line-height: 0.95;
    }

    h2 {
      margin: 0;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
    }

    .panel, .stage {
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(22px);
      border-radius: 26px;
      box-shadow: var(--shadow);
    }

    .panel {
      padding: 20px;
      align-self: start;
      position: sticky;
      top: 18px;
    }

    .subtitle {
      color: var(--muted);
      margin: 0 0 22px;
      line-height: 1.45;
      font-size: 14px;
    }

    .control-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.04em;
    }

    input, select, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      color: var(--ink);
      background: rgba(0, 0, 0, 0.22);
      padding: 10px 11px;
      font: inherit;
      outline: none;
    }

    input[type="checkbox"] {
      width: auto;
      accent-color: var(--accent);
    }

    .check-row {
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 10px 0 2px;
      color: var(--ink);
    }

    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 18px;
    }

    button {
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent), #ff7e57);
      color: #160f07;
      border: 0;
      font-weight: 800;
      letter-spacing: 0.02em;
    }

    button.secondary {
      color: var(--ink);
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--line);
    }

    .status {
      margin-top: 18px;
      padding: 12px;
      border-radius: 18px;
      background: rgba(0, 0, 0, 0.2);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      min-height: 44px;
    }

    .stage {
      padding: 22px;
      display: grid;
      gap: 18px;
      min-width: 0;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      border: 1px solid var(--line);
      background: rgba(0, 0, 0, 0.22);
      border-radius: 18px;
      padding: 12px;
      min-width: 0;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: clamp(18px, 2.3vw, 30px);
      letter-spacing: -0.05em;
      white-space: nowrap;
    }

    .views {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      min-width: 0;
    }

    .view {
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(0, 0, 0, 0.24);
      padding: 14px;
      min-width: 0;
    }

    .view-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 10px;
    }

    .view-title {
      font-size: 14px;
      font-weight: 800;
      letter-spacing: 0.02em;
    }

    .view-note {
      color: var(--muted);
      font-size: 12px;
    }

    canvas {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 18px;
      image-rendering: pixelated;
      background: #060708;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);
    }

    .legend {
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, #0d0887, #5403a0, #8b0aa5, #b93289, #db5c68, #f48849, #feba2c, #f0f921);
      border: 1px solid var(--line);
    }

    @media (max-width: 940px) {
      main { grid-template-columns: 1fr; padding: 18px; }
      .panel { position: static; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .views { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h2>RSE Model</h2>
      <h1>Real-time field viewer</h1>
      <p class="subtitle">Streams the Julia simulation into the browser. GPU mode keeps the model on Metal and sends only display frames.</p>
      <div class="control-grid">
        <label>N<input id="n" type="number" min="5" step="2" value="101"></label>
        <label>FPS<input id="fps" type="number" min="1" max="60" step="1" value="30"></label>
        <label>Backend<select id="backend"><option value="metal">metal</option><option value="cpu">cpu</option></select></label>
        <label>Convolution<select id="conv"><option value="auto">auto</option><option value="separable">separable</option><option value="fft">fft</option></select></label>
        <label>Speed<select id="speed"><option value="1">1x real time</option><option value="0.5">0.5x</option><option value="2">2x</option><option value="0">max</option></select></label>
        <label>Kernel cutoff<input id="kernelCutoff" type="number" min="0.5" max="6" step="0.25" value="2"></label>
        <label>A<input id="amp" type="number" min="0" step="0.05" value="0.7"></label>
        <label>T (ms)<input id="period" type="number" min="1" step="1" value="115"></label>
        <label>Se<input id="se" type="number" min="0.1" step="0.1" value="2"></label>
        <label>Si<input id="si" type="number" min="0.1" step="0.1" value="5"></label>
      </div>
      <label class="check-row"><input id="fastN" type="checkbox" checked> Snap to FFT-friendly odd N</label>
      <label>Seed<input id="seed" type="number" step="1" placeholder="optional"></label>
      <div class="button-row">
        <button id="start">Start stream</button>
        <button id="stop" class="secondary">Stop</button>
      </div>
      <div id="status" class="status">Idle. Start the stream when ready.</div>
    </section>

    <section class="stage">
      <div class="metrics">
        <div class="metric"><span>Sim time</span><strong id="simTime">0 ms</strong></div>
        <div class="metric"><span>Stream FPS</span><strong id="streamFps">0</strong></div>
        <div class="metric"><span>ms / step</span><strong id="msStep">0</strong></div>
        <div class="metric"><span>Real-time x</span><strong id="rtx">0</strong></div>
        <div class="metric"><span>Grid</span><strong id="gridN">-</strong></div>
      </div>
      <div class="views">
        <div class="view">
          <div class="view-head"><div class="view-title">Cortical sheet</div><div class="view-note" id="range">range -</div></div>
          <canvas id="cortical"></canvas>
        </div>
        <div class="view">
          <div class="view-head"><div class="view-title">Retinal view</div><div class="view-note">client-side log-polar map</div></div>
          <canvas id="retinal"></canvas>
        </div>
      </div>
      <div class="legend"></div>
    </section>
  </main>

  <script>
    const stops = [
      [13, 8, 135], [84, 3, 160], [139, 10, 165], [185, 50, 137],
      [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]
    ];

    const els = {
      n: document.getElementById("n"),
      fps: document.getElementById("fps"),
      backend: document.getElementById("backend"),
      conv: document.getElementById("conv"),
      speed: document.getElementById("speed"),
      kernelCutoff: document.getElementById("kernelCutoff"),
      amp: document.getElementById("amp"),
      period: document.getElementById("period"),
      se: document.getElementById("se"),
      si: document.getElementById("si"),
      fastN: document.getElementById("fastN"),
      seed: document.getElementById("seed"),
      start: document.getElementById("start"),
      stop: document.getElementById("stop"),
      status: document.getElementById("status"),
      simTime: document.getElementById("simTime"),
      streamFps: document.getElementById("streamFps"),
      msStep: document.getElementById("msStep"),
      rtx: document.getElementById("rtx"),
      gridN: document.getElementById("gridN"),
      range: document.getElementById("range"),
      cortical: document.getElementById("cortical"),
      retinal: document.getElementById("retinal")
    };

    let socket = null;
    let lastFrameAt = performance.now();
    let retinalMap = null;
    let retinalMapN = 0;

    function palette(v) {
      const t = Math.max(0, Math.min(1, v / 255));
      const scaled = t * (stops.length - 1);
      const idx = Math.min(Math.floor(scaled), stops.length - 2);
      const f = scaled - idx;
      const a = stops[idx], b = stops[idx + 1];
      return [
        Math.round((1 - f) * a[0] + f * b[0]),
        Math.round((1 - f) * a[1] + f * b[1]),
        Math.round((1 - f) * a[2] + f * b[2])
      ];
    }

    function setCanvasSize(canvas, n) {
      if (canvas.width !== n || canvas.height !== n) {
        canvas.width = n;
        canvas.height = n;
      }
    }

    function drawValues(canvas, values, n) {
      setCanvasSize(canvas, n);
      const ctx = canvas.getContext("2d");
      const image = ctx.createImageData(n, n);
      for (let i = 0; i < values.length; i++) {
        const [r, g, b] = palette(values[i]);
        const j = i * 4;
        image.data[j] = r;
        image.data[j + 1] = g;
        image.data[j + 2] = b;
        image.data[j + 3] = 255;
      }
      ctx.putImageData(image, 0, 0);
    }

    function wrapIndex(v, n) {
      return ((v % n) + n) % n;
    }

    function buildRetinalMap(n) {
      const total = n * n;
      const map = new Array(total);
      for (let row = 0; row < n; row++) {
        const y = -1 + 2 * row / Math.max(n - 1, 1);
        for (let col = 0; col < n; col++) {
          const x = -1 + 2 * col / Math.max(n - 1, 1);
          const r = Math.hypot(x, y);
          const theta = (Math.atan2(y, x) + 2 * Math.PI) % (2 * Math.PI);
          const xIn = Math.log(r + 1e-26) / (2 * Math.PI) * n;
          const yIn = theta / (2 * Math.PI) * n;
          const x0Raw = Math.floor(xIn);
          const y0Raw = Math.floor(yIn);
          const dx = xIn - x0Raw;
          const dy = yIn - y0Raw;
          const x0 = wrapIndex(x0Raw, n);
          const x1 = wrapIndex(x0Raw + 1, n);
          const y0 = wrapIndex(y0Raw, n);
          const y1 = wrapIndex(y0Raw + 1, n);
          map[row * n + col] = { x0, x1, y0, y1, dx, dy };
        }
      }
      retinalMap = map;
      retinalMapN = n;
    }

    function drawRetinal(canvas, values, n) {
      if (!retinalMap || retinalMapN !== n) buildRetinalMap(n);
      const mapped = new Uint8Array(values.length);
      for (let i = 0; i < mapped.length; i++) {
        const m = retinalMap[i];
        const v00 = values[m.y0 * n + m.x0];
        const v01 = values[m.y0 * n + m.x1];
        const v10 = values[m.y1 * n + m.x0];
        const v11 = values[m.y1 * n + m.x1];
        const top = (1 - m.dx) * v00 + m.dx * v01;
        const bottom = (1 - m.dx) * v10 + m.dx * v11;
        mapped[i] = Math.max(0, Math.min(255, Math.round((1 - m.dy) * top + m.dy * bottom)));
      }
      drawValues(canvas, mapped, n);
    }

    function decodeFrame(data) {
      const binary = atob(data);
      const values = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) values[i] = binary.charCodeAt(i);
      return values;
    }

    function streamParams() {
      const params = new URLSearchParams();
      params.set("N", els.n.value);
      params.set("fps", els.fps.value);
      params.set("backend", els.backend.value);
      params.set("conv", els.conv.value);
      params.set("speed", els.speed.value);
      params.set("kernel_cutoff", els.kernelCutoff.value);
      params.set("A", els.amp.value);
      params.set("T", els.period.value);
      params.set("Se", els.se.value);
      params.set("Si", els.si.value);
      params.set("fast_n", els.fastN.checked ? "true" : "false");
      if (els.seed.value.trim()) params.set("seed", els.seed.value.trim());
      return params;
    }

    function startStream() {
      stopStream();
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${location.host}/stream?${streamParams().toString()}`;
      socket = new WebSocket(url);
      els.status.textContent = "Connecting...";
      lastFrameAt = performance.now();

      socket.onopen = () => {
        els.status.textContent = "Streaming. Close or press Stop to restart with new parameters.";
      };

      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "hello") {
          els.gridN.textContent = String(msg.N);
          els.status.textContent = `Streaming ${msg.backend}/${msg.conv} at target ${msg.fps} fps.`;
          return;
        }
        if (msg.type === "done") {
          els.status.textContent = `Stream finished after ${msg.frames} frames.`;
          return;
        }
        if (msg.type === "error") {
          els.status.textContent = `Stream error: ${msg.message}`;
          stopStream();
          return;
        }
        if (msg.type !== "frame") return;

        const now = performance.now();
        const observedFps = 1000 / Math.max(1, now - lastFrameAt);
        lastFrameAt = now;
        const values = decodeFrame(msg.data);
        drawValues(els.cortical, values, msg.N);
        drawRetinal(els.retinal, values, msg.N);
        els.simTime.textContent = `${msg.t.toFixed(1)} ms`;
        els.streamFps.textContent = observedFps.toFixed(1);
        els.msStep.textContent = msg.msPerStep.toFixed(3);
        els.rtx.textContent = msg.realtimeX.toFixed(2);
        els.gridN.textContent = String(msg.N);
        els.range.textContent = `${msg.min.toFixed(3)} to ${msg.max.toFixed(3)}`;
      };

      socket.onclose = () => {
        els.status.textContent = "Stopped.";
        socket = null;
      };

      socket.onerror = () => {
        els.status.textContent = "Stream error. Check the Julia terminal for details.";
      };
    }

    function stopStream() {
      if (socket) {
        socket.close();
        socket = null;
      }
    }

    els.start.addEventListener("click", startStream);
    els.stop.addEventListener("click", stopStream);
    startStream();
  </script>
</body>
</html>
"""
