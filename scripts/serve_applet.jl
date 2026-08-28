#!/usr/bin/env julia

using ArgParse
using RSEModel

# Keep command-line concerns in this entry point; simulation and server logic
# live in the package so they remain testable without spawning a process.
function _settings()
    settings = ArgParseSettings(description="Serve the real-time RSE model applet.")

    @add_arg_table! settings begin
        "--host"
            help = "Host interface to bind."
            arg_type = String
            default = "127.0.0.1"
        "--port"
            help = "Port to bind. Use 0 to choose an available port."
            arg_type = Int
            default = 8088
        "--open"
            help = "Open the applet URL in the default browser on macOS."
            action = :store_true
    end

    return settings
end

function main(argv=ARGS)
    args = parse_args(argv, _settings())
    server = serve_applet_async(host=args["host"], port=args["port"], verbose=true)
    url = applet_url(server, args["host"])

    if args["open"]
        try
            run(`open $url`)
        catch err
            @warn "Could not open browser automatically." exception=(err, catch_backtrace())
        end
    end

    wait(server)
    return nothing
end

main()
