# Real-time Strobe Hallucination Simulator

A Julia web application for exploring the Rule-Ermentrout-Stroffegen neural
field model of flicker-induced geometric hallucinations. Julia runs the model
on the CPU or Apple Metal GPU and streams cortical, retinal, and phase-plane
views to a local browser over WebSockets.

The model is based on:

> Rule, M., Stoffregen, M., and Ermentrout, B. (2011). A Model for the Origin
> and Properties of Flicker-Induced Geometric Phosphenes. PLoS Computational
> Biology, 7(9), e1002158.

## Safety

This application intentionally presents flickering visual stimuli. Do not use
it if you may be sensitive to flashing light or have a history of
photosensitive seizures.

## Requirements

- Julia 1.10 or newer.
- macOS on Apple Silicon for the Metal backend.
- Any platform supported by Julia and FFTW for the CPU backend.

## Install

```bash
git clone https://github.com/ejgrove/rse_model.git
cd rse_model
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`Project.toml` and `Manifest.toml` define the complete Julia environment. No
Python or frontend build environment is required.

## Run

```bash
julia --project=. scripts/serve_applet.jl
```

Open `http://127.0.0.1:8088/`. Available server options are:

```text
--host HOST    Interface to bind (default: 127.0.0.1)
--port PORT    Port to bind; use 0 for an available port (default: 8088)
--open         Open the app in the default macOS browser
```

The simulation starts automatically. `Space` pauses or resumes it, and `Enter`
resets the model with the current parameters. Seeds are integers from 1 to 999.
When `Randomize seed on restart` is selected, Reset and Enter generate a
new seed; choosing a preset first runs its saved seed from `data/rse_params.xlsx`.

## App Behavior

- `FPS` is the target number of visualization frames delivered each second.
- `Visualization speed` is relative to wall time: `1` is real time, `0.5` is
  50%, and `2` is 200%.
- `Max speed` adaptively fills each frame interval with as many integration
  steps as the selected backend can complete.
- `Simulation time` is the accumulated model time, and `Real-time (x)` is its
  measured rate relative to browser wall time.
- `Resolution (contours)` controls display color quantization without changing
  model values.
- `Simulation min/max` uses frame-local color bounds for the first 500 ms, then
  fixes the display range from extrema accumulated after that startup period.
- `Aee`, `Aei`, `Aie`, and `Aii` set the recurrent synaptic-weight magnitudes;
  `Ge` and `Gi` scale the strobe input to the excitatory and inhibitory fields.
- `Retinal rendering` selects browser-interpolated output for speed or direct
  high-resolution mapping for greater coordinate precision.
- `Retinal resolution` sets the displayed square grid, defaulting to 321 x 321
  pixels, without changing the neural-field grid or its dynamics.
- Square fields support periodic, edge, zero, and partial-reflection boundaries
  independently along X and Y.
- Double-sech V1 fields use one boundary mode over the masked geometry.
- Coupling can be disabled, represented as two disconnected hemispheres, or
  applied through the overlap region.

The app includes cortical and retinal activity maps plus stimulus, kernel,
neural-field, and phase-plane analysis panes. The double-sech geometry uses the
dipole mapping described by Schira et al. (2010). Retinal projection indices
and bilinear weights are cached once per stream. The default interpolated mode
projects at field resolution and uses browser canvas interpolation for the
larger display; mapped mode samples the transform directly at display
resolution.

## Project Structure

```text
scripts/serve_applet.jl   Command-line server entry point
src/Applet.jl             Live stream runtime, protocol, and HTTP server
src/Model.jl              Neural dynamics and CPU/Metal convolution backends
src/Geometry.jl           Square and dipole double-sech field geometry
src/RetinalMapping.jl     Cortical-to-retinal transforms
src/Kernels.jl            Gaussian kernel construction
src/Grid.jl               Odd and FFT-friendly grid sizing
src/Params.jl             Model parameter definition
data/rse_params.xlsx      Source table for the named parameter presets
web/index.html            App document structure
web/styles.css            App visual design
web/app.js                Browser rendering and controls
test/runtests.jl          Model, backend, protocol, and server tests
```

See [docs/architecture.md](docs/architecture.md) for the runtime data flow and
[docs/web-design-principles.md](docs/web-design-principles.md) before changing
the interface.

## Test

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite checks the field geometry, stimulus, coupling, retinal mapping, CPU
and Metal convolution paths, timing behavior, static assets, WebSocket frame
protocol, and local HTTP server.

## License

MIT. See [LICENSE](LICENSE).
