# Web Application Architecture

This document defines the product boundary and runtime organization of the RSE
web application. New batch tools, notebooks, or alternate language ports should
live in separate repositories rather than expanding this application package.

## Runtime Flow

1. `scripts/serve_applet.jl` parses only host, port, and browser-launch options.
2. `serve_applet_async` serves the three files in `web/` and accepts a WebSocket
   connection at `/stream`.
3. Query parameters are parsed into `LiveConfig` and normalized before any
   model state is allocated.
4. The selected CPU or Metal stream loop owns the neural-field state for that
   WebSocket connection.
5. The loop advances the model by the current step interval, applies a cached
   retinal projection plan, creates the display payloads, and sends one encoded
   `LiveFrame`.
6. `web/app.js` decodes the payload, applies display-only contour quantization,
   optionally interpolates the retinal image to its display resolution, draws
   the canvases, and measures actual FPS and simulation speed in the browser.
7. Visualization-only updates are sent over the existing socket. Model,
   geometry, boundary, and coupling changes reset the stream.

## Model Layers

- `Params.jl` contains the fixed neural-field parameter schema.
- `Grid.jl` owns odd-size validation and FFT-friendly grid selection.
- `Kernels.jl` creates full two-dimensional FFT kernels and truncated
  one-dimensional separable kernels.
- `Geometry.jl` creates square or masked double-sech lattices and performs the
  double-sech inverse map.
- `RetinalMapping.jl` caches and applies the square-sheet log-polar inverse map
  using wrapped cortical-grid sampling.
- `Model.jl` contains the shared stimulus and neural dynamics, CPU FFT path,
  paired Metal convolution path, masks, and hemisphere coupling kernels.
- `Applet.jl` owns live allocation, frame timing, serialization, WebSockets,
  and HTTP routing. It should not contain browser layout or styling.

## Timing Semantics

For a requested visualization rate `fps`, speed factor `v`, and time step `dt`
in milliseconds, the normal step interval is approximately:

```text
steps_per_frame = round((1000 / fps) * v / dt)
```

The interval is clamped to at least one step. This creates a minimum selectable
speed of `dt * fps / 1000`. Max-speed mode estimates step cost and non-step frame
cost independently, then adapts the batch to use 95% of the next frame budget.
The server still throttles frame delivery to the requested FPS.

The FPS and real-time metrics are browser measurements over a rolling one-second
window. They describe frames actually drawn and simulation time actually
observed, not an internal compute-only estimate.

## Boundary Conditions

- `periodic`: samples wrap to the opposite edge.
- `edge`: samples beyond the field use the nearest edge value.
- `zero`: samples beyond the field contribute zero.
- `partial_reflect`: samples mirror back into the field and the mirrored
  contribution is multiplied by the reflection gain.

Non-periodic boundaries require the Metal separable convolution path. The CPU
FFT path is periodic by construction.

## Coupling

Square overlap coupling mixes mirrored rows along the top and bottom midline
bands. Double-sech overlap coupling mixes mirrored nodes around the masked V1
border. Excitatory and inhibitory populations use the same linear mixing gain.
The two hemisphere fields remain separate model states and are combined only
for display and retinal mapping.

## Web Assets

The browser code has no build step:

- `index.html` contains semantic structure only.
- `styles.css` contains design tokens, controls, visualizations, and analysis
  pane layout.
- `app.js` contains cached DOM references, canvas rendering, controls, stream
  lifecycle, and event wiring in that order.

Keep model mathematics and simulation state out of the browser. Browser-side
math is limited to display aids such as the kernel preview, mean-field nullcline
approximation, and color contours.
