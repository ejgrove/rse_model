# Web Design Principles

This guide is the reference for future changes to the real-time applet UI.

## Design Goals

- Keep simulation controls visible, compact, and legible while preserving the visualizer as the main focus.
- Show only controls that are relevant to the active parameter set; hide or collapse expert controls until needed.
- Use text that describes the underlying simulation quantity, not implementation shorthand.
- Prevent label overlap by giving plot annotations predictable margins and by keeping labels short.

## Typography

- Primary UI font: IBM Plex Sans, with Aptos and Helvetica Neue as fallbacks.
- Monospace font: IBM Plex Mono for printed parameter JSON and technical readouts.
- Page title: 24-33 px, 800 weight, tight line-height with enough spacing for readability.
- Section headings: 11 px, 800 weight, title case.
- Subsection headings: 10 px, 800 weight, sentence/title case depending on length.
- Control labels: 10 px, 500-600 weight, sentence case.
- Plot titles: 16 px, 800 weight, title case.
- Plot annotations: 10-11 px, 700-800 weight, short and unobtrusive.
- Metric labels: 9 px, regular/medium weight; metric values: 12-15 px, bold.

## Capitalization

- Use title case for main groups, plot titles, and pane titles: `Model Parameters`, `Visual Field`, `Phase Plane`.
- Use sentence case for control labels, help text, status text, and notes: `Activity scale`, `moving 0.5 s window`.
- Keep scientific symbols, axis labels, and acronyms exact: `FPS`, `GPU`, `FFT`, `E/I`, `V1`, `X`, `Y`, `dt`, `sigma`.
- Use all-caps only for keyboard keycaps and scientific/acronym tokens, not for whole headings.
- Dropdown options should use sentence case unless the option is an acronym or code-like parameter value.

## Color

- Primary ink: deep blue-gray for titles and important labels.
- Muted text: blue-gray for labels, notes, and low-priority metadata.
- Accent teal: active controls, section accents, positive/primary curves.
- Accent amber: secondary highlights and inhibitory/reference curves.
- Avoid using color alone to encode meaning; pair important colors with labels.

## Layout

- The left panel should fit common laptop heights with little or no scrolling in its default collapsed state.
- The app shell should scale as one two-pane object with a bounded width, currently 1400-2000 px; below the minimum, preserve the pane proportions and let the page scroll instead of compressing the layout.
- Keep the parameter pane fixed at its maximum comfortable width, currently 490 px. Let the visualization pane take the remaining shell width, and when the browser is wider than the shell maximum, increase outside margins rather than enlarging indefinitely.
- Let the cortical and visual-field plots fill the visualization pane proportionally so their size follows the two-pane shell, not independent per-plot clamps.
- Keep high-frequency controls expanded: visualization, strobe, and neural field parameters.
- Collapse advanced or situational controls: implementation, boundary, coupling, and presets.
- Use consistent two- or three-column grids, aligned labels, and stable spacing.
- Keep plot titles centered above each plot.
- Place visual annotations close to the corresponding plot edge and outside the activity pixels when possible.
- Keep colorbars and legends in a separate side rail so they do not change the plot frame size or title centering.
- Keep display-only controls such as contours live; they should redraw the cached frame rather than reset the simulation.

## Copy Rules

- Prefer descriptive labels over internal names: `Time step (ms)` instead of `dt (ms)`, `Sheet size` instead of `N`.
- Include units in parentheses in the label: `Period (ms)`, `Duty cycle (%)`.
- Use `Visualization speed` for the simulation-playback speed control, and `FPS` for the stream frame rate.
- In the coupling section, use `Type` for the coupling mode and `Strength` for the overlap coupling gain.
- Keep selected-parameter preset buttons numeric; show the preset title in the adjacent title line on hover, focus, or selection.
- Keep buttons short and action-oriented: `Pause`, `Reset`, `Print settings`.
- Status text should be sentence case and should describe current state, not implementation internals unless useful.
- If a control is hidden because it is not relevant, avoid mentioning it elsewhere on the page.

## Change Checklist

- Does the label use the same term everywhere it appears, including printed settings?
- Is the label visible only when the corresponding parameter applies to the current geometry, boundary, or coupling mode?
- Is the line short enough to fit the control column without wrapping into adjacent values?
- Are titles, pane names, controls, metric labels, and plot annotations following their assigned font sizes?
- Is all-caps limited to keycaps, scientific acronyms, or short units?
- Do plot labels have enough margin from the activity image, color bar, and neighboring panels?
