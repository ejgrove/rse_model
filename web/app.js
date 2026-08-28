// Static UI configuration and cached DOM references

const colorMaps = {
  plasma: [[13, 8, 135], [84, 3, 160], [139, 10, 165], [185, 50, 137], [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]],
  viridis: [[68, 1, 84], [70, 50, 126], [54, 92, 141], [39, 127, 142], [31, 161, 136], [74, 193, 109], [160, 218, 57], [253, 231, 37]],
  magma: [[0, 0, 4], [32, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [252, 137, 97], [254, 194, 135], [252, 253, 191]],
  inferno: [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]],
  cividis: [[0, 34, 78], [31, 59, 110], [61, 82, 128], [91, 105, 135], [121, 128, 137], [153, 153, 134], [188, 180, 120], [225, 210, 92], [255, 233, 69]],
  turbo: [[48, 18, 59], [50, 101, 192], [34, 170, 224], [52, 221, 164], [172, 244, 68], [251, 221, 59], [252, 132, 34], [180, 34, 15], [122, 4, 3]],
  nipy_spectral: [[0, 0, 0], [102, 0, 153], [0, 0, 205], [0, 148, 255], [0, 180, 0], [255, 238, 0], [255, 128, 0], [210, 0, 0], [255, 255, 255]],
  gray: [[0, 0, 0], [36, 36, 36], [73, 73, 73], [109, 109, 109], [146, 146, 146], [182, 182, 182], [219, 219, 219], [255, 255, 255]]
};

const els = {
  n: document.getElementById("n"),
  fps: document.getElementById("fps"),
  backend: document.getElementById("backend"),
  conv: document.getElementById("conv"),
  boundary: document.getElementById("boundary"),
  boundaryControl: document.getElementById("boundaryControl"),
  boundaryX: document.getElementById("boundaryX"),
  boundaryXControl: document.getElementById("boundaryXControl"),
  boundaryY: document.getElementById("boundaryY"),
  boundaryYControl: document.getElementById("boundaryYControl"),
  partialReflectControl: document.getElementById("partialReflectControl"),
  partialReflectStrength: document.getElementById("partialReflectStrength"),
  coupling: document.getElementById("coupling"),
  speed: document.getElementById("speed"),
  maxSpeed: document.getElementById("maxSpeed"),
  maxSpeedControl: document.getElementById("maxSpeedControl"),
  kernelCutoff: document.getElementById("kernelCutoff"),
  amp: document.getElementById("amp"),
  period: document.getElementById("period"),
  duty: document.getElementById("duty"),
  couplingStrength: document.getElementById("couplingStrength"),
  couplingStrengthControl: document.getElementById("couplingStrengthControl"),
  overlapRows: document.getElementById("overlapRows"),
  overlapRowsControl: document.getElementById("overlapRowsControl"),
  colorMap: document.getElementById("colorMap"),
  activityScale: document.getElementById("activityScale"),
  plotContours: document.getElementById("plotContours"),
  retinalResolution: document.getElementById("retinalResolution"),
  retinalRendering: document.getElementById("retinalRendering"),
  se: document.getElementById("se"),
  si: document.getElementById("si"),
  dt: document.getElementById("dt"),
  fieldGeometry: document.getElementById("fieldGeometry"),
  fieldDensity: document.getElementById("fieldDensity"),
  fieldDensityControl: document.getElementById("fieldDensityControl"),
  nControl: document.getElementById("nControl"),
  fastN: document.getElementById("fastN"),
  fastNControl: document.getElementById("fastNControl"),
  seed: document.getElementById("seed"),
  randomizeSeed: document.getElementById("randomizeSeed"),
  pausePlay: document.getElementById("pausePlay"),
  reset: document.getElementById("reset"),
  printParams: document.getElementById("printParams"),
  paramOutput: document.getElementById("paramOutput"),
  presetGrid: document.getElementById("presetGrid"),
  presetTitle: document.getElementById("presetTitle"),
  status: document.getElementById("status"),
  simTime: document.getElementById("simTime"),
  streamFps: document.getElementById("streamFps"),
  stepInterval: document.getElementById("stepInterval"),
  rtx: document.getElementById("rtx"),
  stimulusGraph: document.getElementById("stimulusGraph"),
  stimulusInfo: document.getElementById("stimulusInfo"),
  corticalFrame: document.getElementById("corticalFrame"),
  hemiLeft: document.getElementById("hemiLeft"),
  hemiRight: document.getElementById("hemiRight"),
  cortical: document.getElementById("cortical"),
  retinal: document.getElementById("retinal"),
  kernelGraph: document.getElementById("kernelGraph"),
  kernelInfo: document.getElementById("kernelInfo"),
  frameSelect: document.getElementById("frameSelect"),
  framePanels: Array.from(document.querySelectorAll(".frame-panel")),
  fieldGraph: document.getElementById("fieldGraph"),
  fieldInfo: document.getElementById("fieldInfo"),
  phaseGraph: document.getElementById("phaseGraph"),
  phaseInfo: document.getElementById("phaseInfo"),
  phaseIncludeAverage: document.getElementById("phaseIncludeAverage"),
  legend: document.getElementById("legend"),
  legendLow: document.getElementById("legendLow"),
  legendHigh: document.getElementById("legendHigh")
};

const presetDefaults = {
  fieldGeometry: "square",
  boundaryX: "periodic",
  boundaryY: "periodic",
  boundary: "edge",
  coupling: "off",
  fastN: false,
  kernelCutoff: 3,
  dt: 0.2,
  couplingStrength: 0.02,
  overlapRows: 6
};

// Source: data/rse_params.xlsx, Sheet1 rows 2-15.
const presetRows = [
  { label: "Stripes", values: { n: 121, amp: 0.7, period: 55, duty: 20.5, se: 2, si: 5, seed: 42 } },
  { label: "Dot square grid", values: { n: 81, amp: 0.7, period: 120, duty: 20.5, se: 2, si: 5, seed: 5 } },
  { label: "Dots", values: { n: 121, amp: 0.7, period: 115, duty: 20.5, se: 2, si: 5, seed: 4 } },
  { label: "Dots in line (square spiral?)", values: { n: 81, amp: 0.7, period: 120, duty: 20.5, se: 2, si: 5, seed: 4 } },
  { label: "Stripes", values: { n: 81, amp: 0.7, period: 120, duty: 50, se: 2, si: 5, seed: 42 } },
  { label: "Dots & zig-zags", values: { n: 81, amp: 0.5, period: 125, duty: 50, se: 2, si: 6, seed: 2 } },
  { label: "Unstable square grid", values: { n: 121, amp: 0.7, period: 115, duty: 50, se: 2.5, si: 6.5, seed: 2 } },
  { label: "Waves of dots", values: { n: 81, amp: 0.25, period: 105, duty: 50, se: 2.5, si: 6.875, seed: 42 } },
  { label: "Dots & stripes 1", values: { n: 121, amp: 0.25, period: 105, duty: 50, se: 2.5, si: 6.875, seed: 42 } },
  { label: "Dot honeycomb", values: { n: 121, amp: 0.5, period: 115, duty: 50, se: 2.5, si: 6.875, seed: 4 } },
  { label: "Cross hairs grid", values: { n: 81, amp: 0.5, period: 115, duty: 50, se: 2.5, si: 6.875, seed: 8 } },
  { label: "Two dot honeycomb", values: { n: 121, amp: 0.5, period: 125, duty: 50, se: 2.5, si: 6.875, seed: 8 } },
  { label: "Dots & stripes 2", values: { n: 121, amp: 0.5, period: 125, duty: 50, se: 2.5, si: 6.875, seed: 11 } },
  { label: "Rectangular checkerboard", values: { n: 81, amp: 0.25, period: 85, duty: 50, se: 2.5, si: 7.5, seed: 2 } }
];
const presets = Object.fromEntries(presetRows.map((preset, index) => [
  `p${index + 1}`,
  { label: preset.label, values: { ...presetDefaults, ...preset.values } }
]));
const presetKeys = Object.keys(presets);
const retinalInterpolationCanvas = document.createElement("canvas");

let socket = null;
let paused = false;
let resetting = false;
let activePresetKey = "p1";
let streamToken = 0;
let rateSamples = [];
let lastDisplayFrame = null;
let visualizationUpdateTimer = null;
let resetFallbackTimer = null;
let handledSpaceShortcut = false;
let streamStimulus = {
  A: Number(els.amp.value) || 0,
  period: Number(els.period.value) || 1,
  duty: Number(els.duty.value) || 50
};
const meanFieldParams = {
  Aee: 10.0,
  Aei: 12.0,
  Aie: 8.5,
  Aii: 3.0,
  He: 2.0,
  Hi: 3.5,
  Ge: 1.0,
  Gi: 0.0
};

// Formatting, color scales, and live visualization controls

function activeColorStops() {
  return colorMaps[els.colorMap.value] || colorMaps.plasma;
}

function colorStopString(stops = activeColorStops()) {
  return stops.map((rgb, idx) => `rgb(${rgb.join(",")}) ${(idx / (stops.length - 1)) * 100}%`).join(", ");
}

function activeContourCount() {
  return Math.max(2, Math.min(256, Math.round(Number(els.plotContours.value) || 256)));
}

function randomSeedValue() {
  const values = new Uint32Array(1);
  const limit = Math.floor(0x100000000 / 9999) * 9999;
  do {
    crypto.getRandomValues(values);
  } while (values[0] >= limit);
  return 1 + values[0] % 9999;
}

function normalizedSeedValue() {
  const value = Math.round(Number(els.seed.value));
  const seed = Number.isFinite(value) ? Math.max(1, Math.min(9999, value)) : randomSeedValue();
  els.seed.value = String(seed);
  return seed;
}

function prepareSeedForRestart(randomize = true) {
  const seed = randomize && els.randomizeSeed.checked ? randomSeedValue() : normalizedSeedValue();
  els.seed.value = String(seed);
  return seed;
}

function palette(v) {
  const stops = activeColorStops();
  const contours = activeContourCount();
  const normalized = Math.max(0, Math.min(1, v / 255));
  const t = contours >= 256 ? normalized : Math.round(normalized * (contours - 1)) / (contours - 1);
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

function legendColorString() {
  const contours = activeContourCount();
  if (contours > 32) return colorStopString();
  const segments = [];
  for (let index = 0; index < contours; index++) {
    const [r, g, b] = palette(255 * index / Math.max(contours - 1, 1));
    const start = 100 * index / contours;
    const end = 100 * (index + 1) / contours;
    segments.push(`rgb(${r},${g},${b}) ${start}%, rgb(${r},${g},${b}) ${end}%`);
  }
  return segments.join(", ");
}

function updateLegend() {
  els.legend.style.background = `linear-gradient(0deg, ${legendColorString()})`;
}

function formatSimTime(ms) {
  if (!Number.isFinite(ms)) return "0 ms";
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${ms.toFixed(1)} ms`;
}

function speedNumberString(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return "0.001";
  const text = numeric.toPrecision(4);
  return text.includes(".") ? text.replace(/\.?0+$/, "") : text;
}

function minimumSpeedValue() {
  const fps = Math.max(1, Number(els.fps.value) || 30);
  const dt = Math.max(0.000001, Number(els.dt.value) || 0.2);
  return dt * fps / 1000;
}

function currentSpeedValue() {
  if (els.maxSpeed.checked) return 0;
  return clampSpeedInputToMinimum(false);
}

function clampSpeedInputToMinimum(commit = false) {
  const minSpeed = minimumSpeedValue();
  const rawText = String(els.speed.value).trim();
  const raw = Number(els.speed.value);
  const belowMinimum = rawText === "" || !Number.isFinite(raw) || raw <= 0 || raw < minSpeed;
  if (!els.maxSpeed.checked && belowMinimum) {
    if (commit) {
      els.speed.value = speedNumberString(minSpeed);
    }
    return minSpeed;
  }
  return raw;
}

function syncSpeedControls(commit = true) {
  const minSpeed = minimumSpeedValue();
  const minText = speedNumberString(minSpeed);
  els.speed.min = minText;
  els.speed.placeholder = minText;
  els.speed.title = `1 is real time, 0.5 is 50%, and 2 is 200%. Minimum ${formatSpeed(minSpeed)} gives a step interval of 1.`;
  clampSpeedInputToMinimum(commit);
  els.speed.disabled = els.maxSpeed.checked;
}

function formatSpeed(value) {
  if (value === 0) return "max";
  const numeric = Number(value);
  const absValue = Math.abs(numeric);
  const digits = absValue < 0.01 ? 4 : absValue < 0.1 ? 3 : absValue < 1 ? 2 : 1;
  return `${numeric.toFixed(digits).replace(/\.?0+$/, "")}x`;
}

function boundaryHasReflection() {
  if (els.fieldGeometry.value === "double_sech") {
    return els.boundary.value === "partial_reflect";
  }
  return els.boundaryX.value === "partial_reflect" || els.boundaryY.value === "partial_reflect";
}

function syncReflectControl() {
  const showReflect = boundaryHasReflection();
  els.partialReflectControl.classList.toggle("hidden-control", !showReflect);
  els.partialReflectStrength.disabled = !showReflect;
}

function syncCouplingControls() {
  const showOverlapControls = els.coupling.value === "overlap";
  els.overlapRowsControl.classList.toggle("hidden-control", !showOverlapControls);
  els.couplingStrengthControl.classList.toggle("hidden-control", !showOverlapControls);
  els.overlapRows.disabled = !showOverlapControls;
  els.couplingStrength.disabled = !showOverlapControls;
}

function setOptionAvailable(select, value, available) {
  const option = Array.from(select.options).find((item) => item.value === value);
  if (!option) return;
  option.disabled = !available;
  option.hidden = !available;
}

// Cortical and retinal activity rendering

function setCanvasSize(canvas, rows, cols) {
  if (canvas.width !== cols || canvas.height !== rows) {
    canvas.width = cols;
    canvas.height = rows;
  }
  canvas.style.aspectRatio = `${cols} / ${rows}`;
}

function drawValues(canvas, values, rows, cols) {
  setCanvasSize(canvas, rows, cols);
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(cols, rows);
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

function activeRetinalResolution() {
  let resolution = Math.max(81, Math.min(801, Math.round(Number(els.retinalResolution.value) || 321)));
  if (resolution % 2 === 0) resolution = Math.min(801, resolution + 1);
  return resolution;
}

function setPixel(image, pixelIndex, value) {
  const [r, g, b] = palette(value);
  const j = pixelIndex * 4;
  image.data[j] = r;
  image.data[j + 1] = g;
  image.data[j + 2] = b;
  image.data[j + 3] = 255;
}

function updateCorticalLabels(isCoupled, leftCenter = 50, rightCenter = 50, positions = null) {
  const isDoubleSech = els.fieldGeometry.value === "double_sech";
  els.corticalFrame.classList.toggle("coupled", isCoupled);
  els.corticalFrame.classList.toggle("stacked", isCoupled);
  els.corticalFrame.classList.toggle("double-sech", isDoubleSech);
  els.corticalFrame.style.setProperty("--left-center", `${leftCenter}%`);
  els.corticalFrame.style.setProperty("--right-center", `${rightCenter}%`);
  if (positions) {
    Object.entries(positions).forEach(([key, value]) => {
      els.corticalFrame.style.setProperty(key, value);
    });
  } else {
    els.corticalFrame.style.setProperty("--left-label-y", "8px");
    els.corticalFrame.style.setProperty("--right-label-y", "8px");
    els.corticalFrame.style.setProperty("--left-top-y", "42px");
    els.corticalFrame.style.setProperty("--right-top-y", "42px");
    els.corticalFrame.style.setProperty("--left-bottom-y", "calc(100% - 30px)");
    els.corticalFrame.style.setProperty("--right-bottom-y", "calc(100% - 30px)");
    els.corticalFrame.style.setProperty("--left-ecc-y", "16px");
    els.corticalFrame.style.setProperty("--right-ecc-y", "16px");
  }
  els.hemiLeft.textContent = isCoupled ? "Left hemisphere" : "Cortical sheet";
  els.hemiRight.textContent = "Right hemisphere";
  document.querySelectorAll(".axis-top-left, .axis-top-right").forEach((label) => {
    label.textContent = isCoupled ? "90\u00b0" : "0\u00b0";
  });
  document.querySelectorAll(".axis-bottom-left, .axis-bottom-right").forEach((label) => {
    label.textContent = isCoupled ? "270\u00b0" : "0\u00b0";
  });
}

function drawCortical(canvas, values, rows, cols) {
  const isCoupled = cols >= rows * 1.5;
  if (!isCoupled) {
    updateCorticalLabels(false);
    drawValues(canvas, values, rows, cols);
    return;
  }

  const hemiCols = Math.floor(cols / 2);
  const gapRows = Math.max(12, Math.round(rows * 0.18));
  const drawRows = 2 * rows + gapRows;
  const drawCols = hemiCols;
  const leftBottom = 100 * (rows / drawRows);
  const rightTop = 100 * ((rows + gapRows) / drawRows);
  setCanvasSize(canvas, drawRows, drawCols);
  updateCorticalLabels(true, 50, 50, {
    "--left-label-y": "11px",
    "--left-top-y": "54px",
    "--left-bottom-y": `calc(${leftBottom.toFixed(2)}% + 6px)`,
    "--left-ecc-y": "11px",
    "--right-label-y": `calc(${rightTop.toFixed(2)}% - 12px)`,
    "--right-top-y": `calc(${rightTop.toFixed(2)}% + 14px)`,
    "--right-bottom-y": "calc(100% - 24px)",
    "--right-ecc-y": `calc(${rightTop.toFixed(2)}% - 12px)`
  });

  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(drawCols, drawRows);
  for (let i = 0; i < image.data.length; i += 4) {
    image.data[i] = 248;
    image.data[i + 1] = 251;
    image.data[i + 2] = 253;
    image.data[i + 3] = 255;
  }

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < hemiCols; col++) {
      const rightRow = els.fieldGeometry.value === "double_sech" ? rows - 1 - row : row;
      setPixel(image, row * drawCols + col, values[row * cols + col]);
      setPixel(
        image,
        (row + rows + gapRows) * drawCols + col,
        values[rightRow * cols + hemiCols + col]
      );
    }
  }
  ctx.putImageData(image, 0, 0);
}

function setPauseUi(isPaused) {
  paused = isPaused;
  els.pausePlay.textContent = isPaused ? "Play" : "Pause";
  els.pausePlay.classList.toggle("paused", isPaused);
}

function resetMetrics() {
  els.simTime.textContent = "0 ms";
  els.streamFps.textContent = "0";
  els.stepInterval.textContent = "1";
  els.rtx.textContent = "0";
  els.legendLow.textContent = "low";
  els.legendHigh.textContent = "high";
  rateSamples = [];
  drawStimulusGraph(0);
}

// Rolling browser-side measurements of displayed FPS and simulation speed
function updateRateMetrics(wallTimeMs, simulationTimeMs) {
  rateSamples.push({ wallTimeMs, simulationTimeMs });
  const cutoff = wallTimeMs - 1000;
  while (
    rateSamples.length > 2 &&
    rateSamples[0].wallTimeMs < cutoff &&
    rateSamples[1].wallTimeMs <= cutoff
  ) {
    rateSamples.shift();
  }

  if (rateSamples.length < 2) {
    els.streamFps.textContent = "0";
    els.rtx.textContent = "0";
    return;
  }

  const first = rateSamples[0];
  const last = rateSamples[rateSamples.length - 1];
  const elapsedWallMs = Math.max(1, last.wallTimeMs - first.wallTimeMs);
  const elapsedSimulationMs = Math.max(0, last.simulationTimeMs - first.simulationTimeMs);
  const actualFps = (rateSamples.length - 1) * 1000 / elapsedWallMs;
  const actualRealtimeX = elapsedSimulationMs / elapsedWallMs;
  els.streamFps.textContent = actualFps.toFixed(1);
  els.rtx.textContent = actualRealtimeX.toFixed(2);
}

// Analysis panes: kernel, stimulus, field geometry, and phase plane

function gaussian1dValue(x, sigma) {
  return Math.exp(-(x * x) / (sigma * sigma)) / (Math.sqrt(Math.PI) * sigma);
}

function kernelMass1d(sigma, radius) {
  let sum = 0;
  for (let x = -radius; x <= radius; x++) sum += gaussian1dValue(x, sigma);
  return sum;
}

function drawKernelGraph() {
  const canvas = els.kernelGraph;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(360, Math.round(rect.width * dpr));
  const height = Math.max(160, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  let n = Math.max(5, Number(els.n.value) || 101);
  if (els.fieldGeometry.value === "double_sech") {
    n = Math.max(5, Math.round(81 * (Number(els.fieldDensity.value) || 1)));
    if (n % 2 === 0) n += 1;
  }
  const se = Math.max(0.1, Number(els.se.value) || 2);
  const si = Math.max(0.1, Number(els.si.value) || 5);
  const cutoff = Math.max(0.5, Number(els.kernelCutoff.value) || 3);
  const radiusE = Math.max(1, Math.ceil(cutoff * se));
  const radiusI = Math.max(1, Math.ceil(cutoff * si));
  const fullRadius = Math.max(radiusI, Math.floor(n / 2));
  const retainedE = Math.pow(kernelMass1d(se, radiusE) / kernelMass1d(se, fullRadius), 2) * 100;
  const retainedI = Math.pow(kernelMass1d(si, radiusI) / kernelMass1d(si, fullRadius), 2) * 100;
  const maxRadius = Math.max(radiusE, radiusI, 4);
  const samples = [];
  let maxAbsValue = 0;
  for (let x = -maxRadius; x <= maxRadius; x++) {
    const e = gaussian1dValue(x, se);
    const i = -gaussian1dValue(x, si);
    const net = e + i;
    samples.push({ x, e, i, net });
    maxAbsValue = Math.max(maxAbsValue, Math.abs(e), Math.abs(i), Math.abs(net));
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, width, height);
  bg.addColorStop(0, "#ffffff");
  bg.addColorStop(1, "#eef7f8");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);
  const padL = 42 * dpr, padR = 20 * dpr, padT = 22 * dpr, padB = 34 * dpr;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const xToPx = (x) => padL + ((x + maxRadius) / (2 * maxRadius)) * plotW;
  const yToPx = (v) => padT + (0.5 - v / (2 * maxAbsValue)) * plotH;

  ctx.lineWidth = 1 * dpr;
  ctx.strokeStyle = "#e1edf2";
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padT + (i / 4) * plotH;
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
  }
  ctx.stroke();

  function drawRadius(radius, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2 * dpr;
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    [-radius, radius].forEach((xValue) => {
      const x = xToPx(xValue);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();
    });
    ctx.setLineDash([]);
  }

  drawRadius(radiusI, "rgba(243, 179, 61, 0.58)");
  drawRadius(radiusE, "rgba(0, 158, 170, 0.58)");

  ctx.strokeStyle = "#b9ccd7";
  ctx.beginPath();
  ctx.moveTo(padL, yToPx(0));
  ctx.lineTo(padL + plotW, yToPx(0));
  ctx.moveTo(xToPx(0), padT);
  ctx.lineTo(xToPx(0), padT + plotH);
  ctx.stroke();

  function drawLine(key, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5 * dpr;
    ctx.beginPath();
    samples.forEach((sample, idx) => {
      const x = xToPx(sample.x);
      const y = yToPx(sample[key]);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawLine("i", "#f3b33d");
  drawLine("e", "#009eaa");
  drawLine("net", "#0b3146");
  ctx.fillStyle = "#607284";
  ctx.font = `${11 * dpr}px IBM Plex Sans, sans-serif`;
  ctx.textAlign = "left";
  ctx.fillText(`-${maxRadius}`, padL, height - 10 * dpr);
  ctx.textAlign = "center";
  ctx.fillText(`0`, xToPx(0) - 3 * dpr, height - 10 * dpr);
  ctx.textAlign = "right";
  ctx.fillText(`+${maxRadius}`, padL + plotW, height - 10 * dpr);
  ctx.textAlign = "left";
  ctx.fillStyle = "#009eaa";
  ctx.fillText(`\u03c3\u2091, r=${radiusE}`, padL + 8 * dpr, padT + 14 * dpr);
  ctx.fillStyle = "#f3b33d";
  ctx.fillText(`-\u03c3\u1d62, r=${radiusI}`, padL + 110 * dpr, padT + 14 * dpr);
  ctx.fillStyle = "#0b3146";
  ctx.fillText(`E - I`, padL + 222 * dpr, padT + 14 * dpr);
  els.kernelInfo.textContent =
    `r_e=ceil(${cutoff} x ${se})=${radiusE}; r_i=ceil(${cutoff} x ${si})=${radiusI}; retained mass ${retainedE.toFixed(3)}% / ${retainedI.toFixed(3)}%; inhibitory curve is negative, dark curve is pointwise E - I`;
}

function drawRetinal(canvas, values, rows, cols) {
  if (els.retinalRendering.value !== "interpolated") {
    drawValues(canvas, values, rows, cols);
    return;
  }

  drawValues(retinalInterpolationCanvas, values, rows, cols);
  const resolution = activeRetinalResolution();
  setCanvasSize(canvas, resolution, resolution);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, resolution, resolution);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(retinalInterpolationCanvas, 0, 0, resolution, resolution);
}

function stimulusThreshold(duty) {
  return Math.sin(Math.PI * (0.5 - Math.max(0, Math.min(100, duty)) / 100));
}

function strobeValue(t, stimulus = streamStimulus) {
  const period = Math.max(1e-6, stimulus.period);
  const threshold = stimulusThreshold(stimulus.duty);
  return Math.sin((2 * Math.PI * t) / period) - threshold > 0 ? stimulus.A : 0;
}

function controlStimulus() {
  return {
    A: Number(els.amp.value) || 0,
    period: Math.max(1e-6, Number(els.period.value) || 1),
    duty: Number(els.duty.value) || 50
  };
}

function drawStimulusGraph(t = 0) {
  const canvas = els.stimulusGraph;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.round(rect.width * dpr));
  const height = Math.max(72, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fbfd";
  ctx.fillRect(0, 0, width, height);

  const padL = 34 * dpr, padR = 12 * dpr, padT = 10 * dpr, padB = 18 * dpr;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const windowMs = 500;
  const halfWindowMs = windowMs / 2;
  const start = t - halfWindowMs;
  const samples = Math.max(80, Math.round(plotW));
  const maxA = Math.max(0.001, streamStimulus.A);
  const xFor = (i) => padL + (i / (samples - 1)) * plotW;
  const yFor = (value) => padT + (1 - value / maxA) * plotH;

  ctx.strokeStyle = "#dbe7ef";
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const x = padL + (i / 4) * plotW;
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
  }
  ctx.moveTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  ctx.strokeStyle = "#009eaa";
  ctx.lineWidth = 2.2 * dpr;
  ctx.beginPath();
  for (let i = 0; i < samples; i++) {
    const sampleT = start + (i / (samples - 1)) * windowMs;
    const x = xFor(i);
    const y = yFor(strobeValue(sampleT));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.strokeStyle = "#f3b33d";
  ctx.lineWidth = 1.4 * dpr;
  const nowX = padL + plotW / 2;
  ctx.beginPath();
  ctx.moveTo(nowX, padT - 2 * dpr);
  ctx.lineTo(nowX, padT + plotH + 2 * dpr);
  ctx.stroke();

  ctx.fillStyle = "#607284";
  ctx.font = `${10 * dpr}px IBM Plex Sans, sans-serif`;
  ctx.textAlign = "left";
  ctx.fillText("-0.25 s", padL, height - 5 * dpr);
  ctx.textAlign = "center";
  ctx.fillText("now", nowX, height - 5 * dpr);
  ctx.textAlign = "right";
  ctx.fillText("+0.25 s", padL + plotW, height - 5 * dpr);
  ctx.textAlign = "left";
  els.stimulusInfo.textContent = "moving 0.5 s window";
}

function fieldDimensions() {
  if (lastDisplayFrame) {
    const coupledFrame = lastDisplayFrame.cols >= lastDisplayFrame.rows * 1.5;
    return {
      rows: lastDisplayFrame.rows,
      cols: coupledFrame ? Math.floor(lastDisplayFrame.cols / 2) : lastDisplayFrame.cols,
      coupled: coupledFrame
    };
  }
  let n = Math.max(5, Math.round(Number(els.n.value) || 81));
  const coupled = els.fieldGeometry.value === "double_sech" || els.coupling.value !== "off";
  if (els.fieldGeometry.value === "double_sech") {
    n = Math.max(5, Math.round(81 * (Number(els.fieldDensity.value) || 1)));
    if (n % 2 === 0) n += 1;
    const doubleSechCols = Math.max(7, Math.round(n * 1.57));
    return { rows: n, cols: doubleSechCols % 2 === 0 ? doubleSechCols + 1 : doubleSechCols, coupled };
  }
  return { rows: n, cols: n, coupled };
}

function drawFieldGraph() {
  const canvas = els.fieldGraph;
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(560, Math.round(rect.width * dpr));
  const height = Math.max(250, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, width, height);
  bg.addColorStop(0, "#ffffff");
  bg.addColorStop(1, "#eef7f8");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  const { rows, cols, coupled } = fieldDimensions();
  const overlap = Math.max(0, Math.round(Number(els.overlapRows.value) || 0));
  const hasOverlap = els.coupling.value === "overlap" && overlap > 0;
  const geometry = els.fieldGeometry.value;
  const leftW = width * 0.54;
  const rightW = width - leftW;
  const pad = 22 * dpr;
  const sheetW = leftW - 2 * pad;
  const sheetGap = coupled ? 18 * dpr : 0;
  const sheetH = coupled ? (height - 2 * pad - sheetGap) / 2 : height - 2 * pad;
  const pointStep = Math.max(1, Math.ceil(Math.max(rows, cols) / 96));
  const pointRadius = Math.max(0.55 * dpr, Math.min(1.6 * dpr, 58 / Math.max(rows, cols) * dpr));

  function sheetX(col, x, w) {
    return x + (cols <= 1 ? 0.5 : col / (cols - 1)) * w;
  }

  function sheetY(row, y, h) {
    return y + (rows <= 1 ? 0.5 : row / (rows - 1)) * h;
  }

  function inDoubleSechMask(row, col) {
    if (geometry !== "double_sech") return true;
    const yn = rows <= 1 ? 0 : -1 + 2 * row / (rows - 1);
    const xn = cols <= 1 ? 0 : -1 + 2 * col / (cols - 1);
    const halfWidth = 0.35 + 0.52 * Math.sqrt(Math.max(0, 1 - 0.42 * yn * yn));
    return Math.abs(xn) <= halfWidth;
  }

  function isDoubleSechBorder(row, col) {
    if (!inDoubleSechMask(row, col)) return false;
    const radius = Math.max(1, Math.floor(overlap / 2));
    for (let dr = -radius; dr <= radius; dr++) {
      for (let dc = -radius; dc <= radius; dc++) {
        const rr = row + dr;
        const cc = col + dc;
        if (rr < 0 || rr >= rows || cc < 0 || cc >= cols || !inDoubleSechMask(rr, cc)) {
          return true;
        }
      }
    }
    return false;
  }

  function nodeIsOverlap(row, col) {
    if (!hasOverlap) return false;
    if (geometry === "double_sech") return isDoubleSechBorder(row, col);
    return row < overlap || row >= rows - overlap;
  }

  function drawSheet(x, y, w, h, label) {
    ctx.strokeStyle = "rgba(13, 38, 56, 0.18)";
    ctx.lineWidth = 1.1 * dpr;
    ctx.strokeRect(x, y, w, h);
    if (hasOverlap && geometry !== "double_sech") {
      const band = Math.max(1, overlap / Math.max(rows, 1)) * h;
      ctx.fillStyle = "rgba(243, 179, 61, 0.14)";
      ctx.fillRect(x, y, w, band);
      ctx.fillRect(x, y + h - band, w, band);
    }
    ctx.fillStyle = "#0b3146";
    ctx.font = `${11 * dpr}px IBM Plex Sans, sans-serif`;
    ctx.textAlign = "left";
    ctx.fillText(label, x, y - 7 * dpr);

    for (let row = 0; row < rows; row += pointStep) {
      for (let col = 0; col < cols; col += pointStep) {
        if (!inDoubleSechMask(row, col)) continue;
        ctx.fillStyle = nodeIsOverlap(row, col) ? "rgba(243, 179, 61, 0.88)" : "rgba(0, 112, 124, 0.62)";
        ctx.beginPath();
        ctx.arc(sheetX(col, x, w), sheetY(row, y, h), pointRadius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (geometry === "double_sech") {
      ctx.fillStyle = "#607284";
      ctx.font = `${9 * dpr}px IBM Plex Sans, sans-serif`;
      const eccY = y + h + 13 * dpr < height - 4 * dpr ? y + h + 13 * dpr : y - 8 * dpr;
      ctx.textAlign = "left";
      ctx.fillText("fovea", x, eccY);
      ctx.textAlign = "right";
      ctx.fillText("periphery", x + w, eccY);
    }
  }

  if (coupled) {
    drawSheet(pad, pad + 16 * dpr, sheetW, sheetH, "Left hemisphere");
    drawSheet(pad, pad + 16 * dpr + sheetH + sheetGap, sheetW, sheetH, "Right hemisphere");
  } else {
    drawSheet(pad, pad + 16 * dpr, sheetW, sheetH, "Cortical sheet");
  }

  const cx = leftW + rightW / 2;
  const cy = height / 2 + 8 * dpr;
  const rMax = Math.min(rightW, height) * 0.34;
  ctx.strokeStyle = "rgba(13, 38, 56, 0.22)";
  ctx.lineWidth = 1.2 * dpr;
  ctx.beginPath();
  ctx.arc(cx, cy, rMax, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx - rMax, cy);
  ctx.lineTo(cx + rMax, cy);
  ctx.moveTo(cx, cy - rMax);
  ctx.lineTo(cx, cy + rMax);
  ctx.stroke();
  ctx.fillStyle = "#0b3146";
  ctx.font = `${11 * dpr}px IBM Plex Sans, sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText("Retinal projection", cx, pad + 9 * dpr);

  for (let row = 0; row < rows; row += pointStep) {
    for (let col = 0; col < cols; col += pointStep) {
      if (!inDoubleSechMask(row, col)) continue;
      const theta = Math.PI / 2 - (rows <= 1 ? 0 : row / (rows - 1)) * Math.PI * 2;
      const radius = rMax * (0.08 + 0.9 * (cols <= 1 ? 0 : col / (cols - 1)));
      ctx.fillStyle = nodeIsOverlap(row, col) ? "rgba(243, 179, 61, 0.88)" : "rgba(0, 112, 124, 0.46)";
      ctx.beginPath();
      ctx.arc(cx + Math.cos(theta) * radius, cy - Math.sin(theta) * radius, pointRadius, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.fillStyle = "#607284";
  ctx.font = `${9 * dpr}px IBM Plex Sans, sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText("fovea", cx, cy + 3 * dpr);
  ctx.fillText("periphery", cx + rMax * 0.72, cy + rMax * 0.72);

  els.fieldInfo.textContent =
    `${rows} x ${cols} nodes${pointStep > 1 ? `, showing every ${pointStep}th node` : ""}${hasOverlap ? `, overlap ${geometry === "double_sech" ? "border ring" : "rows"} highlighted: ${overlap}` : ""}`;
}

function drawPhasePlane() {
  const canvas = els.phaseGraph;
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const size = Math.max(420, Math.round(Math.max(rect.width, 320) * dpr));
  if (canvas.width !== size || canvas.height !== size) {
    canvas.width = size;
    canvas.height = size;
  }
  const width = size;
  const height = size;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, width, height);
  bg.addColorStop(0, "#ffffff");
  bg.addColorStop(1, "#eef7f8");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  const padL = 58 * dpr, padR = 20 * dpr, padT = 22 * dpr, padB = 54 * dpr;
  const availableW = width - padL - padR;
  const availableH = height - padT - padB;
  const plotSize = Math.max(120 * dpr, Math.min(availableW, availableH));
  const plotL = padL + (availableW - plotSize) / 2;
  const plotT = padT + (availableH - plotSize) / 2;
  const plotW = plotSize;
  const plotH = plotSize;
  const clampRate = (value) => Math.max(0, Math.min(1, value));
  const xRate = (value) => plotL + clampRate(value) * plotW;
  const yRate = (value) => plotT + (1 - clampRate(value)) * plotH;
  const xFor = (value) => xRate(value / 255);
  const yFor = (value) => yRate(value / 255);
  const dotPath = (x, y, radius) => {
    ctx.moveTo(x + radius, y);
    ctx.arc(x, y, radius, 0, Math.PI * 2);
  };
  const logit = (value) => {
    const safe = Math.max(1e-5, Math.min(1 - 1e-5, value));
    return Math.log(safe / (1 - safe));
  };
  const drawCurve = (sampler, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2 * dpr;
    ctx.beginPath();
    let drawing = false;
    for (let idx = 0; idx <= 256; idx++) {
      const input = idx / 256;
      const point = sampler(input);
      const valid = Number.isFinite(point.x) && Number.isFinite(point.y) &&
        point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1;
      if (!valid) {
        drawing = false;
        continue;
      }
      const x = xRate(point.x);
      const y = yRate(point.y);
      if (!drawing) {
        ctx.moveTo(x, y);
        drawing = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  };
  const drawMeanFieldNullclines = (stim) => {
    const p = meanFieldParams;
    drawCurve((ue) => ({
      x: ue,
      y: (p.Aee * ue - p.He + p.Ge * stim - logit(ue)) / p.Aie
    }), "#009eaa");
    drawCurve((ui) => ({
      x: (logit(ui) + p.Aii * ui + p.Hi - p.Gi * stim) / p.Aei,
      y: ui
    }), "#f3b33d");

    ctx.font = `${10 * dpr}px IBM Plex Sans, sans-serif`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#009eaa";
    ctx.fillText("dUe/dt=0", plotL + plotW - 4 * dpr, plotT + 14 * dpr);
    ctx.fillStyle = "#b37500";
    ctx.fillText("dUi/dt=0", plotL + plotW - 4 * dpr, plotT + 30 * dpr);
  };

  ctx.strokeStyle = "#dbe7ef";
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const x = plotL + (i / 4) * plotW;
    const y = plotT + (i / 4) * plotH;
    ctx.moveTo(x, plotT);
    ctx.lineTo(x, plotT + plotH);
    ctx.moveTo(plotL, y);
    ctx.lineTo(plotL + plotW, y);
  }
  ctx.stroke();

  ctx.strokeStyle = "#8aa2af";
  ctx.lineWidth = 1.2 * dpr;
  ctx.beginPath();
  ctx.moveTo(plotL, plotT + plotH);
  ctx.lineTo(plotL + plotW, plotT + plotH);
  ctx.moveTo(plotL, plotT);
  ctx.lineTo(plotL, plotT + plotH);
  ctx.stroke();

  ctx.setLineDash([4 * dpr, 4 * dpr]);
  ctx.strokeStyle = "rgba(13, 38, 56, 0.26)";
  ctx.beginPath();
  ctx.moveTo(plotL, plotT + plotH);
  ctx.lineTo(plotL + plotW, plotT);
  ctx.stroke();
  ctx.setLineDash([]);

  const currentT = lastDisplayFrame?.t || 0;
  const currentStim = strobeValue(currentT, controlStimulus());
  drawMeanFieldNullclines(currentStim);

  const phaseE = lastDisplayFrame?.phaseEValues || new Uint8Array();
  const phaseI = lastDisplayFrame?.phaseIValues || new Uint8Array();
  const n = Math.min(lastDisplayFrame?.phaseCount || phaseE.length, phaseE.length, phaseI.length);
  const includeAverage = els.phaseIncludeAverage?.checked ?? true;
  if (n === 0) {
    ctx.fillStyle = "#607284";
    ctx.font = `${12 * dpr}px IBM Plex Sans, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText("Waiting for a live frame...", plotL + plotW / 2, plotT + plotH / 2);
  } else {
    let meanE = 0;
    let meanI = 0;
    const pointRadius = Math.max(0.85, Math.min(2.1, 32 / Math.sqrt(n)) * dpr);
    ctx.fillStyle = "rgba(0, 87, 99, 0.34)";
    ctx.beginPath();
    for (let idx = 0; idx < n; idx++) {
      const e = phaseE[idx];
      const i = phaseI[idx];
      meanE += e;
      meanI += i;
      dotPath(xFor(e), yFor(i), pointRadius);
    }
    ctx.fill();
    meanE /= n;
    meanI /= n;

    if (includeAverage) {
      ctx.fillStyle = "#071018";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.4 * dpr;
      ctx.beginPath();
      ctx.arc(xFor(meanE), yFor(meanI), 4.8 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  ctx.fillStyle = "#607284";
  ctx.font = `${10 * dpr}px IBM Plex Sans, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const label = (i / 4).toFixed(i === 0 || i === 4 ? 0 : 2);
    ctx.fillText(label, plotL + (i / 4) * plotW, plotT + plotH + 7 * dpr);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = 1 - i / 4;
    const label = value.toFixed(value === 0 || value === 1 ? 0 : 2);
    ctx.fillText(label, plotL - 8 * dpr, plotT + (i / 4) * plotH);
  }

  ctx.fillStyle = "#0b3146";
  ctx.font = `${11 * dpr}px IBM Plex Sans, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("Excitatory firing rate (Ue)", plotL + plotW / 2, height - 12 * dpr);
  ctx.save();
  ctx.translate(16 * dpr, plotT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Inhibitory firing rate (Ui)", 0, 0);
  ctx.restore();
}

function drawCurrentFrame() {
  if (!lastDisplayFrame) return;
  drawCortical(
    els.cortical,
    lastDisplayFrame.values,
    lastDisplayFrame.rows,
    lastDisplayFrame.cols
  );
  drawRetinal(
    els.retinal,
    lastDisplayFrame.retinalValues,
    lastDisplayFrame.retinalRows,
    lastDisplayFrame.retinalCols
  );
  if (els.frameSelect.value === "field") drawFieldGraph();
  if (els.frameSelect.value === "phase") drawPhasePlane();
}

// Stream configuration, presets, and conditional controls

function updateFramePanel() {
  const selected = els.frameSelect.value;
  els.framePanels.forEach((panel) => {
    panel.classList.toggle("hidden-control", panel.dataset.frame !== selected);
  });
  if (selected === "stimulus") {
    drawStimulusGraph(lastDisplayFrame?.t || 0);
  } else if (selected === "kernel") {
    drawKernelGraph();
  } else if (selected === "field") {
    drawFieldGraph();
  } else if (selected === "phase") {
    drawPhasePlane();
  }
}

function decodeFrame(data) {
  const binary = atob(data);
  const values = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) values[i] = binary.charCodeAt(i);
  return values;
}

function streamParams() {
  syncConvolutionControls();
  const params = new URLSearchParams();
  const isDoubleSech = els.fieldGeometry.value === "double_sech";
  params.set("field_geometry", els.fieldGeometry.value);
  if (isDoubleSech) {
    params.set("field_density", els.fieldDensity.value);
  } else {
    params.set("N", els.n.value);
    params.set("fast_n", els.fastN.checked ? "true" : "false");
  }
  params.set("fps", els.fps.value);
  params.set("backend", els.backend.value);
  params.set("conv", els.conv.value);
  params.set("activity_scale", els.activityScale.value);
  const retinalResolution = activeRetinalResolution();
  els.retinalResolution.value = String(retinalResolution);
  params.set("retinal_resolution", String(retinalResolution));
  params.set("retinal_rendering", els.retinalRendering.value);
  if (isDoubleSech) {
    params.set("boundary", els.boundary.value);
  } else {
    params.set("boundary_x", els.boundaryX.value);
    params.set("boundary_y", els.boundaryY.value);
  }
  params.set("partial_reflect_strength", els.partialReflectStrength.value);
  params.set("coupling", els.coupling.value);
  params.set("speed", String(currentSpeedValue()));
  params.set("kernel_cutoff", els.kernelCutoff.value);
  params.set("A", els.amp.value);
  params.set("T", els.period.value);
  params.set("duty_cycle", els.duty.value);
  params.set("coupling_strength", els.couplingStrength.value);
  params.set("overlap_rows", els.overlapRows.value);
  params.set("Se", els.se.value);
  params.set("Si", els.si.value);
  params.set("dt", els.dt.value);
  params.set("seed", String(normalizedSeedValue()));
  return params;
}

function collectParameterSnapshot() {
  return {
    boundary: {
      boundary: els.fieldGeometry.value === "double_sech" ? els.boundary.value : null,
      boundary_x: els.fieldGeometry.value === "double_sech" ? null : els.boundaryX.value,
      boundary_y: els.fieldGeometry.value === "double_sech" ? null : els.boundaryY.value,
      reflect_gain: boundaryHasReflection() ? Number(els.partialReflectStrength.value) || 0 : null
    },
    coupling: {
      mode: els.coupling.value,
      overlap_rows: Number(els.overlapRows.value) || 0,
      gain: Number(els.couplingStrength.value) || 0
    },
    strobe: {
      amplitude: Number(els.amp.value) || 0,
      period_ms: Number(els.period.value) || 0,
      duty_cycle_percent: Number(els.duty.value) || 0
    },
    neural_field: {
      geometry: els.fieldGeometry.value,
      density: els.fieldGeometry.value === "double_sech" ? Number(els.fieldDensity.value) || 1 : null,
      sheet_size: els.fieldGeometry.value === "double_sech" ? null : Number(els.n.value) || 0,
      sigma_e: Number(els.se.value) || 0,
      sigma_i: Number(els.si.value) || 0,
      time_step_ms: Number(els.dt.value) || 0,
      kernel_cutoff: Number(els.kernelCutoff.value) || 3,
      seed: normalizedSeedValue(),
      randomize_seed_on_restart: els.randomizeSeed.checked
    }
  };
}

function printParameters(label = "Current parameters") {
  const snapshot = collectParameterSnapshot();
  els.paramOutput.textContent = `${label}\n${JSON.stringify(snapshot, null, 2)}`;
}

function setControlValue(id, value) {
  const el = els[id];
  if (!el || value === undefined) return;
  if (el.type === "checkbox") {
    el.checked = Boolean(value);
  } else {
    el.value = String(value);
  }
}

function presetDisplay(key) {
  const preset = presets[key];
  const index = presetKeys.indexOf(key) + 1;
  if (!preset || index <= 0) return "";
  return `${index}. ${preset.label}`;
}

function setPresetTitle(key) {
  const display = presetDisplay(key);
  if (display) {
    els.presetTitle.textContent = display;
  }
}

function setActivePreset(key) {
  if (!presets[key]) return;
  activePresetKey = key;
  document.querySelectorAll("[data-preset]").forEach((button) => {
    const isActive = button.dataset.preset === key;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
  setPresetTitle(key);
}

function renderPresetButtons() {
  const buttons = presetKeys.map((key, index) => {
    const button = document.createElement("button");
    const number = document.createElement("strong");
    button.type = "button";
    button.className = "preset-button";
    button.dataset.preset = key;
    button.setAttribute("aria-label", `Preset ${index + 1}: ${presets[key].label}`);
    button.setAttribute("aria-pressed", "false");
    number.textContent = String(index + 1);
    button.append(number);
    button.addEventListener("mouseenter", () => setPresetTitle(key));
    button.addEventListener("focus", () => setPresetTitle(key));
    button.addEventListener("mouseleave", () => setPresetTitle(activePresetKey));
    button.addEventListener("blur", () => setPresetTitle(activePresetKey));
    button.addEventListener("click", () => applyPreset(key));
    return button;
  });
  els.presetGrid.replaceChildren(...buttons);
}

function applyPreset(key) {
  const preset = presets[key];
  if (!preset) return;
  const values = preset.values;
  Object.entries(values).forEach(([id, value]) => setControlValue(id, value));
  if (values.fieldGeometry === "square") {
    els.fieldDensity.value = "1";
  }
  applyGeometryDefaults();
  syncCouplingControls();
  syncReflectControl();
  drawKernelGraph();
  drawStimulusGraph(lastDisplayFrame?.t || 0);
  drawFieldGraph();
  setActivePreset(key);
  printParameters(`Applied preset: ${preset.label}`);
  resetStream({ randomizeSeed: false });
}

function syncGeometryControls() {
  const isDoubleSech = els.fieldGeometry.value === "double_sech";
  els.nControl.classList.toggle("hidden-control", isDoubleSech);
  els.fastNControl.classList.toggle("hidden-control", isDoubleSech);
  els.fieldDensityControl.classList.toggle("hidden-control", !isDoubleSech);
  els.boundaryControl.classList.toggle("hidden-control", !isDoubleSech);
  els.boundaryXControl.classList.toggle("hidden-control", isDoubleSech);
  els.boundaryYControl.classList.toggle("hidden-control", isDoubleSech);
  els.n.disabled = isDoubleSech;
  els.fastN.disabled = isDoubleSech;
  els.fieldDensity.disabled = !isDoubleSech;
  els.boundary.disabled = !isDoubleSech;
  els.boundaryX.disabled = isDoubleSech;
  els.boundaryY.disabled = isDoubleSech;
  syncReflectControl();
}

function syncConvolutionControls() {
  const isDoubleSech = els.fieldGeometry.value === "double_sech";
  if (isDoubleSech) {
    els.backend.value = "metal";
  }
  els.backend.disabled = isDoubleSech;

  const isCpu = els.backend.value === "cpu";
  const squareUsesNonPeriodic =
    !isDoubleSech && (els.boundaryX.value !== "periodic" || els.boundaryY.value !== "periodic");

  if (isCpu) {
    els.boundaryX.value = "periodic";
    els.boundaryY.value = "periodic";
    els.boundaryX.disabled = true;
    els.boundaryY.disabled = true;
  } else if (!isDoubleSech) {
    els.boundaryX.disabled = false;
    els.boundaryY.disabled = false;
  }

  if (isDoubleSech || squareUsesNonPeriodic) {
    els.conv.value = "separable";
    els.conv.disabled = true;
    setOptionAvailable(els.conv, "separable", true);
    setOptionAvailable(els.conv, "fft", false);
  } else if (isCpu) {
    els.conv.value = "fft";
    els.conv.disabled = true;
    setOptionAvailable(els.conv, "separable", false);
    setOptionAvailable(els.conv, "fft", true);
  } else {
    setOptionAvailable(els.conv, "separable", true);
    setOptionAvailable(els.conv, "fft", true);
    els.conv.disabled = false;
  }
  syncReflectControl();
}

function applyGeometryDefaults() {
  syncGeometryControls();
  syncConvolutionControls();
  if (els.fieldGeometry.value !== "double_sech") return;
  els.backend.value = "metal";
  els.conv.value = "separable";
}

function sendVisualizationUpdate() {
  visualizationUpdateTimer = null;
  const fps = Math.max(1, Math.round(Number(els.fps.value) || 30));
  const speed = currentSpeedValue();
  const activityScale = els.activityScale.value === "simulation" ? "simulation" : "frame";
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(`visual:fps=${fps}&speed=${speed}&activity_scale=${activityScale}`);
    const scaleText = activityScale === "simulation" ? "simulation min/max" : "frame min/max";
    els.status.textContent = `Updated visualization: stream ${fps} fps, simulation speed ${formatSpeed(speed)}, activity scale ${scaleText}. Simulation state preserved.`;
  }
}

function queueVisualizationUpdate(delayMs = 120) {
  if (visualizationUpdateTimer) clearTimeout(visualizationUpdateTimer);
  visualizationUpdateTimer = setTimeout(sendVisualizationUpdate, delayMs);
}

// WebSocket lifecycle and live frame handling

function startStream() {
  stopStream();
  applyGeometryDefaults();
  syncReflectControl();
  syncSpeedControls();
  resetMetrics();
  drawKernelGraph();
  drawFieldGraph();
  updateFramePanel();
  updateLegend();
  lastDisplayFrame = null;
  setPauseUi(false);
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const url = `${protocol}//${location.host}/stream?${streamParams().toString()}`;
  socket = new WebSocket(url);
  const currentSocket = socket;
  const token = ++streamToken;
  els.status.textContent = "Connecting...";

  socket.onopen = () => {
    if (token !== streamToken || socket !== currentSocket) return;
    els.status.textContent = "Streaming. Use Pause to hold the current state or Reset to restart with new parameters.";
  };

  socket.onmessage = (event) => {
    if (token !== streamToken || socket !== currentSocket) return;
    const msg = JSON.parse(event.data);
    if (msg.type === "hello") {
      const duty = msg.dutyCycle === null ? "default" : `${msg.dutyCycle.toFixed(1)}% duty`;
      const coupling = msg.coupling === "overlap" ? `, overlap gain ${msg.couplingStrength}` : (msg.coupling === "no_connection" || msg.fieldGeometry === "double_sech") ? ", two hemispheres with no connection" : "";
      if (msg.speed === 0) {
        els.maxSpeed.checked = true;
      } else {
        els.maxSpeed.checked = false;
        els.speed.value = String(msg.speed);
      }
      syncSpeedControls();
      streamStimulus = {
        A: Number(msg.A) || 0,
        period: Number(msg.T) || 1,
        duty: msg.dutyCycle === null ? Number(els.duty.value) || 50 : Number(msg.dutyCycle)
      };
      drawStimulusGraph(0);
      const speedText = msg.speed === 0 ? "max" : formatSpeed(msg.speed);
      if (msg.activityScale) els.activityScale.value = msg.activityScale;
      if (msg.seed !== null && msg.seed !== undefined) els.seed.value = String(msg.seed);
      if (msg.retinalResolution) els.retinalResolution.value = String(msg.retinalResolution);
      if (msg.retinalRendering) els.retinalRendering.value = msg.retinalRendering;
      const scaleText = msg.activityScale === "simulation" ? "simulation min/max" : "frame min/max";
      const geometryText = msg.fieldGeometry === "double_sech" ? `, double-sech V1 density ${msg.fieldDensity}` : "";
      const boundaryText = msg.boundaryX === msg.boundaryY ? `boundary ${msg.boundaryX}` : `boundary x ${msg.boundaryX}, y ${msg.boundaryY}`;
      const reflectText = (msg.boundaryX === "partial_reflect" || msg.boundaryY === "partial_reflect") ? `, reflect gain ${msg.partialReflectStrength}` : "";
      const retinalText = msg.retinalRendering === "interpolated" ? "interpolated retinal rendering" : "mapped retinal rendering";
      els.status.textContent = `Streaming ${msg.backend}/${msg.conv}, ${boundaryText}${reflectText}${geometryText}, \u03c3\u2091=${msg.Se}, \u03c3\u1d62=${msg.Si}, seed ${msg.seed}, time step ${msg.dt} ms, stream ${msg.fps} fps, simulation speed ${speedText}, ${retinalText}, ${scaleText}, ${duty}${coupling}.`;
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

    const values = decodeFrame(msg.data);
    const rows = msg.rows || msg.N;
    const cols = msg.cols || msg.N;
    const retinalValues = decodeFrame(msg.retinalData || msg.data);
    const retinalRows = msg.retinalRows || msg.retinalN || msg.N;
    const retinalCols = msg.retinalCols || msg.retinalN || msg.N;
    const phaseEValues = msg.phaseEData ? decodeFrame(msg.phaseEData) : new Uint8Array();
    const phaseIValues = msg.phaseIData ? decodeFrame(msg.phaseIData) : new Uint8Array();
    lastDisplayFrame = {
      values,
      rows,
      cols,
      retinalValues,
      retinalRows,
      retinalCols,
      phaseEValues,
      phaseIValues,
      phaseCount: msg.phaseCount || phaseEValues.length,
      t: msg.t
    };
    drawCurrentFrame();
    updateRateMetrics(performance.now(), msg.t);
    els.simTime.textContent = formatSimTime(msg.t);
    els.stepInterval.textContent = String(msg.stepInterval ?? msg.stepsPerFrame ?? 1);
    els.legendLow.textContent = msg.min.toFixed(3);
    els.legendHigh.textContent = msg.max.toFixed(3);
    drawStimulusGraph(msg.t);
  };

  socket.onclose = () => {
    if (token !== streamToken) return;
    if (!resetting) {
      els.status.textContent = "Paused/disconnected. Press Play to start a stream.";
      setPauseUi(true);
    }
    if (socket === currentSocket) socket = null;
  };

  socket.onerror = () => {
    if (token !== streamToken || socket !== currentSocket) return;
    els.status.textContent = "Stream error. Check the Julia terminal for details.";
  };
}

function stopStream() {
  if (!socket) return null;
  const closingSocket = socket;
  socket = null;
  try { closingSocket.send("close"); } catch (_) {}
  try { closingSocket.close(); } catch (_) {}
  return closingSocket;
}

function resetStream({ randomizeSeed = true } = {}) {
  prepareSeedForRestart(randomizeSeed);
  if (resetFallbackTimer) clearTimeout(resetFallbackTimer);
  resetting = true;
  resetMetrics();
  setPauseUi(false);
  els.status.textContent = "Resetting stream...";

  const closingSocket = socket;
  if (!closingSocket || closingSocket.readyState === WebSocket.CLOSED) {
    resetting = false;
    startStream();
    return;
  }

  socket = null;
  streamToken += 1;
  let restarted = false;
  const restart = () => {
    if (restarted) return;
    restarted = true;
    if (resetFallbackTimer) clearTimeout(resetFallbackTimer);
    resetFallbackTimer = null;
    resetting = false;
    startStream();
  };

  closingSocket.onmessage = () => {};
  closingSocket.onclose = restart;
  closingSocket.onerror = restart;
  try { closingSocket.send("close"); } catch (_) {}
  try { closingSocket.close(); } catch (_) { restart(); }
  resetFallbackTimer = setTimeout(restart, 1200);
}

function togglePausePlay() {
  if (!socket || socket.readyState === WebSocket.CLOSED || socket.readyState === WebSocket.CLOSING) {
    prepareSeedForRestart();
    startStream();
    return;
  }
  if (paused) {
    socket.send("play");
    rateSamples = [];
    els.streamFps.textContent = "0";
    els.rtx.textContent = "0";
    setPauseUi(false);
    els.status.textContent = "Streaming resumed.";
  } else {
    socket.send("pause");
    rateSamples = [];
    els.streamFps.textContent = "0";
    els.rtx.textContent = "0";
    setPauseUi(true);
    els.status.textContent = "Paused. The simulation state is held on the Julia side.";
  }
}

function isTypingShortcutTarget(element) {
  const tag = element?.tagName;
  if (element?.isContentEditable || tag === "TEXTAREA") return true;
  if (tag !== "INPUT") return false;
  const inputType = (element.getAttribute("type") || "text").toLowerCase();
  return ["text", "search", "email", "password", "tel", "url"].includes(inputType);
}

// Event wiring and initial render

els.pausePlay.addEventListener("click", togglePausePlay);
els.reset.addEventListener("click", () => resetStream());
els.fieldGeometry.addEventListener("change", () => {
  applyGeometryDefaults();
  resetStream();
});
[els.backend, els.boundary, els.boundaryX, els.boundaryY].forEach((el) => {
  el.addEventListener("change", syncConvolutionControls);
});
els.fps.addEventListener("input", () => {
  syncSpeedControls();
  queueVisualizationUpdate(160);
});
els.dt.addEventListener("input", syncSpeedControls);
els.dt.addEventListener("change", syncSpeedControls);
els.speed.addEventListener("change", () => {
  syncSpeedControls(true);
  queueVisualizationUpdate(0);
});
els.speed.addEventListener("input", () => {
  syncSpeedControls(false);
  queueVisualizationUpdate(180);
});
els.maxSpeed.addEventListener("change", () => {
  syncSpeedControls();
  queueVisualizationUpdate(0);
});
els.activityScale.addEventListener("change", () => queueVisualizationUpdate(0));
els.frameSelect.addEventListener("change", () => {
  updateFramePanel();
});
[els.phaseIncludeAverage].forEach((el) => {
  el.addEventListener("change", drawPhasePlane);
});
[els.amp, els.period, els.duty].forEach((el) => {
  el.addEventListener("input", drawPhasePlane);
  el.addEventListener("change", drawPhasePlane);
});
els.printParams.addEventListener("click", () => printParameters());
els.colorMap.addEventListener("change", () => {
  updateLegend();
  drawCurrentFrame();
});
els.plotContours.addEventListener("input", () => {
  updateLegend();
  drawCurrentFrame();
});
els.plotContours.addEventListener("change", () => {
  els.plotContours.value = String(activeContourCount());
  updateLegend();
  drawCurrentFrame();
});
els.retinalResolution.addEventListener("change", () => {
  els.retinalResolution.value = String(activeRetinalResolution());
  if (els.retinalRendering.value === "interpolated") {
    drawCurrentFrame();
  } else {
    resetStream();
  }
});
els.retinalRendering.addEventListener("change", resetStream);
document.addEventListener("keydown", (event) => {
  const active = document.activeElement;
  const tag = active?.tagName;
  if (event.code === "Space" && !event.repeat && !isTypingShortcutTarget(active)) {
    event.preventDefault();
    handledSpaceShortcut = true;
    active?.blur?.();
    togglePausePlay();
  }
  if (event.code === "Enter" && !event.repeat && tag !== "TEXTAREA" && !active?.isContentEditable) {
    event.preventDefault();
    active?.blur?.();
    resetStream();
  }
});
document.addEventListener("keyup", (event) => {
  if (event.code !== "Space" || !handledSpaceShortcut) return;
  event.preventDefault();
  handledSpaceShortcut = false;
});
[
  els.n, els.fieldDensity, els.kernelCutoff, els.se, els.si
].forEach((el) => el.addEventListener("input", drawKernelGraph));
[
  els.n, els.fieldDensity, els.overlapRows, els.coupling, els.fieldGeometry
].forEach((el) => {
  el.addEventListener("input", drawFieldGraph);
  el.addEventListener("change", drawFieldGraph);
});
els.coupling.addEventListener("change", () => {
  syncCouplingControls();
  resetStream();
});
[
  els.overlapRows, els.couplingStrength
].forEach((el) => el.addEventListener("change", resetStream));
window.addEventListener("resize", () => {
  drawKernelGraph();
  drawStimulusGraph(lastDisplayFrame?.t || 0);
  drawFieldGraph();
  drawPhasePlane();
});
syncSpeedControls();
syncReflectControl();
syncCouplingControls();
renderPresetButtons();
setActivePreset(activePresetKey);
drawKernelGraph();
drawFieldGraph();
updateFramePanel();
updateLegend();
startStream();
