/* A dependency-free view of generated, immutable experimental evidence.
 * Fetch/DOM/SVG APIs: https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API
 * and https://developer.mozilla.org/en-US/docs/Web/SVG. No statistics are refitted here.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const SVG_NS = "http://www.w3.org/2000/svg";
const COLOR = { treated: "#087d75", baseline: "#82929d", odd: "#087d75", even: "#b7773b", failed: "#a85d27" };
const state = { evidence: null, cohort: null, condition: null, shard: null, request: 0, cache: new Map() };
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const number = (value, digits = 3) => finite(value) ? value.toLocaleString("en", { maximumFractionDigits: digits, minimumFractionDigits: digits }) : "Unavailable";
const signed = (value, digits = 3) => finite(value) ? `${value > 0 ? "+" : ""}${number(value, digits)}` : "Unavailable";
const percent = (value) => finite(value) ? `${number(value * 100, 1)}%` : "Unavailable";
const mean = (value) => finite(value) ? value : value?.mean;

function node(tag, attrs = {}, text = null, svg = false) {
  const el = svg ? document.createElementNS(SVG_NS, tag) : document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, String(value)));
  if (text !== null) el.textContent = String(text);
  return el;
}

function svgNode(tag, attrs, text) { return node(tag, attrs, text, true); }

function setOptions(select, options, preferred) {
  select.replaceChildren(...options.map(({ value, label }) => node("option", { value }, label)));
  if (options.some((option) => String(option.value) === String(preferred))) select.value = preferred;
  select.disabled = options.length === 0;
}

function emptyChart(id, message) { $(id).replaceChildren(node("div", { class: "empty-chart" }, message)); }

function chart(id, width, height, title, description) {
  const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${title}. ${description}` });
  svg.append(svgNode("title", {}, title), svgNode("desc", {}, description));
  $(id).replaceChildren(svg);
  return svg;
}

function line(svg, x1, y1, x2, y2, attrs = {}) { svg.append(svgNode("line", { x1, y1, x2, y2, ...attrs })); }
function text(svg, x, y, label, attrs = {}) { svg.append(svgNode("text", { x, y, class: "chart-text", ...attrs }, label)); }

// Nice ticks preserve the measured extent; they never replace or discard endpoints.
function ticks(low, high, count = 5) {
  if (!finite(low) || !finite(high) || high <= low) return [low];
  const raw = (high - low) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = [1, 2, 5, 10].find((candidate) => candidate * power >= raw) * power;
  const values = [];
  for (let value = Math.ceil(low / unit) * unit; value <= high + unit * 1e-8; value += unit) values.push(Math.abs(value) < unit * 1e-8 ? 0 : value);
  return values;
}

function extent(values, includeZero = false) {
  const valid = values.filter(finite);
  if (includeZero) valid.push(0);
  if (!valid.length) return [-1, 1];
  let low = Math.min(...valid), high = Math.max(...valid);
  const pad = high === low ? Math.max(Math.abs(high) * 0.1, 0.1) : (high - low) * 0.08;
  return [low - pad, high + pad];
}

function labelFor(condition) {
  return condition.label || condition.id.replaceAll("__", " · ").replaceAll("_", " ");
}

function conditionLabel(condition) {
  if (!finite(condition.alpha)) return labelFor(condition);
  const value = signed(condition.alpha, condition.alpha_unit === "radians" ? 6 : 4).replace(/0+$/, "").replace(/\.$/, "");
  return condition.alpha_unit === "radians" ? `Perceived rotation ${value} rad` : `α ${value}`;
}

function metricValue(id, value, unit = "") {
  const el = $(id);
  el.replaceChildren(document.createTextNode(value));
  if (unit) el.append(node("span", { class: "unit" }, unit));
}

function baseline() { return state.cohort.conditions.find((condition) => condition.kind === "baseline"); }

function renderSummary() {
  const condition = state.condition, cohort = state.cohort, base = baseline();
  const effect = condition.pairedEffect;
  const metrics = condition.metrics || {}, baseMetrics = base?.metrics || {};
  metricValue("effect-value", signed(effect?.mean), finite(effect?.mean) ? "m/s" : "");
  $("effect-interval").textContent = Array.isArray(effect?.ci95) && effect.ci95.every(finite)
    ? `95% interval [${signed(effect.ci95[0])}, ${signed(effect.ci95[1])}] · ${effect.ci_method || "paired episodes"}`
    : "No recorded interval for this condition.";
  metricValue("retention-value", percent(metrics.forward_speed_retention));
  const sampleSize = condition.seeds?.length ?? cohort.n;
  metricValue("failure-value", finite(metrics.failures) ? `${metrics.failures} / ${sampleSize}` : "Unavailable");
  $("failure-context").textContent = finite(baseMetrics.failures) ? `Baseline: ${baseMetrics.failures} / ${sampleSize} · All recorded pairs included` : "All recorded pairs included";
  const gate = effect?.gate === true ? "This condition passes its recorded pilot gate." : effect?.gate === false ? "This condition fails its recorded pilot gate." : "No separate acceptance gate recorded for this condition.";
  const cosine = finite(condition.cosine_with_primary) ? ` Cosine with the primary vector: ${number(condition.cosine_with_primary)}.` : "";
  $("selection-note").textContent = `${labelFor(condition)} · ${sampleSize} paired resets.${cosine} ${gate} Original experiment status remains ${state.evidence.metadata.status}.`;
  const descriptions = {
    confirmation: "Locked settings evaluated on held-out resets. This is the original acceptance cohort.",
    validation: "Exploratory strength selection. These comparisons informed the locked treatment and are not held-out confirmation.",
    fresh_audit: "Later, separately preregistered mechanism audit. These resets include a previously reserved application cohort; this is not the formal replication split."
  };
  $("cohort-description").textContent = descriptions[cohort.id] || cohort.description || cohort.label;
  const rows = [
    ["World-Y velocity (m/s)", "y_velocity", number],
    ["Forward velocity (m/s)", "x_velocity", number],
    ["Mean squared action", "effort", number],
    ["Total episode reward", "total_return", (v) => number(v, 1)],
    ["Post-onset mean reward", "reward", number],
    ["Physical failures", "failures", (v) => finite(v) ? `${v} / ${sampleSize}` : "Unavailable"],
    ["Prefix-only episodes", "prefix_only", (v) => finite(v) ? `${v} / ${sampleSize}` : "Unavailable"]
  ];
  $("metrics-table").replaceChildren(...rows.map(([label, key, format]) => {
    const row = node("tr");
    row.append(node("th", { scope: "row" }, label), node("td", {}, format(baseMetrics[key])), node("td", {}, format(metrics[key])));
    return row;
  }));
}

function comparisonRows() {
  const mode = $("comparison-select").value;
  return state.cohort.conditions.filter((condition) => condition.kind !== "baseline" && (
    mode === "all" || mode === "vector" && condition.vector === state.condition.vector ||
    mode === "random" && condition.kind === "random" || mode === "nonrandom" && condition.kind !== "random"
  ));
}

function renderEffects() {
  const rows = comparisonRows().filter((condition) => finite(condition.pairedEffect?.mean));
  if (!rows.length) {
    emptyChart("effect-chart", "No recorded effects match this comparison filter.");
    $("effect-chart-note").textContent = "Select another comparison filter to inspect recorded estimates.";
    return;
  }
  const width = 650, top = 28, rowHeight = 31, bottom = 48, left = 205, right = 25;
  const height = top + rows.length * rowHeight + bottom;
  const range = extent(rows.flatMap((condition) => [condition.pairedEffect.mean, ...(condition.pairedEffect.ci95 || [])]), true);
  const scale = (value) => left + (value - range[0]) / (range[1] - range[0]) * (width - left - right);
  const svg = chart("effect-chart", width, height, "World-Y velocity effects and recorded confidence intervals", `${rows.length} conditions. Points are paired episode mean changes. Horizontal segments show available 95 percent intervals. All effects use world Y, in meters per second.`);
  ticks(...range).forEach((value) => {
    line(svg, scale(value), top - 8, scale(value), height - bottom + 5, { class: value === 0 ? "chart-zero" : "chart-grid" });
    text(svg, scale(value), height - bottom + 24, number(value, 2), { "text-anchor": "middle" });
  });
  if (!ticks(...range).includes(0)) line(svg, scale(0), top - 8, scale(0), height - bottom + 5, { class: "chart-zero" });
  rows.forEach((condition, index) => {
    const y = top + index * rowHeight + rowHeight / 2;
    const selected = condition.id === state.condition.id;
    if (selected) svg.append(svgNode("rect", { x: 0, y: y - 13, width, height: 26, fill: "#e6f3ed", rx: 4 }));
    const name = labelFor(condition);
    const label = svgNode("text", { x: left - 12, y: y + 3, class: "chart-text", "text-anchor": "end", "font-weight": selected ? 700 : 400 }, name.length > 30 ? `${name.slice(0, 29)}…` : name);
    label.append(svgNode("title", {}, name)); svg.append(label);
    const effect = condition.pairedEffect, color = selected ? COLOR.treated : condition.kind === "random" ? COLOR.baseline : "#54847b";
    if (Array.isArray(effect.ci95) && effect.ci95.every(finite)) {
      line(svg, scale(effect.ci95[0]), y, scale(effect.ci95[1]), y, { stroke: color, "stroke-width": 2 });
      effect.ci95.forEach((value) => line(svg, scale(value), y - 4, scale(value), y + 4, { stroke: color, "stroke-width": 1.5 }));
    }
    const point = svgNode("circle", { cx: scale(effect.mean), cy: y, r: selected ? 4.5 : 3, fill: color });
    point.append(svgNode("title", {}, `${name}: ${signed(effect.mean)} m/s${effect.ci95 ? `; 95% interval ${effect.ci95.map((v) => signed(v)).join(" to ")}` : ""}`)); svg.append(point);
  });
  text(svg, (left + width - right) / 2, height - 5, "Δ world-Y velocity (m/s)", { class: "chart-unit", "text-anchor": "middle" });
  $("effect-chart-note").textContent = `${rows.length} matching conditions shown. Intervals use each condition’s recorded estimator; validation comparisons are exploratory. A nonzero effect does not imply acceptance or label specificity.`;
}

function renderOddEven() {
  const magnitude = Math.abs(state.condition.alpha);
  const seen = new Set();
  const rows = state.cohort.conditions.filter((condition) => {
    if (!condition.oddEven || !finite(mean(condition.oddEven.odd)) || !finite(mean(condition.oddEven.even))) return false;
    if (condition.alpha_unit !== state.condition.alpha_unit) return false;
    if (finite(state.condition.alpha) && finite(condition.alpha) && Math.abs(Math.abs(condition.alpha) - magnitude) > 1e-8) return false;
    const key = `${condition.vector}:${Math.abs(condition.alpha)}`;
    if (seen.has(key)) return false;
    seen.add(key); return true;
  });
  if (!rows.length) {
    emptyChart("odd-chart", "No saved odd/even summary is available for this selection. Choose a validation or audit activation condition to inspect signed responses.");
    $("odd-note").textContent = "Only decompositions present in the generated evidence are shown. A single selected confirmation arm cannot determine its odd and even components; no missing outcome is imputed.";
    return;
  }
  const width = 1050, left = 230, right = 24, top = 30, bottom = 50, rowHeight = 28;
  const height = top + rows.length * rowHeight + bottom;
  const range = extent(rows.flatMap((condition) => [mean(condition.oddEven.odd), mean(condition.oddEven.even)]), true);
  const scale = (value) => left + (value - range[0]) / (range[1] - range[0]) * (width - left - right);
  const svg = chart("odd-chart", width, height, "Odd and even parts of signed responses", `${rows.length} direction and magnitude pairs. Green points show the odd component; orange points show the even component. These are saved point estimates, not uncertainty intervals.`);
  ticks(...range, 7).forEach((value) => {
    line(svg, scale(value), top - 7, scale(value), height - bottom + 3, { class: value === 0 ? "chart-zero" : "chart-grid" });
    text(svg, scale(value), height - 25, number(value, 2), { "text-anchor": "middle" });
  });
  line(svg, scale(0), top - 7, scale(0), height - bottom + 3, { class: "chart-zero" });
  rows.forEach((condition, index) => {
    const y = top + index * rowHeight + 11;
    const fullLabel = `${condition.vector} · |α| ${number(Math.abs(condition.alpha), 3)}`;
    const label = svgNode("text", { x: left - 12, y: y + 3, class: "chart-text", "text-anchor": "end" }, fullLabel.length > 36 ? `${fullLabel.slice(0, 35)}…` : fullLabel);
    label.append(svgNode("title", {}, fullLabel)); svg.append(label);
    ["odd", "even"].forEach((key, position) => {
      const value = mean(condition.oddEven[key]);
      const circle = svgNode("circle", { cx: scale(value), cy: y + (position ? 3 : -3), r: 3.5, fill: COLOR[key] });
      circle.append(svgNode("title", {}, `${fullLabel}: ${key} ${signed(value)} m/s`)); svg.append(circle);
    });
  });
  text(svg, left, 13, "● Odd: sign-dependent", { fill: COLOR.odd });
  text(svg, left + 175, 13, "● Even: shared by both signs", { fill: COLOR.even });
  text(svg, (left + width - right) / 2, height - 4, "Component of world-Y response (m/s)", { class: "chart-unit", "text-anchor": "middle" });
  $("odd-note").textContent = `${rows.length} saved signed pairs at the selected magnitude. Odd = (response at +α − response at −α) / 2; even = (response at +α + response at −α) / 2, with each response relative to baseline. Point estimates alone do not establish significance.`;
}

function pointsFor(arm, xKey, yKey) {
  const xs = arm?.[xKey] || [], ys = arm?.[yKey] || [];
  return xs.map((x, index) => ({ x, y: ys[index], index })).filter((point) => finite(point.x) && finite(point.y));
}

function drawPath(svg, points, xScale, yScale, color) {
  if (!points.length) return;
  const path = points.map((point, index) => `${index ? "L" : "M"}${xScale(point.x).toFixed(3)},${yScale(point.y).toFixed(3)}`).join(" ");
  svg.append(svgNode("path", { d: path, fill: "none", stroke: color, "stroke-width": 2.2, "stroke-linecap": "round", "stroke-linejoin": "round" }));
}

function endpoint(svg, point, xScale, yScale, failed, color, label) {
  const x = xScale(point.x), y = yScale(point.y);
  if (failed) {
    line(svg, x - 4, y - 4, x + 4, y + 4, { stroke: COLOR.failed, "stroke-width": 2 });
    line(svg, x - 4, y + 4, x + 4, y - 4, { stroke: COLOR.failed, "stroke-width": 2 });
  } else svg.append(svgNode("circle", { cx: x, cy: y, r: 3, fill: color }));
  const target = svgNode("circle", { cx: x, cy: y, r: 8, fill: "transparent" });
  target.append(svgNode("title", {}, label)); svg.append(target);
}

function plotEpisode(id, pair, kind) {
  const trajectory = kind === "trajectory", width = 600, height = 390;
  const margin = { left: 60, right: 22, top: 25, bottom: 55 };
  const plotWidth = width - margin.left - margin.right, plotHeight = height - margin.top - margin.bottom;
  const arms = [
    { name: "Baseline", arm: pair.baseline, color: COLOR.baseline },
    { name: "Selected", arm: pair.treated, color: COLOR.treated }
  ].map((item) => ({ ...item, points: pointsFor(item.arm, trajectory ? "x" : "time", trajectory ? "y" : "reward") }));
  const all = arms.flatMap((item) => item.points);
  if (!all.length) return emptyChart(id, "The selected pair has no plottable recorded samples.");
  let [xMin, xMax] = extent(all.map((point) => point.x), true);
  let [yMin, yMax] = extent(all.map((point) => point.y), true);
  if (trajectory) {
    // Both world axes use exactly the same pixels-per-meter scale.
    const scale = Math.min(plotWidth / (xMax - xMin), plotHeight / (yMax - yMin));
    const xCenter = (xMin + xMax) / 2, yCenter = (yMin + yMax) / 2;
    xMin = xCenter - plotWidth / scale / 2; xMax = xCenter + plotWidth / scale / 2;
    yMin = yCenter - plotHeight / scale / 2; yMax = yCenter + plotHeight / scale / 2;
  }
  const xScale = (value) => margin.left + (value - xMin) / (xMax - xMin) * plotWidth;
  const yScale = (value) => margin.top + plotHeight - (value - yMin) / (yMax - yMin) * plotHeight;
  const svg = chart(id, width, height, trajectory ? "Paired ground-plane trajectory" : "Paired evaluation reward", `Reset seed ${pair.seed}. ${trajectory ? "X and Y have equal distance scales. Recorded endpoints and intervention onset are marked." : "Original reward block means plotted against simulation time; early episodes stop at their recorded endpoint."}`);
  ticks(xMin, xMax, 6).forEach((value) => {
    line(svg, xScale(value), margin.top, xScale(value), height - margin.bottom, { class: "chart-grid" });
    text(svg, xScale(value), height - margin.bottom + 22, number(value, Math.abs(xMax - xMin) < 5 ? 1 : 0), { "text-anchor": "middle" });
  });
  ticks(yMin, yMax, 5).forEach((value) => {
    line(svg, margin.left, yScale(value), width - margin.right, yScale(value), { class: "chart-grid" });
    text(svg, margin.left - 10, yScale(value) + 3, number(value, Math.abs(yMax - yMin) < 5 ? 1 : 0), { "text-anchor": "end" });
  });
  line(svg, margin.left, height - margin.bottom, width - margin.right, height - margin.bottom, { class: "chart-axis" });
  line(svg, margin.left, margin.top, margin.left, height - margin.bottom, { class: "chart-axis" });
  text(svg, margin.left, 13, trajectory ? "World Y (m)" : "Reward per transition", { class: "chart-unit" });
  text(svg, margin.left + plotWidth / 2, height - 8, trajectory ? "World X (m)" : "Simulation time (s)", { class: "chart-unit", "text-anchor": "middle" });
  const warmup = state.shard?.warmup ?? state.evidence.metadata.warmup;
  const dt = state.shard?.dt ?? state.evidence.metadata.dt;
  if (!trajectory && finite(warmup) && finite(dt) && warmup * dt <= Math.max(...all.map((point) => point.x))) {
    const x = xScale(warmup * dt);
    line(svg, x, margin.top, x, height - margin.bottom, { stroke: "#3d5458", "stroke-dasharray": "4 4", "stroke-width": 1 });
    text(svg, x + 5, margin.top + 11, "onset");
  }
  arms.forEach(({ name, arm, points, color }) => {
    drawPath(svg, points, xScale, yScale, color);
    if (points.length) endpoint(svg, points.at(-1), xScale, yScale, arm.failed, color, `${name} endpoint: ${arm.length} transitions${arm.failed ? ", classified as failed" : ""}`);
    if (trajectory) {
      const onsetIndex = arm.steps?.indexOf(warmup);
      const onset = points.find((point) => point.index === onsetIndex);
      if (onset && arm.length > warmup) {
        const dot = svgNode("circle", { cx: xScale(onset.x), cy: yScale(onset.y), r: 3.3, fill: "#182e36", stroke: "white", "stroke-width": 1 });
        dot.append(svgNode("title", {}, `${name} intervention onset, step ${warmup}`)); svg.append(dot);
      }
    }
  });
}

function renderEpisode() {
  const seed = Number($("seed-select").value);
  const pair = state.shard?.episodes?.find((episode) => episode.seed === seed);
  if (!pair) {
    emptyChart("trajectory-chart", "No episode is available for this seed and condition.");
    emptyChart("reward-chart", "No reward series is available for this selection.");
    return;
  }
  const describe = (arm) => `${arm.length} recorded transitions${arm.prefix_only ? ", ended before intervention" : ""}${arm.failed ? ", failed" : ", no recorded physical failure"}`;
  $("episode-status").textContent = `Seed ${seed} · Baseline: ${describe(pair.baseline)}. Selected: ${describe(pair.treated)}.`;
  $("episode-status").classList.toggle("warning", Boolean(pair.baseline.failed || pair.treated.failed));
  plotEpisode("trajectory-chart", pair, "trajectory");
  plotEpisode("reward-chart", pair, "reward");
  $("reward-note").textContent = state.evidence.metadata.reward_curve_definition || "Saved evaluation reward is shown as block means. These are not policy-training losses or learning curves.";
  $("sampling-note").textContent = state.evidence.metadata.visual_sampling_definition || "Only the display arrays are downsampled. Full saved episodes determine numerical summaries; onset and both recorded endpoints are retained in the visual data.";
}

async function loadEpisodes() {
  const request = ++state.request;
  state.shard = null;
  $("seed-select").disabled = true;
  $("episode-status").classList.remove("warning");
  $("episode-status").textContent = "Loading the saved episode pair…";
  emptyChart("trajectory-chart", "Loading saved trajectories…");
  emptyChart("reward-chart", "Loading saved reward curves…");
  const url = state.condition.episodesUrl;
  if (!url) {
    $("episode-status").textContent = "No episode shard is published for this condition.";
    return;
  }
  try {
    if (!state.cache.has(url)) state.cache.set(url, fetchJSON(url));
    const shard = await state.cache.get(url);
    if (request !== state.request) return;
    if (!Array.isArray(shard.episodes)) throw new Error("The episode file has no episodes array.");
    state.shard = shard;
    const previous = $("seed-select").value;
    setOptions($("seed-select"), shard.episodes.map((episode) => ({ value: episode.seed, label: `${episode.seed}${episode.baseline?.failed || episode.treated?.failed ? " · failure recorded" : ""}` })), previous);
    renderEpisode();
  } catch (error) {
    state.cache.delete(url);
    if (request !== state.request) return;
    $("episode-status").textContent = `Episode data could not be loaded: ${error.message} Reload the page to retry.`;
    $("episode-status").classList.add("warning");
    emptyChart("trajectory-chart", "Saved trajectory data is unavailable.");
    emptyChart("reward-chart", "Saved reward data is unavailable.");
  }
}

function selectCondition(preferred) {
  const candidates = state.cohort.conditions.filter((condition) => condition.kind !== "baseline" && condition.vector === $("vector-select").value);
  setOptions($("condition-select"), candidates.map((condition) => ({ value: condition.id, label: conditionLabel(condition) })), preferred);
  state.condition = candidates.find((condition) => condition.id === $("condition-select").value);
  if (!state.condition) return;
  renderSummary(); renderEffects(); renderOddEven(); loadEpisodes();
}

function selectCohort() {
  state.cohort = state.evidence.cohorts.find((cohort) => cohort.id === $("cohort-select").value);
  const preferred = state.cohort.conditions.find((condition) => condition.id === state.evidence.metadata.featured?.condition)
    || state.cohort.conditions.find((condition) => condition.kind === "primary")
    || state.cohort.conditions.find((condition) => condition.kind !== "baseline");
  const vectors = [...new Set(state.cohort.conditions.filter((condition) => condition.kind !== "baseline").map((condition) => condition.vector))];
  setOptions($("vector-select"), vectors.map((value) => ({ value, label: value.replaceAll("_", " ") })), state.condition?.vector || preferred?.vector);
  selectCondition(preferred?.id);
}

function renderContext() {
  const metadata = state.evidence.metadata;
  $("historical-status").textContent = metadata.status;
  $("limitations").replaceChildren(...(metadata.limitations || []).map((limitation) => node("li", {}, limitation)));
  $("estimand-note").textContent = `${metadata.coordinate || "World-Y ground-plane velocity"}. ${metadata.estimand || "Equal-episode summaries of available post-onset motion; prefix-only episodes retain zero behavioral summaries and count as failures."}`;
  const links = [{ label: "Download the explorer evidence (JSON)", url: "data/ant005.json" }, ...(metadata.source_links || [])];
  $("source-links").replaceChildren(...links.flatMap(({ label, url }) => {
    try {
      const target = new URL(url, document.baseURI);
      if (!["https:", "http:"].includes(target.protocol)) return [];
      const item = node("li"), link = node("a", { href: target.href }, label);
      if (target.origin !== location.origin) { link.target = "_blank"; link.rel = "noopener noreferrer"; }
      link.append(node("span", { "aria-hidden": "true" }, "↗")); item.append(link); return [item];
    } catch { return []; }
  }));
}

async function fetchJSON(url) {
  const response = await fetch(url, { cache: "no-cache" });
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return response.json();
}

async function initialize() {
  try {
    const evidence = await fetchJSON("data/ant005.json");
    if (!evidence.metadata || !Array.isArray(evidence.cohorts) || !evidence.cohorts.length) throw new Error("The evidence schema is incomplete.");
    evidence.cohorts.forEach((cohort) => {
      if (!Array.isArray(cohort.conditions) || !cohort.conditions.some((condition) => condition.kind !== "baseline")) throw new Error(`No treatment conditions in ${cohort.id}.`);
    });
    state.evidence = evidence;
    setOptions($("cohort-select"), evidence.cohorts.map((cohort) => ({ value: cohort.id, label: cohort.label })), evidence.metadata.featured?.cohort || "confirmation");
    renderContext();
    $("load-state").hidden = true; $("explorer").hidden = false; $("episode").hidden = false;
    $("cohort-select").addEventListener("change", selectCohort);
    $("vector-select").addEventListener("change", () => selectCondition());
    $("condition-select").addEventListener("change", () => selectCondition($("condition-select").value));
    $("comparison-select").addEventListener("change", renderEffects);
    $("seed-select").addEventListener("change", renderEpisode);
    selectCohort();
  } catch (error) {
    $("load-state").classList.add("error-state");
    $("load-state").setAttribute("role", "alert");
    $("load-state").textContent = `The evidence could not be loaded: ${error.message} Serve this directory over HTTP or open the published GitHub Pages site. The source reports remain available below.`;
  }
}

initialize();
