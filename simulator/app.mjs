import { compare } from "./market-sim.mjs";

(function () {
  "use strict";

  const PATHS = 5000;
  const elements = {
    fundamental: document.getElementById("fundamental"),
    news: document.getElementById("news"),
    momentum: document.getElementById("momentum"),
    flow: document.getElementById("flow"),
    regime: document.getElementById("regime"),
    volatility: document.getElementById("volatility"),
    depth: document.getElementById("depth"),
    crowding: document.getElementById("crowding"),
    leverage: document.getElementById("leverage"),
    seed: document.getElementById("seed"),
    run: document.getElementById("run"),
    error: document.getElementById("error"),
    busy: document.getElementById("results"),
    meta: document.getElementById("run-meta"),
    histogram: document.getElementById("histogram")
  };

  const percent = new Intl.NumberFormat("es-MX", { style: "percent", maximumFractionDigits: 1 });
  const fixedPercent = new Intl.NumberFormat("es-MX", { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const integer = new Intl.NumberFormat("es-MX");
  const inputLabels = [
    [elements.fundamental, "fundamental-value", "%"],
    [elements.news, "news-value", "%"],
    [elements.momentum, "momentum-value", "%"],
    [elements.flow, "flow-value", "%"],
    [elements.volatility, "volatility-value", "%"],
    [elements.depth, "depth-value", "×"],
    [elements.crowding, "crowding-value", "%"],
    [elements.leverage, "leverage-value", "%"]
  ];

  function readConfig() {
    return {
      fundamentalPct: Number(elements.fundamental.value),
      newsShockPct: Number(elements.news.value),
      momentumPct: Number(elements.momentum.value),
      flowAnomalyPct: Number(elements.flow.value),
      regime: elements.regime.value,
      dailyVolatilityPct: Number(elements.volatility.value),
      depth: Number(elements.depth.value),
      crowding: Number(elements.crowding.value) / 100,
      leverage: Number(elements.leverage.value) / 100
    };
  }

  function formatMetric(value) {
    return percent.format(value);
  }

  function updateInputLabels() {
    for (const [input, outputId, suffix] of inputLabels) {
      const value = Number(input.value);
      const output = document.getElementById(outputId);
      output.textContent = suffix === "×" ? `${value.toFixed(1)}×` : `${value.toFixed(1)}${suffix}`;
    }
  }

  function histogramCounts(values, bins, min, max) {
    const counts = new Array(bins).fill(0);
    for (const value of values) {
      const index = Math.max(0, Math.min(bins - 1, Math.floor((value - min) / (max - min) * bins)));
      counts[index] += 1;
    }
    return counts;
  }

  function renderHistogram(feedbackOn, feedbackOff) {
    const width = 820;
    const height = 350;
    const left = 64;
    const right = 18;
    const top = 18;
    const bottom = 53;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const min = -0.15;
    const max = 0.15;
    const bins = 40;
    const onCounts = histogramCounts(feedbackOn.returns, bins, min, max);
    const offCounts = histogramCounts(feedbackOff.returns, bins, min, max);
    const maxCount = Math.max(...onCounts, ...offCounts, 1);
    const yMax = Math.ceil(maxCount / 250) * 250;
    const step = plotWidth / bins;
    const barWidth = step * 0.42;
    const y = (count) => top + plotHeight - (count / yMax) * plotHeight;
    const x = (value) => left + (value - min) / (max - min) * plotWidth;
    let markup = `<title>Distribución de resultados simulados tras cinco sesiones</title><desc>Compara cinco mil rutas con el bucle de ventas forzadas activo frente al mismo escenario sin ventas forzadas.</desc>`;

    for (let tick = 0; tick <= 4; tick += 1) {
      const value = yMax * tick / 4;
      const tickY = y(value);
      markup += `<line x1="${left}" y1="${tickY}" x2="${width - right}" y2="${tickY}" stroke="var(--line)" stroke-width="1"/>`;
      markup += `<text x="${left - 9}" y="${tickY + 4}" text-anchor="end" fill="var(--subtle)" font-size="12">${integer.format(Math.round(value))}</text>`;
    }

    for (let index = 0; index < bins; index += 1) {
      const binStart = min + index * (max - min) / bins;
      const center = x(binStart + (max - min) / bins / 2);
      const onHeight = plotHeight - (y(onCounts[index]) - top);
      const offHeight = plotHeight - (y(offCounts[index]) - top);
      const onX = center - barWidth - 0.5;
      const offX = center + 0.5;
      const labelLow = index === 0 ? "≤ −15%" : `${(binStart * 100).toFixed(1)}%`;
      const labelHigh = index === bins - 1 ? "≥ +15%" : `${((binStart + (max - min) / bins) * 100).toFixed(1)}%`;
      markup += `<rect x="${onX}" y="${y(onCounts[index])}" width="${barWidth}" height="${onHeight}" fill="var(--blue)" opacity=".82"><title>${labelLow} a ${labelHigh}: bucle activo, ${integer.format(onCounts[index])} rutas</title></rect>`;
      markup += `<rect x="${offX}" y="${y(offCounts[index])}" width="${barWidth}" height="${offHeight}" fill="var(--orange)" opacity=".82"><title>${labelLow} a ${labelHigh}: sin ventas forzadas, ${integer.format(offCounts[index])} rutas</title></rect>`;
    }

    const zeroX = x(0);
    markup += `<line x1="${zeroX}" y1="${top}" x2="${zeroX}" y2="${top + plotHeight}" stroke="var(--ink)" stroke-width="1.5"/>`;
    for (const tickValue of [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]) {
      const tickX = x(tickValue);
      const label = `${tickValue > 0 ? "+" : ""}${Math.round(tickValue * 100)}%`;
      markup += `<line x1="${tickX}" y1="${top + plotHeight}" x2="${tickX}" y2="${top + plotHeight + 5}" stroke="var(--subtle)" stroke-width="1"/>`;
      markup += `<text x="${tickX}" y="${height - 22}" text-anchor="middle" fill="var(--subtle)" font-size="12">${label}</text>`;
    }
    markup += `<text x="${left}" y="12" fill="var(--subtle)" font-size="11">Rutas</text>`;
    elements.histogram.innerHTML = markup;
  }

  function renderComparison(on, off, seed) {
    document.getElementById("up-on").textContent = formatMetric(on.summary.probabilityUp);
    document.getElementById("up-off").textContent = formatMetric(off.summary.probabilityUp);
    document.getElementById("median-on").textContent = fixedPercent.format(on.summary.medianReturn);
    document.getElementById("median-off").textContent = fixedPercent.format(off.summary.medianReturn);
    document.getElementById("loss-on").textContent = formatMetric(on.summary.probabilityDownFive);
    document.getElementById("loss-off").textContent = formatMetric(off.summary.probabilityDownFive);
    elements.meta.textContent = `${integer.format(PATHS)} rutas · semilla ${seed} · horizonte fijo de 5 sesiones`;
    renderHistogram(on, off);
  }

  function run() {
    const seedValue = Number(elements.seed.value);
    if (!Number.isInteger(seedValue) || seedValue < 1 || seedValue > 4294967295) {
      elements.error.textContent = "Escribe una semilla entera entre 1 y 4,294,967,295.";
      return;
    }
    elements.error.textContent = "";
    elements.busy.setAttribute("aria-busy", "true");
    const paired = compare(readConfig(), seedValue, PATHS);
    renderComparison(paired.feedbackOn, paired.feedbackOff, seedValue);
    elements.busy.setAttribute("aria-busy", "false");
  }

  for (const [input] of inputLabels) input.addEventListener("input", updateInputLabels);
  elements.run.addEventListener("click", run);
  updateInputLabels();
  run();
})();
