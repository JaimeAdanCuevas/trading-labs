"use strict";

export const HORIZON = 5;
export const REGIMES = Object.freeze({
    calm: Object.freeze({ volatility: 0.68, depth: 1.25, eventProbability: 0.015, jumpSize: 1.8 }),
    mixed: Object.freeze({ volatility: 1, depth: 1, eventProbability: 0.045, jumpSize: 2.2 }),
    stressed: Object.freeze({ volatility: 1.85, depth: 0.62, eventProbability: 0.12, jumpSize: 2.8 })
});

function mulberry32(seed) {
    let state = seed >>> 0;
    return function () {
      state += 0x6D2B79F5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

function normal(rng) {
    const a = Math.max(rng(), Number.EPSILON);
    const b = rng();
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
  }

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

function quantile(sorted, probability) {
    const position = (sorted.length - 1) * probability;
    const lower = Math.floor(position);
    const fraction = position - lower;
    const upper = Math.min(lower + 1, sorted.length - 1);
    return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
  }

function summarize(returns, maxDrawdowns) {
    const sorted = returns.slice().sort((a, b) => a - b);
    const sortedDrawdowns = maxDrawdowns.slice().sort((a, b) => a - b);
    const tailCount = Math.max(1, Math.ceil(sorted.length * 0.05));
    const worstFivePercent = sorted.slice(0, tailCount);
    return {
      probabilityUp: returns.filter((value) => value > 0).length / returns.length,
      medianReturn: quantile(sorted, 0.5),
      probabilityDownFive: returns.filter((value) => value <= -0.05).length / returns.length,
      averageWorstFivePercent: worstFivePercent.reduce((sum, value) => sum + value, 0) / worstFivePercent.length,
      medianMaxDrawdown: quantile(sortedDrawdowns, 0.5)
    };
  }

export function simulate(config, seed, paths, feedbackEnabled) {
    if (!Number.isInteger(paths) || paths < 1) throw new RangeError("paths debe ser un entero positivo");
    const regime = REGIMES[config.regime] || REGIMES.mixed;
    const dailyVolatility = clamp(Number(config.dailyVolatilityPct) || 0, 0.1, 8) / 100 * regime.volatility;
    const depth = clamp(Number(config.depth) || 1, 0.25, 3) * regime.depth;
    const crowding = clamp(Number(config.crowding) || 0, 0, 1);
    const leverage = feedbackEnabled ? clamp(Number(config.leverage) || 0, 0, 1) : 0;
    const fundamentalDrift = Math.log1p(clamp(Number(config.fundamentalPct) || 0, -25, 25) / 100) / HORIZON;
    const eventRepricing = Math.log1p(clamp(Number(config.newsShockPct) || 0, -25, 25) / 100);
    const abnormalFlow = Math.log1p(clamp(Number(config.flowAnomalyPct) || 0, -25, 25) / 100);
    const initialMomentum = Math.log1p(clamp(Number(config.momentumPct) || 0, -25, 25) / 100) / HORIZON;
    const rootSeed = Number(seed) >>> 0;
    const returns = new Array(paths);
    const maxDrawdowns = new Array(paths);

    for (let pathIndex = 0; pathIndex < paths; pathIndex += 1) {
      const pathSeed = (rootSeed ^ Math.imul(pathIndex + 1, 0x9E3779B1)) >>> 0;
      const rng = mulberry32(pathSeed);
      let marketLogPrice = 0;
      let fairValueLog = 0;
      let peakLogPrice = 0;
      let previousReturn = initialMomentum;
      let worstDrawdown = 0;

      for (let session = 0; session < HORIZON; session += 1) {
        const informationNoise = normal(rng) * dailyVolatility * 0.22;
        const marketNoise = normal(rng) * dailyVolatility;
        const eventJump = rng() < regime.eventProbability
          ? (rng() < 0.5 ? -1 : 1) * regime.jumpSize * dailyVolatility
          : 0;
        const news = session === 0 ? eventRepricing : 0;
        const newInformation = fundamentalDrift + informationNoise + eventJump + news;
        fairValueLog += newInformation;

        const fundamentalDemand = 0.32 * (fairValueLog - marketLogPrice);
        const trendDemand = crowding * 0.42 * previousReturn;
        const liquiditySupply = -0.18 * (1 - 0.45 * crowding) * previousReturn;
        const drawdown = Math.max(0, peakLogPrice - marketLogPrice);
        worstDrawdown = Math.max(worstDrawdown, drawdown);
        const forcedSelling = leverage * Math.max(0, drawdown - 0.025) * 1.7;
        const liquidityContraction = 1 - leverage * Math.min(1, drawdown / 0.08) * 0.42;
        const effectiveDepth = Math.max(0.25, depth * (feedbackEnabled ? liquidityContraction : 1));
        const unexplainedFlow = session === 0 ? abnormalFlow : 0;
        const immediateNewsReaction = session === 0 ? 0.68 * news : 0;
        const rawReturn = (
          newInformation - news + immediateNewsReaction + fundamentalDemand + trendDemand + liquiditySupply + marketNoise + unexplainedFlow - forcedSelling
        ) / effectiveDepth;
        const sessionReturn = clamp(rawReturn, -0.25, 0.25);

        marketLogPrice += sessionReturn;
        peakLogPrice = Math.max(peakLogPrice, marketLogPrice);
        previousReturn = sessionReturn;
      }

      worstDrawdown = Math.max(worstDrawdown, peakLogPrice - marketLogPrice);
      returns[pathIndex] = Math.expm1(marketLogPrice);
      maxDrawdowns[pathIndex] = Math.expm1(-worstDrawdown);
    }

    return { returns, maxDrawdowns, summary: summarize(returns, maxDrawdowns) };
  }

export function compare(config, seed, paths) {
  return {
    feedbackOn: simulate(config, seed, paths, true),
    feedbackOff: simulate(config, seed, paths, false)
  };
}
