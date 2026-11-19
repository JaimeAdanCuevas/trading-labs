import test from 'node:test';
import assert from 'node:assert/strict';
import { simulate, compare } from '../simulator/market-sim.mjs';

const base = {
  fundamentalPct: 0,
  newsShockPct: 0,
  momentumPct: 0,
  flowAnomalyPct: 0,
  regime: 'mixed',
  dailyVolatilityPct: 1.2,
  depth: 1,
  crowding: 0.55,
  leverage: 0.65
};

test('same inputs and seed reproduce identical five-session paths', () => {
  const first = simulate(base, 20261118, 100, true);
  const second = simulate(base, 20261118, 100, true);
  assert.deepEqual(first.returns, second.returns);
  assert.deepEqual(first.summary, second.summary);
});

test('paired scenarios use the same random shocks when feedback is disabled', () => {
  const paired = compare({ ...base, leverage: 0 }, 20261118, 250);
  assert.deepEqual(paired.feedbackOn.returns, paired.feedbackOff.returns);
});

test('news and fundamental shifts move the simulated median in their stated direction', () => {
  const downNews = simulate({ ...base, newsShockPct: -5 }, 11, 2000, false);
  const upNews = simulate({ ...base, newsShockPct: 5 }, 11, 2000, false);
  const downFundamentals = simulate({ ...base, fundamentalPct: -4 }, 11, 2000, false);
  const upFundamentals = simulate({ ...base, fundamentalPct: 4 }, 11, 2000, false);
  assert.ok(upNews.summary.medianReturn > downNews.summary.medianReturn);
  assert.ok(upFundamentals.summary.medianReturn > downFundamentals.summary.medianReturn);
});

test('technical trend and unexplained flow controls preserve directional sensitivity', () => {
  const downTrend = simulate({ ...base, momentumPct: -5 }, 13, 2000, false);
  const upTrend = simulate({ ...base, momentumPct: 5 }, 13, 2000, false);
  const downFlow = simulate({ ...base, flowAnomalyPct: -5 }, 13, 2000, false);
  const upFlow = simulate({ ...base, flowAnomalyPct: 5 }, 13, 2000, false);
  assert.ok(upTrend.summary.medianReturn > downTrend.summary.medianReturn);
  assert.ok(upFlow.summary.medianReturn > downFlow.summary.medianReturn);
});

test('shallower depth and stressed regimes widen the simulated downside tail', () => {
  const shallow = simulate({ ...base, depth: 0.5 }, 13, 2000, false);
  const deep = simulate({ ...base, depth: 2 }, 13, 2000, false);
  const stressed = simulate({ ...base, regime: 'stressed' }, 13, 2000, false);
  const calm = simulate({ ...base, regime: 'calm' }, 13, 2000, false);
  assert.ok(shallow.summary.probabilityDownFive > deep.summary.probabilityDownFive);
  assert.ok(stressed.summary.probabilityDownFive > calm.summary.probabilityDownFive);
});

test('forced-selling feedback adds downside in the configured stress scenario', () => {
  const paired = compare(base, 20261118, 2000);
  assert.ok(paired.feedbackOn.summary.probabilityDownFive > paired.feedbackOff.summary.probabilityDownFive);
});

test('results are finite, bounded, and contain the requested number of paths', () => {
  const result = simulate({ ...base, regime: 'stressed', leverage: 1 }, 99, 1000, true);
  assert.equal(result.returns.length, 1000);
  assert.equal(result.maxDrawdowns.length, 1000);
  assert.ok(result.returns.every(Number.isFinite));
  assert.ok(result.maxDrawdowns.every(Number.isFinite));
  assert.ok(result.returns.every((value) => value > -1 && value <= Math.expm1(0.25 * 5)));
});

test('invalid path counts are rejected', () => {
  assert.throws(() => simulate(base, 1, 0, true), RangeError);
});
