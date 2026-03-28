# Orcastrading

A modular trading intelligence system built in sequential phases, each feeding into the next.

---

## Architecture Overview

```
P1 → P2 → P3 → P4 → P5 → P6
```

---

## Phases

### P1 — Asset Analysis Engine `[BUILDING]`
**Technical indicators + news + macro + geopolitics via web search → structured bias**

Aggregates multi-source data (technical indicators, news, macroeconomic signals, and geopolitical context) through web search to produce a structured directional bias for any given asset.

---

### P2 — Trading Setups Decision Tree
**Scenario branches: direction, entry, SL/TP, RR, probability, EV, trade type, duration**

Takes the structured bias from P1 and branches it into concrete trade scenarios. Each branch specifies direction, entry price, stop loss, take profit, risk/reward ratio, probability estimate, expected value, trade type, and duration.

---

### P3 — Historical Strategy Backtester
**Historical EV per setup, win rate, profit factor, drawdown, strategy comparison**

Validates setups generated in P2 against historical data. Outputs expected value per setup, win rate, profit factor, drawdown metrics, and side-by-side strategy comparisons.

---

### P4 — Trader Personality Profiler
**Behavioral modeling → risk tolerance mapping → strategy templates per archetype**

Models trader behavior to map individual risk tolerance and psychological tendencies to specific trader archetypes, then surfaces strategy templates suited to each archetype.

---

### P5 — Lock-in Trade Journal
**Trade entry locked once opened — discipline enforcement, statistical data integrity**

A write-once trade journal where entries are locked upon creation. Enforces trading discipline and ensures statistical integrity of the performance dataset.

---

### P6 — Parallel Trading Profiles
**Shadow strategies running in parallel — alternative exits, regime adaptations, A/B testing**

Runs shadow strategies alongside live trades to test alternative exits, adapt to changing market regimes, and A/B test strategy variants without affecting the live position.
