"""Probe: what LightGBM-style MAPE / Huber do under unit changes (same relative errors)."""

import numpy as np


def lgb_mape(y, p):
    return float(np.mean(np.abs(y - p) / np.maximum(1.0, np.abs(y))))


def lizy_mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def huber(y, p, d):
    e = np.abs(y - p)
    return float(np.mean(np.where(e <= d, 0.5 * e**2, d * (e - 0.5 * d))))


rng = np.random.default_rng(0)
y = rng.uniform(50, 150, size=1000)  # e.g. prices in JPY 100s
p = y * (1 + rng.normal(scale=0.1, size=1000))  # ~10% relative error
for unit, k in (("x1", 1.0), ("x0.01", 0.01), ("x0.001", 0.001)):
    print(
        f"{unit:7s} lgb_mape={lgb_mape(y * k, p * k):.5f} lizy_mape={lizy_mape(y * k, p * k):.3f}%"
        f" huber(d=0.9)={huber(y * k, p * k, 0.9):.6f} huber(d=1.0)={huber(y * k, p * k, 1.0):.6f}"
    )
y0 = np.array([0.0, 2.0, 10.0])
p0 = np.array([0.5, 2.2, 9.0])
print("zero target: lgb_mape =", lgb_mape(y0, p0))
