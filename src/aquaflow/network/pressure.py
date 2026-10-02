"""Pressure zone classification with elevation-head correction.

Telemetry reports gauge pressure (kPa) measured at the node.  Nodes in a
pressure zone sit at different elevations, so readings are referred to the
zone datum before being compared against the operating envelope.  The
elevation head for a node ``h`` metres above the datum is ``rho * g * h``
kilopascals (water at 1000 kg/m^3, standard gravity 9.80665 m/s^2):

    head_kpa     = elevation_m * 9.80665
    corrected    = pressure_kpa + head_kpa

The correction is additive and applied inside the classifier (the single
decision point) so it cannot be lost between telemetry ingestion and the
classification decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# kilopascals of pressure head per metre of water elevation
KPA_PER_METRE_HEAD = 9.80665


@dataclass(frozen=True)
class Node:
    id: str
    elevation_m: float
    pressure_kpa: float
    enabled: bool = True


def elevation_head_kpa(elevation_m: float) -> float:
    """Convert elevation above the zone datum to pressure head in kPa."""
    return elevation_m * KPA_PER_METRE_HEAD


def validate(node: Node, low: float, high: float) -> None:
    """Validate the operating envelope and the telemetry record."""
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("invalid operating envelope")
    if not math.isfinite(node.elevation_m):
        raise ValueError("illegal reading: elevation_m must be finite")
    if not math.isfinite(node.pressure_kpa):
        raise ValueError("illegal reading: pressure_kpa must be finite")


def corrected_pressure_kpa(node: Node) -> float:
    """Gauge pressure referred to the zone datum, in kPa."""
    return node.pressure_kpa + elevation_head_kpa(node.elevation_m)


def classify(rows: list[Node], low: float, high: float) -> dict[str, list[str]]:
    """Group node ids into ``low`` / ``normal`` / ``high`` / ``disabled``.

    Disabled readings are excluded from threshold comparison but retained in
    their own bucket.  Illegal readings and envelopes raise ValueError.
    Boundary readings equal to ``low`` or ``high`` are ``normal`` (the
    envelope bounds are inclusive).
    """
    out: dict[str, list[str]] = {"low": [], "normal": [], "high": [], "disabled": []}
    for node in rows:
        validate(node, low, high)
        if not node.enabled:
            out["disabled"].append(node.id)
            continue
        corrected = corrected_pressure_kpa(node)
        if corrected < low:
            out["low"].append(node.id)
        elif corrected > high:
            out["high"].append(node.id)
        else:
            out["normal"].append(node.id)
    return out


def evidence(rows: list[Node], low: float, high: float) -> list[dict[str, object]]:
    """Explainable per-node record of the classification decision."""
    records: list[dict[str, object]] = []
    for node in rows:
        validate(node, low, high)
        head_kpa = elevation_head_kpa(node.elevation_m)
        corrected = node.pressure_kpa + head_kpa
        if not node.enabled:
            state = "disabled"
        elif corrected < low:
            state = "low"
        elif corrected > high:
            state = "high"
        else:
            state = "normal"
        records.append(
            {
                "id": node.id,
                "state": state,
                "elevation_m": node.elevation_m,
                "pressure_kpa": node.pressure_kpa,
                "head_correction_kpa": head_kpa,
                "corrected_kpa": corrected,
                "low_limit_kpa": low,
                "high_limit_kpa": high,
            }
        )
    return records
