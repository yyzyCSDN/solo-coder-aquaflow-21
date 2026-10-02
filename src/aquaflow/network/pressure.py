"""Pressure-zone classification with elevation-head correction.

Telemetry reports gauge pressure at the node's physical elevation.  Before a
reading can be compared against a zone envelope (which is expressed on the
zone datum), it must be reduced to that datum by adding the elevation head:

    corrected_kpa = pressure_kpa + elevation_m * KPA_PER_METRE_HEAD

The correction is applied here, at the classification boundary, so the value
carried from telemetry through to the decision can never be dropped.  Water
adds 9.80665 kPa per metre (rho*g = 1000 kg/m^3 * 9.80665 m/s^2 / 1000).

Disabled readings are surfaced as their own set and never compared; readings
that are not finite (NaN / +-inf) are rejected as illegal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# kilopascals exerted by one metre of water column (rho*g, kPa/m).
KPA_PER_METRE_HEAD = 9.80665

_STATES = ("low", "normal", "high", "disabled")


@dataclass(frozen=True)
class Node:
    id: str
    elevation_m: float
    pressure_kpa: float
    enabled: bool = True


@dataclass(frozen=True)
class Reading:
    """One node's classification with explainable evidence."""

    node_id: str
    state: str
    pressure_kpa: float
    elevation_m: float
    elevation_head_kpa: float
    corrected_kpa: float
    low: float
    high: float
    enabled: bool


def elevation_head_kpa(elevation_m: float) -> float:
    """Pressure-equivalent (kPa) of ``elevation_m`` metres of water."""
    return elevation_m * KPA_PER_METRE_HEAD


def datum_pressure_kpa(pressure_kpa: float, elevation_m: float) -> float:
    """Reduce a gauge pressure at elevation ``elevation_m`` to the datum."""
    return pressure_kpa + elevation_head_kpa(elevation_m)


def validate(node: Node) -> None:
    """Reject illegal readings; disabled readings are otherwise left intact."""
    if not math.isfinite(node.elevation_m):
        raise ValueError(f"illegal elevation for node {node.id!r}: {node.elevation_m}")
    if not math.isfinite(node.pressure_kpa):
        raise ValueError(f"illegal pressure for node {node.id!r}: {node.pressure_kpa}")


def assess(node: Node, low: float, high: float) -> Reading:
    """Classify a single node against ``[low, high]`` on the datum."""
    if low >= high:
        raise ValueError("invalid operating envelope")
    validate(node)
    head = elevation_head_kpa(node.elevation_m)
    corrected = datum_pressure_kpa(node.pressure_kpa, node.elevation_m)
    if not node.enabled:
        state = "disabled"
    elif corrected < low:
        state = "low"
    elif corrected > high:
        state = "high"
    else:
        state = "normal"
    return Reading(
        node_id=node.id,
        state=state,
        pressure_kpa=node.pressure_kpa,
        elevation_m=node.elevation_m,
        elevation_head_kpa=head,
        corrected_kpa=corrected,
        low=low,
        high=high,
        enabled=node.enabled,
    )


def assess_many(nodes: list[Node], low: float, high: float) -> list[Reading]:
    """Classify every node, preserving input order."""
    return [assess(node, low, high) for node in nodes]


def classify(nodes: list[Node], low: float, high: float) -> dict[str, list[str]]:
    """Bucket node ids by datum-corrected state.

    Returns the intervening sets ``low``/``high`` plus ``normal`` (the
    monitorable set) and ``disabled``.
    """
    out: dict[str, list[str]] = {state: [] for state in _STATES}
    for reading in assess_many(nodes, low, high):
        out[reading.state].append(reading.node_id)
    return out
