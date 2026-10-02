import math
import unittest

from aquaflow.network.pressure import (
    KPA_PER_METRE_HEAD,
    Node,
    assess,
    classify,
    datum_pressure_kpa,
    elevation_head_kpa,
)
from aquaflow.api import AquaFlowService

G = KPA_PER_METRE_HEAD
LOW = 95.0
HIGH = 200.0


class PressureZoneTests(unittest.TestCase):
    def test_elevation_head_units_and_sign(self):
        # one metre of water = 9.80665 kPa; datum pressure = gauge + head
        self.assertEqual(G, 9.80665)
        self.assertTrue(math.isclose(elevation_head_kpa(10.0), 98.0665))
        self.assertTrue(math.isclose(datum_pressure_kpa(100.0, 10.0), 198.0665))

    def test_node_crosses_low_only_with_elevation_correction(self):
        # 10 m BELOW datum: raw 100 kPa reads normal, but the zone hydraulic
        # grade (1.93 kPa) is genuinely low.
        node = Node("below-datum", elevation_m=-10.0, pressure_kpa=100.0)
        self.assertGreaterEqual(node.pressure_kpa, LOW)  # raw telemetry is fine
        reading = assess(node, LOW, HIGH)
        self.assertLess(reading.corrected_kpa, LOW)
        self.assertEqual(reading.state, "low")
        buckets = classify([node], LOW, HIGH)
        self.assertEqual(buckets["low"], ["below-datum"])

    def test_high_pressure_node(self):
        node = Node("hill", elevation_m=5.0, pressure_kpa=160.0)  # 209.03 > high
        self.assertEqual(assess(node, LOW, HIGH).state, "high")
        self.assertEqual(classify([node], LOW, HIGH)["high"], ["hill"])

    def test_normal_node(self):
        node = Node("flat", elevation_m=0.0, pressure_kpa=120.0)
        self.assertEqual(assess(node, LOW, HIGH).state, "normal")

    def test_exact_boundaries_are_normal(self):
        self.assertEqual(assess(Node("lo", 0.0, LOW), LOW, HIGH).state, "normal")
        self.assertEqual(assess(Node("hi", 0.0, HIGH), LOW, HIGH).state, "normal")
        self.assertEqual(assess(Node("x", 0.0, LOW - 1e-9), LOW, HIGH).state, "low")
        self.assertEqual(assess(Node("y", 0.0, HIGH + 1e-9), LOW, HIGH).state, "high")

    def test_boundary_at_elevation(self):
        z = 10.0
        base = LOW - z * G
        r = assess(Node("b", z, base), LOW, HIGH)
        self.assertTrue(math.isclose(r.corrected_kpa, LOW, rel_tol=1e-12, abs_tol=1e-12))
        self.assertEqual(assess(Node("b", z, base + 1e-6), LOW, HIGH).state, "normal")
        self.assertEqual(assess(Node("b", z, base - 1e-6), LOW, HIGH).state, "low")

    def test_disabled_reading_is_not_classified(self):
        disabled = Node("off", elevation_m=0.0, pressure_kpa=-50.0, enabled=False)
        buckets = classify([disabled], LOW, HIGH)
        self.assertEqual(buckets, {"low": [], "normal": [], "high": [], "disabled": ["off"]})

    def test_illegal_readings_rejected(self):
        for bad in (
            Node("nan", 0.0, float("nan")),
            Node("inf", 0.0, float("inf")),
            Node("ninf", 0.0, float("-inf")),
            Node("badelev", float("nan"), 100.0),
        ):
            with self.subTest(node=bad.id):
                with self.assertRaises(ValueError):
                    classify([bad], LOW, HIGH)

    def test_envelope_validated(self):
        with self.assertRaises(ValueError):
            classify([Node("x", 0.0, 100.0)], 100.0, 100.0)


class ZonePressureWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.svc = AquaFlowService()
        self.payload = {
            "nodes": [
                {"id": "below-datum", "elevation_m": -10.0, "pressure_kpa": 100.0},
                {"id": "flat", "elevation_m": 0.0, "pressure_kpa": 120.0},
                {"id": "hill", "elevation_m": 5.0, "pressure_kpa": 160.0},
                {"id": "off", "elevation_m": 0.0, "pressure_kpa": 1.0, "enabled": False},
            ],
            "low": LOW,
            "high": HIGH,
        }

    def test_service_distinguishes_intervene_monitor_disabled(self):
        resp = self.svc.run_workflow("zone_pressure", self.payload)
        self.assertEqual(resp["decision"], "intervene")
        self.assertEqual(resp["intervene"], ["below-datum", "hill"])
        self.assertEqual(resp["monitor"], ["flat"])
        self.assertEqual(resp["disabled"], ["off"])
        self.assertEqual(resp["value"]["low"], ["below-datum"])
        self.assertEqual(resp["value"]["high"], ["hill"])
        self.assertEqual(resp["value"]["normal"], ["flat"])
        self.assertEqual(resp["audit"]["contract"], "v1")

    def test_monitor_decision_when_no_intervention(self):
        resp = self.svc.run_workflow("zone_pressure", {
            "nodes": [{"id": "flat", "elevation_m": 0.0, "pressure_kpa": 120.0}],
            "low": LOW,
            "high": HIGH,
        })
        self.assertEqual(resp["decision"], "monitor")
        self.assertEqual(resp["intervene"], [])
        self.assertEqual(resp["monitor"], ["flat"])

    def test_explainable_evidence(self):
        resp = self.svc.run_workflow("zone_pressure", self.payload)
        evidence = {e["id"]: e for e in resp["evidence"]["readings"]}
        e = evidence["below-datum"]
        self.assertEqual(e["state"], "low")
        self.assertTrue(math.isclose(e["elevation_head_kpa"], -98.0665, abs_tol=1e-9))
        self.assertTrue(math.isclose(e["corrected_kpa"], 100.0 - 98.0665, abs_tol=1e-9))
        self.assertLess(e["corrected_kpa"], e["low"])
        self.assertEqual(resp["evidence"]["nodes"], 4)

    def test_illegal_reading_rejected_through_service(self):
        with self.assertRaises(ValueError):
            self.svc.run_workflow("zone_pressure", {
                "nodes": [{"id": "x", "elevation_m": 0.0, "pressure_kpa": float("nan")}],
                "low": LOW,
                "high": HIGH,
            })


if __name__ == "__main__":
    unittest.main()
