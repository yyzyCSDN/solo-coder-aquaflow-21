import unittest

from aquaflow.api import AquaFlowService
from aquaflow.network.pressure import (
    KPA_PER_METRE_HEAD,
    Node,
    classify,
    corrected_pressure_kpa,
    elevation_head_kpa,
    evidence,
)

LOW = 150.0
HIGH = 300.0


class ElevationHeadTests(unittest.TestCase):
    def test_head_units_and_sign(self):
        # 1 m of water is exactly 9.80665 kPa of head, added (not divided by 10)
        self.assertEqual(elevation_head_kpa(1.0), 9.80665)
        self.assertEqual(elevation_head_kpa(-1.0), -9.80665)

    def test_node_crosses_low_threshold_only_with_correction(self):
        # 1 m below the datum: measured 150 kPa looks normal at the node, but
        # referred to the datum it loses 9.80665 kPa and falls below 150.
        node = Node("below", elevation_m=-1.0, pressure_kpa=150.0)
        self.assertAlmostEqual(corrected_pressure_kpa(node), 150.0 - 9.80665)
        out = classify([node], LOW, HIGH)
        self.assertEqual(out["low"], ["below"])
        self.assertEqual(out["normal"], [])

    def test_correction_lifts_node_out_of_low(self):
        # 1 m above datum: 145 kPa measured would be low without the additive
        # correction, but 154.80665 kPa at the datum is normal.
        node = Node("hill", elevation_m=1.0, pressure_kpa=145.0)
        self.assertGreater(corrected_pressure_kpa(node), LOW)
        out = classify([node], LOW, HIGH)
        self.assertEqual(out["normal"], ["hill"])
        self.assertEqual(out["low"], [])

    def test_high_pressure_node(self):
        node = Node("surging", elevation_m=1.0, pressure_kpa=300.0)
        out = classify([node], LOW, HIGH)
        self.assertEqual(out["high"], ["surging"])

    def test_exact_boundaries_are_normal(self):
        at_low = Node("at-low", elevation_m=1.0, pressure_kpa=LOW - KPA_PER_METRE_HEAD)
        at_high = Node("at-high", elevation_m=1.0, pressure_kpa=HIGH - KPA_PER_METRE_HEAD)
        self.assertEqual(corrected_pressure_kpa(at_low), LOW)
        self.assertEqual(corrected_pressure_kpa(at_high), HIGH)
        out = classify([at_low, at_high], LOW, HIGH)
        self.assertEqual(sorted(out["normal"]), ["at-high", "at-low"])
        self.assertEqual(out["low"], [])
        self.assertEqual(out["high"], [])

    def test_disabled_reading_is_not_classified_against_thresholds(self):
        node = Node("off", elevation_m=-10.0, pressure_kpa=10.0, enabled=False)
        out = classify([node], LOW, HIGH)
        self.assertEqual(out["disabled"], ["off"])
        self.assertEqual(out["low"], [])
        self.assertEqual(out["high"], [])
        self.assertEqual(out["normal"], [])

    def test_illegal_readings_and_envelope_rejected(self):
        with self.assertRaises(ValueError):
            classify([Node("nan", 1.0, float("nan"))], LOW, HIGH)
        with self.assertRaises(ValueError):
            classify([Node("inf", float("inf"), 200.0)], LOW, HIGH)
        with self.assertRaises(ValueError):
            classify([Node("bad", 1.0, 200.0)], HIGH, LOW)


class WorkflowResponseTests(unittest.TestCase):
    def setUp(self):
        self.service = AquaFlowService()

    def _payload(self, nodes):
        return {"nodes": nodes, "low": LOW, "high": HIGH}

    def test_response_separates_intervention_from_monitorable(self):
        payload = self._payload(
            [
                {"id": "low-node", "elevation_m": -1.0, "pressure_kpa": 150.0},
                {"id": "high-node", "elevation_m": 1.0, "pressure_kpa": 300.0},
                {"id": "ok-node", "elevation_m": 0.0, "pressure_kpa": 200.0},
                {"id": "edge-node", "elevation_m": 1.0, "pressure_kpa": LOW - 9.80665},
                {"id": "off-node", "elevation_m": -10.0, "pressure_kpa": 10.0, "enabled": False},
            ]
        )
        resp = self.service.run_workflow("zone_pressure", payload)

        self.assertEqual(resp["decision"], "intervene")
        self.assertEqual(resp["intervention_nodes"], ["high-node", "low-node"])
        self.assertEqual(resp["monitorable_nodes"], ["edge-node", "ok-node"])
        self.assertEqual(resp["disabled_nodes"], ["off-node"])
        self.assertEqual(resp["workflow"], "zone_pressure")
        self.assertEqual(resp["audit"]["contract"], "v1")

        detail = {row["id"]: row for row in resp["evidence"]["detail"]}
        low_row = detail["low-node"]
        self.assertAlmostEqual(low_row["corrected_kpa"], 150.0 - 9.80665)
        self.assertEqual(low_row["head_correction_kpa"], -9.80665)
        self.assertEqual(low_row["state"], "low")
        self.assertEqual(detail["high-node"]["state"], "high")
        self.assertEqual(detail["ok-node"]["state"], "normal")
        self.assertEqual(detail["edge-node"]["state"], "normal")
        self.assertEqual(detail["off-node"]["state"], "disabled")
        self.assertEqual(resp["evidence"]["head_correction"], "add")
        self.assertEqual(resp["evidence"]["kpa_per_metre_head"], 9.80665)

    def test_all_normal_decision_is_monitor(self):
        payload = self._payload(
            [{"id": "ok", "elevation_m": 2.0, "pressure_kpa": 200.0}]
        )
        resp = self.service.run_workflow("zone_pressure", payload)
        self.assertEqual(resp["decision"], "monitor")
        self.assertEqual(resp["intervention_nodes"], [])
        self.assertEqual(resp["monitorable_nodes"], ["ok"])

    def test_illegal_reading_rejected_by_service(self):
        payload = self._payload(
            [{"id": "bad", "elevation_m": 1.0, "pressure_kpa": float("nan")}]
        )
        with self.assertRaises(ValueError):
            self.service.run_workflow("zone_pressure", payload)


if __name__ == "__main__":
    unittest.main()
