import json
from pathlib import Path
import unittest

from src.event_engine import (
    MovementSample,
    classify_analysis_outcome,
    detect_slowdowns,
)
from src.event_explainer import explain_event, format_timestamp


class DetectSlowdownsTests(unittest.TestCase):
    def test_detects_sustained_speed_drop_and_reports_measurements(self):
        samples = [
            MovementSample(index * 0.5, speed)
            for index, speed in enumerate(
                [1.0, 0.9, 1.1, 0.95, 0.12, 0.1, 0.08]
            )
        ]

        events = detect_slowdowns(samples)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["timestamp_seconds"], 2.0)
        self.assertAlmostEqual(events[0]["baseline_speed"], 0.95)
        self.assertAlmostEqual(events[0]["observed_speed"], 0.1)
        self.assertGreaterEqual(events[0]["confidence"], 0)
        self.assertLessEqual(events[0]["confidence"], 1)

    def test_ignores_small_or_low_baseline_changes(self):
        samples = [
            MovementSample(index * 0.5, speed)
            for index, speed in enumerate(
                [0.3, 0.35, 0.32, 0.12, 0.1, 0.11]
            )
        ]

        self.assertEqual(detect_slowdowns(samples), [])

    def test_synthetic_example_runs_through_event_detector(self):
        fixture_path = (
            Path(__file__).parent.parent / "data" / "synthetic_movement.json"
        )
        with fixture_path.open(encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        samples = [
            MovementSample(
                item["timestamp_seconds"],
                item["speed_body_lengths_per_second"],
            )
            for item in fixture["samples"]
        ]

        events = detect_slowdowns(samples)

        self.assertEqual(fixture["title"], "Synthetic sustained movement slowdown")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["timestamp_seconds"], 2.0)

    def test_explanation_includes_observable_evidence(self):
        event = detect_slowdowns(
            [
                MovementSample(index, speed)
                for index, speed in enumerate([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
            ]
        )[0]

        explained = explain_event(event)

        self.assertEqual(explained["timestamp"], "00:03")
        self.assertIn("90% decrease", explained["observation"])
        self.assertIn("not evidence of a specific injury", explained["observation"])

    def test_formats_long_timestamps(self):
        self.assertEqual(format_timestamp(367), "06:07")

    def test_rejects_invalid_measurements(self):
        with self.assertRaises(ValueError):
            detect_slowdowns([MovementSample(0, -1)])
        with self.assertRaises(ValueError):
            detect_slowdowns([], window_size=0)

    def test_classifies_video_analysis_outcomes(self):
        sufficient_track = {
            1: [MovementSample(index * 0.5, 1.0) for index in range(6)]
        }
        insufficient_track = {1: [MovementSample(0, 1.0)]}

        self.assertEqual(
            classify_analysis_outcome(False, {}, []), "no_people"
        )
        self.assertEqual(
            classify_analysis_outcome(True, insufficient_track, []),
            "insufficient_samples",
        )
        self.assertEqual(
            classify_analysis_outcome(True, sufficient_track, []), "no_events"
        )
        self.assertEqual(
            classify_analysis_outcome(
                True, sufficient_track, [{"timestamp_seconds": 1}]
            ),
            "events",
        )


if __name__ == "__main__":
    unittest.main()
