from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from src.event_engine import MovementSample, classify_analysis_outcome
from src.pose_estimation import PoseFramePerson
from src.video_analysis import SampledVideoFrame, VideoAnalysisResult


class AppSmokeTests(unittest.TestCase):
    app_path = Path(__file__).parent.parent / "app.py"

    def test_dashboard_starts_with_outcome_classifier_imported(self):
        self.assertEqual(
            classify_analysis_outcome(False, {}, []), "no_people"
        )

        app = AppTest.from_file(str(self.app_path)).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "PitchGuard AI")
        self.assertIn("not injury diagnoses", app.warning[0].value)
        self.assertIn(
            "No video selected",
            " ".join(element.value for element in app.info),
        )
        self.assertIn(
            "Synthetic sustained movement slowdown",
            " ".join(element.value for element in app.caption),
        )

    def run_uploaded_analysis(self, result=None, error=None):
        app = AppTest.from_file(str(self.app_path)).run()
        app.file_uploader[0].upload(
            "clip.mp4", b"mock video bytes", "video/mp4"
        ).run()
        with patch("src.video_analysis.analyze_video") as analyze:
            if error is not None:
                analyze.side_effect = error
            else:
                analyze.return_value = result
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_uploaded_video_shows_distinct_analysis_outcomes(self):
        sufficient_track = {
            1: [MovementSample(index, 1.0) for index in range(6)]
        }
        insufficient_track = {1: [MovementSample(0, 1.0)]}
        event_track = {
            1: [
                MovementSample(index, speed)
                for index, speed in enumerate(
                    [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
                )
            ]
        }
        scenarios = (
            (
                VideoAnalysisResult({}, False, 70, 30),
                "No people were detected",
            ),
            (
                VideoAnalysisResult(insufficient_track, True, 12, 12),
                "not enough continuous movement samples",
            ),
            (
                VideoAnalysisResult(sufficient_track, True, 12, 12),
                "Analysis complete. No candidate movement changes",
            ),
        )

        for result, expected_message in scenarios:
            with self.subTest(message=expected_message):
                app = self.run_uploaded_analysis(result=result)

                self.assertTrue(
                    any(expected_message in item.value for item in app.info)
                )
                self.assertTrue(
                    any(
                        "Full video duration:" in item.value
                        and "Analyzed interval:" in item.value
                        for item in app.caption
                    )
                )

        event_app = self.run_uploaded_analysis(
            result=VideoAnalysisResult(event_track, True, 12, 12)
        )
        self.assertTrue(
            any(
                "Candidate movement change" in item.value
                for item in event_app.subheader
            )
        )

    def test_uploaded_video_analysis_failure_is_reported(self):
        app = self.run_uploaded_analysis(
            error=ValueError("mocked unreadable video")
        )

        self.assertTrue(
            any(
                "Video analysis failed" in item.value
                and "mocked unreadable video" in item.value
                for item in app.error
            )
        )

    def test_candidate_event_opens_nearby_annotated_frame_inspection(self):
        event_track = {
            1: [
                MovementSample(index, speed)
                for index, speed in enumerate(
                    [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
                )
            ]
        }
        encoded, buffer = cv2.imencode(
            ".jpg", np.zeros((48, 64, 3), dtype=np.uint8)
        )
        self.assertTrue(encoded)
        frame_bytes = bytes(buffer)
        person = PoseFramePerson(
            track_id=1,
            bbox=(10, 5, 20, 35),
            association_reliable=True,
            pose=None,
            pose_association_reliable=False,
        )
        frames = tuple(
            SampledVideoFrame(
                timestamp_seconds=timestamp,
                image_bytes=frame_bytes,
                people=(person,),
            )
            for timestamp in (2.5, 3.0, 3.5)
        )
        result = VideoAnalysisResult(
            event_track,
            True,
            12,
            12,
            frames=frames,
        )

        app = self.run_uploaded_analysis(result=result)

        self.assertTrue(
            any(
                "Frame inspection near" in item.value
                for item in app.subheader
            )
        )
        self.assertTrue(
            any("Frame at 3.0 seconds" in item.value for item in app.caption)
        )


if __name__ == "__main__":
    unittest.main()
