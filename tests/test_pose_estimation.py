import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.pose_estimation import (
    OpenCVZooPoseEstimator,
    PoseDetection,
    PoseFramePerson,
    PoseLandmark,
    PosePrediction,
    associate_pose_detections,
    draw_pose,
    select_reliable_highlight,
)


class PoseEstimationTests(unittest.TestCase):
    def test_draws_skeleton_for_visible_landmarks(self):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        landmarks = [None] * 33
        landmarks[11] = PoseLandmark(20, 20, 0.9)
        landmarks[13] = PoseLandmark(30, 30, 0.9)
        prediction = PosePrediction(0.9, tuple(landmarks))

        draw_pose(frame, prediction)

        self.assertGreater(int(frame.sum()), 0)

    def test_missing_landmarks_are_not_drawn(self):
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        prediction = PosePrediction(0.9, (None,) * 33)

        draw_pose(frame, prediction)

        self.assertEqual(int(frame.sum()), 0)

    def test_pose_detection_is_associated_only_when_match_is_unique(self):
        detection = PoseDetection(
            bbox=(10, 10, 20, 40),
            roi_landmarks=(),
            confidence=0.9,
            pose=PosePrediction(0.9, (None,) * 33),
        )

        matches, uncertain = associate_pose_detections(
            [(10, 10, 20, 40)],
            [detection],
        )

        self.assertEqual(matches, {0: detection})
        self.assertFalse(uncertain)

    def test_ambiguous_pose_detection_is_not_associated(self):
        detection = PoseDetection(
            bbox=(10, 10, 20, 40),
            roi_landmarks=(),
            confidence=0.9,
            pose=PosePrediction(0.9, (None,) * 33),
        )

        matches, uncertain = associate_pose_detections(
            [(9, 10, 20, 40), (11, 10, 20, 40)],
            [detection],
        )

        self.assertEqual(matches, {})
        self.assertTrue(uncertain)

    def test_highlights_only_reliably_tracked_event_player(self):
        reliable = PoseFramePerson(
            track_id=4,
            bbox=(10, 10, 20, 40),
            association_reliable=True,
            pose=None,
            pose_association_reliable=False,
        )
        uncertain = PoseFramePerson(
            track_id=5,
            bbox=(10, 10, 20, 40),
            association_reliable=False,
            pose=None,
            pose_association_reliable=False,
        )

        self.assertEqual(select_reliable_highlight([reliable], 4), 4)
        self.assertIsNone(select_reliable_highlight([uncertain], 5))
        self.assertIsNone(select_reliable_highlight([reliable], None))

    def test_missing_model_files_have_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                FileNotFoundError, "See the README for model setup"
            ):
                OpenCVZooPoseEstimator(
                    Path(temp_dir) / "person.onnx",
                    Path(temp_dir) / "pose.onnx",
                )

    def test_mocked_local_models_return_pose_landmarks(self):
        class FakeNetwork:
            def __init__(self, outputs):
                self.outputs = outputs

            def setPreferableBackend(self, _backend):
                pass

            def setPreferableTarget(self, _target):
                pass

            def setInput(self, _input_blob):
                pass

            def getUnconnectedOutLayersNames(self):
                return ("output",)

            def forward(self, _output_names):
                return self.outputs

        anchor_count = 2254
        person_boxes = np.zeros(
            (1, anchor_count, 12),
            dtype=np.float32,
        )
        person_scores = np.full(
            (1, anchor_count, 1),
            -100,
            dtype=np.float32,
        )
        center_anchor = (13 * 28 + 13) * 2
        person_boxes[0, center_anchor, 2:4] = 90
        person_boxes[0, center_anchor, 5] = -40
        person_scores[0, center_anchor, 0] = 10

        pose_landmarks = np.zeros((1, 195), dtype=np.float32)
        pose_landmarks.reshape(39, 5)[:, 3:5] = 10
        pose_outputs = (
            pose_landmarks,
            np.asarray([[0.9]], dtype=np.float32),
            np.zeros((1, 256, 256, 1), dtype=np.float32),
            np.zeros((1, 64, 64, 39), dtype=np.float32),
            np.zeros((1, 117), dtype=np.float32),
        )
        networks = [
            FakeNetwork((person_boxes, person_scores)),
            FakeNetwork(pose_outputs),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            person_path = Path(temp_dir) / "person.onnx"
            pose_path = Path(temp_dir) / "pose.onnx"
            person_path.touch()
            pose_path.touch()
            with patch.object(
                cv2.dnn,
                "readNetFromONNX",
                side_effect=networks,
            ):
                estimator = OpenCVZooPoseEstimator(person_path, pose_path)
                results = estimator.estimate(
                    np.zeros((224, 224, 3), dtype=np.uint8)
                )

        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].pose)
        self.assertEqual(len(results[0].pose.landmarks), 33)
        self.assertAlmostEqual(results[0].pose.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
