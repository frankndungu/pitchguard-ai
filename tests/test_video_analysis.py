import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.video_analysis import (
    MAX_VIDEO_SECONDS,
    SampledVideoFrame,
    analyze_video,
    select_nearby_frames,
)


class FakeCapture:
    def __init__(self, fps, frame_count, *, opened=True):
        self.fps = fps
        self.frame_count = frame_count
        self.opened = opened
        self.read_count = 0
        self.released = False
        self.fail_at = None

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        if property_id == 1:
            return self.fps
        if property_id == 2:
            return self.frame_count
        raise AssertionError(f"Unexpected video property: {property_id}")

    def read(self):
        self.read_count += 1
        if self.fail_at == self.read_count:
            return False, None
        return True, self.read_count

    def release(self):
        self.released = True


class FakeHog:
    def __init__(self, detections, *, error_type=None):
        self.detections = iter(detections)
        self.error_type = error_type

    def setSVMDetector(self, _detector):
        pass

    def detectMultiScale(self, _frame, **_options):
        if self.error_type is not None:
            raise self.error_type("mock detector failure")
        return next(self.detections, []), []


class VideoAnalysisTests(unittest.TestCase):
    def make_cv2(self, capture, detections=(), *, raise_detector_error=False):
        opencv_error = type("OpenCVError", (Exception,), {})
        fake_cv2 = SimpleNamespace(
            CAP_PROP_FPS=1,
            CAP_PROP_FRAME_COUNT=2,
            VideoCapture=lambda _path: capture,
            HOGDescriptor=lambda: FakeHog(
                detections,
                error_type=opencv_error if raise_detector_error else None,
            ),
            HOGDescriptor_getDefaultPeopleDetector=lambda: [],
            imencode=lambda _extension, _frame: (True, b"mock-jpeg"),
            error=opencv_error,
        )
        return fake_cv2

    def run_analysis(self, fps, frame_count, detections):
        capture = FakeCapture(fps, frame_count)
        fake_cv2 = self.make_cv2(capture, detections)
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            result = analyze_video("mocked-video")
        return result, capture

    def test_caps_analysis_at_first_30_seconds_and_reports_full_duration(self):
        fps = 2
        frame_count = fps * 70
        result, capture = self.run_analysis(fps, frame_count, [])

        self.assertEqual(MAX_VIDEO_SECONDS, 30)
        self.assertEqual(capture.read_count, fps * MAX_VIDEO_SECONDS)
        self.assertEqual(result.video_duration_seconds, 70)
        self.assertEqual(result.analyzed_duration_seconds, 30)
        self.assertFalse(result.people_detected)
        self.assertTrue(capture.released)

    def test_short_video_reports_full_clip_as_analyzed_interval(self):
        result, _capture = self.run_analysis(2, 20, [])

        self.assertEqual(result.video_duration_seconds, 10)
        self.assertEqual(result.analyzed_duration_seconds, 10)

    def test_tracks_people_and_generates_movement_samples(self):
        detections = [
            [(x, 10, 20, 40)]
            for x in (0, 2, 4, 6, 8, 10)
        ]
        result, capture = self.run_analysis(4, 12, detections)

        self.assertTrue(result.people_detected)
        self.assertEqual(list(result.movement_by_track), [0])
        samples = result.movement_by_track[0]
        self.assertEqual(len(samples), 5)
        self.assertEqual(
            [sample.timestamp_seconds for sample in samples],
            [0.5, 1.0, 1.5, 2.0, 2.5],
        )
        for sample in samples:
            self.assertAlmostEqual(
                sample.speed_body_lengths_per_second, 0.1
            )
        self.assertTrue(capture.released)

    def test_pose_model_error_does_not_stop_video_analysis(self):
        capture = FakeCapture(4, 12)
        fake_cv2 = self.make_cv2(
            capture,
            [[(x, 10, 20, 40)] for x in (0, 2, 4, 6, 8, 10)],
        )

        class FailingPoseEstimator:
            def estimate(self, _frame):
                raise RuntimeError("mock pose inference failure")

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            result = analyze_video(
                "mocked-video",
                pose_estimator=FailingPoseEstimator(),
            )

        self.assertTrue(result.people_detected)
        self.assertEqual(len(result.movement_by_track[0]), 5)
        self.assertEqual(result.pose_error, "mock pose inference failure")
        self.assertTrue(capture.released)
        self.assertTrue(result.frames)

    def test_greedy_tracking_does_not_match_a_detection_twice(self):
        detections = [
            [(0, 0, 20, 40), (200, 0, 20, 40)],
            [(1, 0, 20, 40), (2, 0, 20, 40)],
        ]
        result, _capture = self.run_analysis(2, 2, detections)

        self.assertEqual(list(result.movement_by_track), [0])
        self.assertEqual(len(result.movement_by_track[0]), 1)
        self.assertAlmostEqual(
            result.movement_by_track[0][0].speed_body_lengths_per_second,
            0.05,
        )

    def test_distant_detection_starts_a_new_track(self):
        result, _capture = self.run_analysis(
            2,
            2,
            [[(0, 0, 20, 40)], [(200, 0, 20, 40)]],
        )

        self.assertTrue(result.people_detected)
        self.assertEqual(result.movement_by_track, {})

    def test_records_people_detection_without_movement_samples(self):
        result, _capture = self.run_analysis(
            2,
            4,
            [[(0, 0, 20, 40)], [], [], []],
        )

        self.assertTrue(result.people_detected)
        self.assertEqual(result.movement_by_track, {})

    def test_reports_no_people_when_every_sampled_frame_has_no_detections(self):
        result, capture = self.run_analysis(2, 4, [[], []])

        self.assertFalse(result.people_detected)
        self.assertEqual(result.movement_by_track, {})
        self.assertTrue(capture.released)

    def test_rejects_unopened_video_and_releases_capture(self):
        capture = FakeCapture(2, 4, opened=False)
        fake_cv2 = self.make_cv2(capture)

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            with self.assertRaisesRegex(ValueError, "could not be opened"):
                analyze_video("mocked-video")

        self.assertTrue(capture.released)

    def test_rejects_invalid_frame_rate_and_frame_count(self):
        for fps, frame_count, message in (
            (0, 10, "frame rate"),
            (float("nan"), 10, "frame rate"),
            (2, 0, "frame count"),
            (2, float("inf"), "frame count"),
        ):
            with self.subTest(fps=fps, frame_count=frame_count):
                capture = FakeCapture(fps, frame_count)
                fake_cv2 = self.make_cv2(capture)
                with patch.dict(sys.modules, {"cv2": fake_cv2}):
                    with self.assertRaisesRegex(ValueError, message):
                        analyze_video("mocked-video")
                self.assertTrue(capture.released)

    def test_unreadable_frame_fails_analysis_and_releases_video(self):
        capture = FakeCapture(2, 4)
        capture.fail_at = 2
        fake_cv2 = self.make_cv2(capture)

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            with self.assertRaisesRegex(ValueError, "ended unexpectedly"):
                analyze_video("mocked-video")

        self.assertTrue(capture.released)

    def test_wraps_opencv_processing_error_and_releases_video(self):
        capture = FakeCapture(2, 4)
        fake_cv2 = self.make_cv2(
            capture,
            [[]],
            raise_detector_error=True,
        )

        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            with self.assertRaisesRegex(RuntimeError, "OpenCV could not process"):
                analyze_video("mocked-video")

        self.assertTrue(capture.released)

    def test_reports_missing_opencv_dependency(self):
        with patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaisesRegex(RuntimeError, "OpenCV is not installed"):
                analyze_video("mocked-video")

    def test_selects_nearest_event_frames_in_timestamp_order(self):
        frames = tuple(
            SampledVideoFrame(timestamp, b"frame", ())
            for timestamp in (0.5, 1.0, 1.5, 2.0, 2.5)
        )

        selected = select_nearby_frames(frames, 1.2, max_frames=3)

        self.assertEqual(
            [frame.timestamp_seconds for frame in selected],
            [0.5, 1.0, 1.5],
        )
        self.assertEqual(select_nearby_frames(frames, 1.2, 0), [])


if __name__ == "__main__":
    unittest.main()
