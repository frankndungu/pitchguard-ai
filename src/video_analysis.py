import math
from dataclasses import dataclass
from pathlib import Path

from src.event_engine import MovementSample
from src.pose_estimation import (
    PoseFramePerson,
    associate_pose_detections,
)


SAMPLE_RATE = 2
MAX_VIDEO_SECONDS = 30


@dataclass(frozen=True)
class SampledVideoFrame:
    timestamp_seconds: float
    image_bytes: bytes | None
    people: tuple[PoseFramePerson, ...]
    preview_error: str | None = None


@dataclass(frozen=True)
class VideoAnalysisResult:
    movement_by_track: dict[int, list[MovementSample]]
    people_detected: bool
    video_duration_seconds: float
    analyzed_duration_seconds: float
    frames: tuple[SampledVideoFrame, ...] = ()
    pose_error: str | None = None


def select_nearby_frames(
    frames: tuple[SampledVideoFrame, ...] | list[SampledVideoFrame],
    timestamp_seconds: float,
    max_frames: int = 3,
) -> list[SampledVideoFrame]:
    if max_frames <= 0:
        return []
    nearest = sorted(
        frames,
        key=lambda frame: abs(frame.timestamp_seconds - timestamp_seconds),
    )[:max_frames]
    return sorted(nearest, key=lambda frame: frame.timestamp_seconds)


def analyze_video(
    video_path: Path,
    pose_estimator=None,
) -> VideoAnalysisResult:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is not installed. Install dependencies from requirements.txt."
        ) from error

    capture = None
    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("The selected file could not be opened as a video.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("The video does not report a usable frame rate.")
        if not math.isfinite(frame_count) or frame_count <= 0:
            raise ValueError("The video does not report a usable frame count.")

        frame_step = max(1, round(fps / SAMPLE_RATE))
        frame_limit = min(int(frame_count), int(fps * MAX_VIDEO_SECONDS))
        video_duration_seconds = frame_count / fps
        analyzed_duration_seconds = min(
            video_duration_seconds, MAX_VIDEO_SECONDS
        )
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        tracks = {}
        track_match_streaks = {}
        previous_detections = []
        next_track_id = 0
        people_detected = False
        sampled_frames = []
        pose_error = None

        for frame_index in range(frame_limit):
            success, frame = capture.read()
            if not success:
                raise ValueError(
                    "The video ended unexpectedly or contains unreadable frames."
                )
            if frame_index % frame_step:
                continue

            timestamp_seconds = frame_index / fps
            boxes, _weights = hog.detectMultiScale(
                frame,
                hitThreshold=0,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            people_detected = people_detected or len(boxes) > 0
            current_detections = []
            for x, y, width, height in boxes:
                current_detections.append(
                    {
                        "center": (x + width / 2, y + height / 2),
                        "height": float(height),
                        "track_id": None,
                        "association_reliable": False,
                    }
                )

            candidate_matches = []
            current_candidate_counts = [0] * len(current_detections)
            previous_candidate_counts = [0] * len(previous_detections)
            for current_index, detection in enumerate(current_detections):
                for previous_index, previous in enumerate(previous_detections):
                    dx = detection["center"][0] - previous["center"][0]
                    dy = detection["center"][1] - previous["center"][1]
                    distance = math.hypot(dx, dy)
                    max_distance = max(
                        detection["height"], previous["height"]
                    ) * 1.5
                    if distance <= max_distance:
                        current_candidate_counts[current_index] += 1
                        previous_candidate_counts[previous_index] += 1
                        candidate_matches.append(
                            (distance, current_index, previous_index)
                        )

            matched_current = set()
            matched_previous = set()
            for _distance, current_index, previous_index in sorted(
                candidate_matches
            ):
                if (
                    current_index in matched_current
                    or previous_index in matched_previous
                ):
                    continue
                detection = current_detections[current_index]
                previous = previous_detections[previous_index]
                detection["track_id"] = previous["track_id"]
                unambiguous = (
                    current_candidate_counts[current_index] == 1
                    and previous_candidate_counts[previous_index] == 1
                )
                track_id = detection["track_id"]
                track_match_streaks[track_id] = (
                    track_match_streaks.get(track_id, 0) + 1
                    if unambiguous
                    else 0
                )
                detection["association_reliable"] = (
                    track_match_streaks[track_id] >= 2
                )
                elapsed = frame_step / fps
                pixel_distance = math.hypot(
                    detection["center"][0] - previous["center"][0],
                    detection["center"][1] - previous["center"][1],
                )
                body_height = (detection["height"] + previous["height"]) / 2
                speed = pixel_distance / body_height / elapsed
                tracks[detection["track_id"]].append(
                    MovementSample(
                        timestamp_seconds=frame_index / fps,
                        speed_body_lengths_per_second=speed,
                    )
                )
                matched_current.add(current_index)
                matched_previous.add(previous_index)

            for current_index, detection in enumerate(current_detections):
                if current_index not in matched_current:
                    detection["track_id"] = next_track_id
                    track_match_streaks[next_track_id] = 0
                    tracks[next_track_id] = []
                    next_track_id += 1

            pose_detections = []
            if pose_estimator is not None:
                try:
                    pose_detections = pose_estimator.estimate(frame)
                except (OSError, RuntimeError, ValueError) as error:
                    pose_error = str(error)
                    pose_estimator = None

            hog_boxes = [
                (
                    int(box[0]),
                    int(box[1]),
                    int(box[2]),
                    int(box[3]),
                )
                for box in boxes
            ]
            pose_matches, _association_uncertain = associate_pose_detections(
                hog_boxes, pose_detections
            )
            frame_people = tuple(
                PoseFramePerson(
                    track_id=detection["track_id"],
                    bbox=hog_boxes[index],
                    association_reliable=detection[
                        "association_reliable"
                    ],
                    pose=(
                        pose_matches[index].pose
                        if index in pose_matches
                        else None
                    ),
                    pose_association_reliable=index in pose_matches,
                )
                for index, detection in enumerate(current_detections)
            )
            preview_error = None
            image_bytes = None
            try:
                encoded, buffer = cv2.imencode(".jpg", frame)
                if encoded:
                    image_bytes = bytes(buffer)
                else:
                    preview_error = "OpenCV could not encode this sampled frame."
            except cv2.error:
                preview_error = "OpenCV could not encode this sampled frame."
            sampled_frames.append(
                SampledVideoFrame(
                    timestamp_seconds=timestamp_seconds,
                    image_bytes=image_bytes,
                    people=frame_people,
                    preview_error=preview_error,
                )
            )

            previous_detections = current_detections

        return VideoAnalysisResult(
            movement_by_track={
                track_id: samples
                for track_id, samples in tracks.items()
                if samples
            },
            people_detected=people_detected,
            video_duration_seconds=video_duration_seconds,
            analyzed_duration_seconds=analyzed_duration_seconds,
            frames=tuple(sampled_frames),
            pose_error=pose_error,
        )
    except cv2.error as error:
        raise RuntimeError(
            "OpenCV could not process this video. Try a different short clip."
        ) from error
    finally:
        if capture is not None:
            capture.release()
