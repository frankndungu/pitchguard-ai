import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSE_MODEL_INPUT_SIZE = 256
PERSON_MODEL_INPUT_SIZE = 224
POSE_LANDMARK_COUNT = 33
POSE_CONFIDENCE_THRESHOLD = 0.5
LANDMARK_CONFIDENCE_THRESHOLD = 0.5
POSE_ASSOCIATION_IOU_THRESHOLD = 0.3
POSE_ASSOCIATION_MARGIN = 0.1

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
    (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28),
    (28, 30), (30, 32), (28, 32),
)


@dataclass(frozen=True)
class PoseLandmark:
    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class PosePrediction:
    confidence: float
    landmarks: tuple[PoseLandmark | None, ...]


@dataclass(frozen=True)
class PoseDetection:
    bbox: tuple[float, float, float, float]
    roi_landmarks: tuple[tuple[float, float], ...]
    confidence: float
    pose: PosePrediction | None


@dataclass(frozen=True)
class PoseFramePerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    association_reliable: bool
    pose: PosePrediction | None
    pose_association_reliable: bool


def _generate_person_anchors():
    anchors = []
    for stride in (8, 16, 32, 32, 32):
        grid_size = math.ceil(PERSON_MODEL_INPUT_SIZE / stride)
        for y in range(grid_size):
            for x in range(grid_size):
                point = (
                    (x + 0.5) / grid_size,
                    (y + 0.5) / grid_size,
                )
                anchors.extend((point, point))
    return np.asarray(anchors, dtype=np.float32)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))


def _intersection_over_union(first, second) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def associate_pose_detections(hog_boxes, pose_detections):
    scores = [
        [
            _intersection_over_union(hog_box, pose.bbox)
            for pose in pose_detections
        ]
        for hog_box in hog_boxes
    ]
    if not scores or not pose_detections:
        return {}, bool(scores or pose_detections)

    def clear_best(values):
        ranked = sorted(values, reverse=True)
        return (
            ranked[0] >= POSE_ASSOCIATION_IOU_THRESHOLD
            and (
                len(ranked) == 1
                or ranked[0] - ranked[1] >= POSE_ASSOCIATION_MARGIN
            )
        )

    assignments = {}
    uncertain = False
    for hog_index, row in enumerate(scores):
        best_pose = max(range(len(row)), key=row.__getitem__)
        pose_scores = [scores[index][best_pose] for index in range(len(scores))]
        if clear_best(row) and clear_best(pose_scores):
            assignments[hog_index] = pose_detections[best_pose]
        elif max(row, default=0.0) >= POSE_ASSOCIATION_IOU_THRESHOLD:
            uncertain = True

    return assignments, uncertain


class OpenCVZooPoseEstimator:
    def __init__(self, person_model_path: Path, pose_model_path: Path):
        import cv2

        person_model_path = Path(person_model_path)
        pose_model_path = Path(pose_model_path)
        missing = [
            path.name for path in (person_model_path, pose_model_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Pose model file(s) are missing: "
                + ", ".join(missing)
                + ". See the README for model setup."
            )

        try:
            self._cv2 = cv2
            self._np = np
            self._person_net = cv2.dnn.readNetFromONNX(
                str(person_model_path)
            )
            self._pose_net = cv2.dnn.readNetFromONNX(str(pose_model_path))
            for network in (self._person_net, self._pose_net):
                network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._anchors = _generate_person_anchors()
        except cv2.error as error:
            raise RuntimeError(
                "OpenCV could not load the local pose models."
            ) from error

    def estimate(self, frame) -> list[PoseDetection]:
        try:
            detections = self._detect_people(frame)
            return [
                PoseDetection(
                    bbox=detection.bbox,
                    roi_landmarks=detection.roi_landmarks,
                    confidence=detection.confidence,
                    pose=self._estimate_pose(frame, detection),
                )
                for detection in detections
            ]
        except self._cv2.error as error:
            raise RuntimeError(
                "OpenCV pose inference failed for this frame."
            ) from error

    def _detect_people(self, frame) -> list[PoseDetection]:
        cv2 = self._cv2
        np = self._np
        height, width = frame.shape[:2]
        ratio = min(
            PERSON_MODEL_INPUT_SIZE / width,
            PERSON_MODEL_INPUT_SIZE / height,
        )
        resized_width = max(1, int(width * ratio))
        resized_height = max(1, int(height * ratio))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = rgb / 127.5 - 1.0
        resized = cv2.resize(rgb, (resized_width, resized_height))
        pad_left = (PERSON_MODEL_INPUT_SIZE - resized_width) // 2
        pad_top = (PERSON_MODEL_INPUT_SIZE - resized_height) // 2
        canvas = np.zeros(
            (PERSON_MODEL_INPUT_SIZE, PERSON_MODEL_INPUT_SIZE, 3),
            dtype=np.float32,
        )
        canvas[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ] = resized
        self._person_net.setInput(
            np.transpose(canvas, (2, 0, 1))[np.newaxis, ...]
        )
        outputs = self._person_net.forward(
            self._person_net.getUnconnectedOutLayersNames()
        )
        if len(outputs) < 2:
            raise RuntimeError("The person model returned unexpected outputs.")
        box_data = np.asarray(outputs[0])[0]
        score_data = np.asarray(outputs[1])[0].reshape(-1)
        if len(box_data) != len(self._anchors) or len(score_data) != len(
            self._anchors
        ):
            raise RuntimeError("The person model output shape is invalid.")

        scale = max(width, height)
        ranked = []
        for index, row in enumerate(box_data):
            score = _sigmoid(float(np.clip(score_data[index], -100, 100)))
            if score < POSE_CONFIDENCE_THRESHOLD:
                continue
            center = row[:2] / PERSON_MODEL_INPUT_SIZE
            size = row[2:4] / PERSON_MODEL_INPUT_SIZE
            anchor_x, anchor_y = self._anchors[index]
            x1 = (center[0] - size[0] / 2 + anchor_x) * scale
            y1 = (center[1] - size[1] / 2 + anchor_y) * scale
            x2 = (center[0] + size[0] / 2 + anchor_x) * scale
            y2 = (center[1] + size[1] / 2 + anchor_y) * scale
            x1 = min(width, max(0.0, x1 - pad_left / ratio))
            y1 = min(height, max(0.0, y1 - pad_top / ratio))
            x2 = min(width, max(0.0, x2 - pad_left / ratio))
            y2 = min(height, max(0.0, y2 - pad_top / ratio))
            if x2 <= x1 or y2 <= y1:
                continue
            points = row[4:12].reshape(4, 2) / PERSON_MODEL_INPUT_SIZE
            points[:, 0] = (
                points[:, 0] + self._anchors[index, 0]
            ) * scale - pad_left / ratio
            points[:, 1] = (
                points[:, 1] + self._anchors[index, 1]
            ) * scale - pad_top / ratio
            ranked.append(
                (
                    score,
                    (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    tuple((float(point[0]), float(point[1])) for point in points),
                )
            )

        ranked.sort(key=lambda detection: detection[0], reverse=True)
        if not ranked:
            return []
        boxes_for_nms = [
            [
                round(box[0]),
                round(box[1]),
                max(1, round(box[2])),
                max(1, round(box[3])),
            ]
            for _score, box, _points in ranked
        ]
        retained = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            [detection[0] for detection in ranked],
            POSE_CONFIDENCE_THRESHOLD,
            0.3,
            top_k=20,
        )
        if retained is None or len(retained) == 0:
            return []
        selected_indices = np.asarray(retained).reshape(-1)
        return [
            PoseDetection(
                bbox=ranked[int(index)][1],
                roi_landmarks=ranked[int(index)][2],
                confidence=ranked[int(index)][0],
                pose=None,
            )
            for index in selected_indices
        ]

    def _estimate_pose(self, frame, detection: PoseDetection):
        cv2 = self._cv2
        np = self._np
        hip_x, hip_y = detection.roi_landmarks[0]
        full_x, full_y = detection.roi_landmarks[1]
        distance = float(math.hypot(full_x - hip_x, full_y - hip_y))
        if not math.isfinite(distance) or distance < 2:
            return None
        full_box = np.asarray(
            [
                [hip_x - distance, hip_y - distance],
                [hip_x + distance, hip_y + distance],
            ],
            dtype=np.int32,
        )
        height, width = frame.shape[:2]
        person_box = full_box.copy()
        person_box[:, 0] = np.clip(person_box[:, 0], 0, width)
        person_box[:, 1] = np.clip(person_box[:, 1], 0, height)
        if (
            person_box[1, 0] <= person_box[0, 0]
            or person_box[1, 1] <= person_box[0, 1]
        ):
            return None
        cropped = frame[
            person_box[0, 1] : person_box[1, 1],
            person_box[0, 0] : person_box[1, 0],
        ]
        left, top = person_box[0] - full_box[0]
        right, bottom = full_box[1] - person_box[1]
        crop = cv2.copyMakeBorder(
            cropped,
            int(top),
            int(bottom),
            int(left),
            int(right),
            cv2.BORDER_CONSTANT,
            None,
            (0, 0, 0),
        )
        pad_bias = person_box[0] - [left, top]
        local_hip = np.asarray([hip_x, hip_y]) - pad_bias
        local_full = np.asarray([full_x, full_y]) - pad_bias
        radians = math.pi / 2 - math.atan2(
            -(local_full[1] - local_hip[1]),
            local_full[0] - local_hip[0],
        )
        radians -= 2 * math.pi * math.floor((radians + math.pi) / (2 * math.pi))
        angle = math.degrees(radians)
        matrix = cv2.getRotationMatrix2D(
            (float(local_hip[0]), float(local_hip[1])), angle, 1.0
        )
        rotated = cv2.warpAffine(crop, matrix, (crop.shape[1], crop.shape[0]))
        resized = cv2.resize(
            rotated, (POSE_MODEL_INPUT_SIZE, POSE_MODEL_INPUT_SIZE)
        ).astype(np.float32)
        model_input = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
        self._pose_net.setInput(model_input[np.newaxis, ...])
        outputs = self._pose_net.forward(
            self._pose_net.getUnconnectedOutLayersNames()
        )
        if len(outputs) < 5:
            raise RuntimeError("The pose model returned unexpected outputs.")
        confidence = float(np.asarray(outputs[1]).reshape(-1)[0])
        if confidence < POSE_CONFIDENCE_THRESHOLD:
            return None

        rows = np.asarray(outputs[0])[0].reshape(-1, 5)
        if len(rows) < POSE_LANDMARK_COUNT:
            raise RuntimeError("The pose model returned too few landmarks.")
        rows = rows.copy()
        rows[:, 3:5] = 1.0 / (1.0 + np.exp(-np.clip(rows[:, 3:5], -60, 60)))
        scale = np.asarray(crop.shape[1::-1], dtype=np.float32) / np.asarray(
            [POSE_MODEL_INPUT_SIZE, POSE_MODEL_INPUT_SIZE],
            dtype=np.float32,
        )
        rows[:, :2] = (
            rows[:, :2] - POSE_MODEL_INPUT_SIZE / 2
        ) * scale
        rotation = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_points = rows[:, :2] @ rotation[:, :2]
        rotation_component = np.asarray(
            [
                [matrix[0, 0], matrix[1, 0]],
                [matrix[0, 1], matrix[1, 1]],
            ]
        )
        translation = np.asarray([matrix[0, 2], matrix[1, 2]])
        inverse_rotation = np.column_stack(
            (
                rotation_component,
                -rotation_component @ translation,
            )
        )
        crop_center = np.asarray(
            [crop.shape[1] / 2, crop.shape[0] / 2, 1.0]
        )
        original_center = crop_center @ inverse_rotation.T
        original_points = rotated_points + original_center + pad_bias
        landmarks = []
        for index, row in enumerate(rows[:POSE_LANDMARK_COUNT]):
            visibility = float(row[3])
            presence = float(row[4])
            if min(visibility, presence) < LANDMARK_CONFIDENCE_THRESHOLD:
                landmarks.append(None)
                continue
            landmarks.append(
                PoseLandmark(
                    x=float(original_points[index, 0]),
                    y=float(original_points[index, 1]),
                    visibility=visibility,
                )
            )
        return PosePrediction(confidence, tuple(landmarks))


def draw_pose(frame, prediction: PosePrediction | None, color=(0, 200, 255)):
    if prediction is None:
        return frame

    import cv2

    height, width = frame.shape[:2]
    for first, second in POSE_CONNECTIONS:
        a = prediction.landmarks[first]
        b = prediction.landmarks[second]
        if a is None or b is None:
            continue
        if not (0 <= a.x < width and 0 <= a.y < height):
            continue
        if not (0 <= b.x < width and 0 <= b.y < height):
            continue
        cv2.line(
            frame,
            (round(a.x), round(a.y)),
            (round(b.x), round(b.y)),
            color,
            2,
        )

    for landmark in prediction.landmarks:
        if landmark is None:
            continue
        if not (0 <= landmark.x < width and 0 <= landmark.y < height):
            continue
        cv2.circle(
            frame,
            (round(landmark.x), round(landmark.y)),
            3,
            color,
            -1,
        )
    return frame


def draw_annotated_frame(frame_bytes: bytes, people, selected_track_id=None):
    import cv2
    import numpy as np

    frame = cv2.imdecode(
        np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        raise ValueError("The sampled frame image could not be decoded.")

    for person in people:
        x, y, width, height = person.bbox
        highlighted = (
            selected_track_id is not None
            and person.track_id == selected_track_id
            and person.association_reliable
        )
        color = (0, 140, 255) if highlighted else (0, 220, 0)
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            color,
            3 if highlighted else 2,
        )
        if person.pose is not None and person.pose_association_reliable:
            draw_pose(frame, person.pose, color)
    return frame


def select_reliable_highlight(people, selected_track_id):
    if selected_track_id is None:
        return None
    for person in people:
        if (
            person.track_id == selected_track_id
            and person.association_reliable
        ):
            return selected_track_id
    return None
