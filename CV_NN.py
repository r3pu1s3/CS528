"""
This version of `CV_NN` uses OpenCV Haar cascades.
It detects the face with `haarcascade_frontalface_default.xml`, and detects the eyes with
`haarcascade_eye_tree_eyeglasses.xml` with option to use `haarcascade_eye.xml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


LEFT_LABEL = "LEFT"
CENTER_LABEL = "CENTER"
RIGHT_LABEL = "RIGHT"
LABELS = (LEFT_LABEL, CENTER_LABEL, RIGHT_LABEL)
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}

EYE_IMAGE_SIZE = (48, 24)  # width, height
CALIBRATION_TARGETS = (
    (CENTER_LABEL, "Look straight ahead"),
    (LEFT_LABEL, "Look left"),
    (RIGHT_LABEL, "Look right"),
)


@dataclass
class FrameDetection:
    face_box: tuple[int, int, int, int]
    left_eye_box: tuple[int, int, int, int]
    right_eye_box: tuple[int, int, int, int]
    features: np.ndarray


@dataclass
class GazePrediction:
    gaze_x: float
    gaze_y: float
    label: str
    confidence: float
    probabilities: np.ndarray
    detection: FrameDetection


@dataclass
class CalibrationResult:
    center_x: float
    center_y: float
    label_counts: dict[str, int]


class TinyMLP:
    """Small NumPy MLP for calibration-time gaze classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, seed: int = 7):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = len(LABELS)
        self.rng = np.random.default_rng(seed)

        self.feature_mean = np.zeros((1, self.input_dim), dtype=np.float32)
        self.feature_std = np.ones((1, self.input_dim), dtype=np.float32)

        w1_scale = np.sqrt(2.0 / max(self.input_dim, 1))
        w2_scale = np.sqrt(2.0 / max(self.hidden_dim, 1))
        self.w1 = (self.rng.standard_normal((self.input_dim, self.hidden_dim)) * w1_scale).astype(np.float32)
        self.b1 = np.zeros((1, self.hidden_dim), dtype=np.float32)
        self.w2 = (self.rng.standard_normal((self.hidden_dim, self.output_dim)) * w2_scale).astype(np.float32)
        self.b2 = np.zeros((1, self.output_dim), dtype=np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / self.feature_std

    def fit(
        self,
        feature_rows: np.ndarray,
        labels: list[str] | np.ndarray,
        epochs: int = 350,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
    ) -> None:
        x = np.asarray(feature_rows, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected feature matrix of shape (n, {self.input_dim}), got {x.shape}.")

        y = np.array([LABEL_TO_INDEX[label] if isinstance(label, str) else int(label) for label in labels], dtype=np.int32)
        if len(x) != len(y):
            raise ValueError("Feature row count and label count must match.")

        # Lightweight augmentation helps the model tolerate webcam noise.
        noise = self.rng.normal(0.0, 0.015, size=(2, *x.shape)).astype(np.float32)
        x_aug = np.concatenate([x, x + noise[0], x + noise[1]], axis=0)
        y_aug = np.concatenate([y, y, y], axis=0)

        self.feature_mean = x_aug.mean(axis=0, keepdims=True)
        self.feature_std = x_aug.std(axis=0, keepdims=True) + 1e-6
        x_norm = self._normalize(x_aug)

        y_one_hot = np.zeros((len(y_aug), self.output_dim), dtype=np.float32)
        y_one_hot[np.arange(len(y_aug)), y_aug] = 1.0

        for _ in range(int(epochs)):
            z1 = x_norm @ self.w1 + self.b1
            a1 = np.tanh(z1)

            logits = a1 @ self.w2 + self.b2
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-6)

            d_logits = (probs - y_one_hot) / len(x_norm)
            d_w2 = a1.T @ d_logits + l2 * self.w2
            d_b2 = d_logits.sum(axis=0, keepdims=True)

            d_a1 = d_logits @ self.w2.T
            d_z1 = d_a1 * (1.0 - a1 * a1)
            d_w1 = x_norm.T @ d_z1 + l2 * self.w1
            d_b1 = d_z1.sum(axis=0, keepdims=True)

            self.w2 -= learning_rate * d_w2
            self.b2 -= learning_rate * d_b2
            self.w1 -= learning_rate * d_w1
            self.b1 -= learning_rate * d_b1

    def predict_proba(self, feature_rows: np.ndarray) -> np.ndarray:
        x = np.asarray(feature_rows, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected feature vectors with width {self.input_dim}, got {x.shape[1]}.")

        x_norm = self._normalize(x)
        a1 = np.tanh(x_norm @ self.w1 + self.b1)
        logits = a1 @ self.w2 + self.b2
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-6)


class NeuralGazeDetector:
    def __init__(self):
        self.face_cascade = _load_cascade("haarcascade_frontalface_default.xml")
        eye_cascade_name = "haarcascade_eye_tree_eyeglasses.xml"
        try:
            self.eye_cascade = _load_cascade(eye_cascade_name)
        except FileNotFoundError:
            self.eye_cascade = _load_cascade("haarcascade_eye.xml")
        self.model: TinyMLP | None = None

    def detect(self, frame: np.ndarray) -> FrameDetection | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        face_box = self._detect_face(gray)
        if face_box is None:
            return None

        left_eye_box, right_eye_box = self._detect_eyes(gray, face_box)
        left_eye = _crop(gray, left_eye_box)
        right_eye = _crop(gray, right_eye_box)
        if left_eye.size == 0 or right_eye.size == 0:
            return None

        features = self._extract_features(left_eye, right_eye)
        return FrameDetection(
            face_box=face_box,
            left_eye_box=left_eye_box,
            right_eye_box=right_eye_box,
            features=features,
        )

    def fit(self, feature_rows: list[np.ndarray], labels: list[str]) -> None:
        x = np.asarray(feature_rows, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("Expected a 2D feature matrix for gaze calibration.")
        self.model = TinyMLP(input_dim=x.shape[1])
        self.model.fit(x, labels)

    def predict_from_features(self, features: np.ndarray, detection: FrameDetection | None = None) -> GazePrediction:
        if self.model is None:
            raise RuntimeError("Gaze model is not calibrated yet.")

        probs = self.model.predict_proba(features)[0]
        label_index = int(np.argmax(probs))
        gaze_x = float(probs[LABEL_TO_INDEX[RIGHT_LABEL]] - probs[LABEL_TO_INDEX[LEFT_LABEL]])
        return GazePrediction(
            gaze_x=gaze_x,
            gaze_y=0.0,
            label=INDEX_TO_LABEL[label_index],
            confidence=float(probs[label_index]),
            probabilities=probs,
            detection=detection if detection is not None else FrameDetection((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), np.asarray(features, dtype=np.float32)),
        )

    def predict(self, frame: np.ndarray) -> GazePrediction | None:
        detection = self.detect(frame)
        if detection is None:
            return None
        return self.predict_from_features(detection.features, detection)

    def _detect_face(self, gray: np.ndarray) -> tuple[int, int, int, int] | None:
        detections = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(96, 96),
        )
        if len(detections) == 0:
            return None
        x, y, w, h = max(detections, key=lambda box: int(box[2]) * int(box[3]))
        return int(x), int(y), int(w), int(h)

    def _detect_eyes(self, gray: np.ndarray, face_box: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        x, y, w, h = face_box
        upper_h = max(int(round(h * 0.58)), 1)
        roi = gray[y : y + upper_h, x : x + w]

        eye_candidates = self.eye_cascade.detectMultiScale(
            roi,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(max(18, int(w * 0.12)), max(12, int(h * 0.08))),
        )

        candidates: list[tuple[int, int, int, int]] = []
        for ex, ey, ew, eh in eye_candidates:
            center_y = ey + eh / 2.0
            if center_y > upper_h * 0.8:
                continue
            if ew > w * 0.55 or eh > h * 0.35:
                continue
            candidates.append((int(x + ex), int(y + ey), int(ew), int(eh)))

        pair = self._select_eye_pair(candidates, face_box)
        if pair is None:
            return self._heuristic_eye_boxes(face_box, gray.shape)

        left_box = _grow_box(pair[0], gray.shape, x_scale=1.25, y_scale=1.55)
        right_box = _grow_box(pair[1], gray.shape, x_scale=1.25, y_scale=1.55)
        return left_box, right_box

    def _select_eye_pair(
        self,
        candidates: list[tuple[int, int, int, int]],
        face_box: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
        if len(candidates) < 2:
            return None

        fx, _, fw, _ = face_box
        face_center_x = fx + fw / 2.0
        best_pair = None
        best_score = -1e18

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                first = candidates[i]
                second = candidates[j]
                left_box, right_box = sorted((first, second), key=lambda box: box[0] + box[2] / 2.0)

                left_center_x = left_box[0] + left_box[2] / 2.0
                right_center_x = right_box[0] + right_box[2] / 2.0
                if left_center_x >= face_center_x or right_center_x <= face_center_x:
                    continue

                horizontal_gap = right_center_x - left_center_x
                if horizontal_gap < fw * 0.18:
                    continue

                left_center_y = left_box[1] + left_box[3] / 2.0
                right_center_y = right_box[1] + right_box[3] / 2.0
                y_penalty = abs(left_center_y - right_center_y)

                left_area = left_box[2] * left_box[3]
                right_area = right_box[2] * right_box[3]
                size_penalty = abs(left_area - right_area) / max(left_area, right_area, 1)

                score = horizontal_gap - 2.0 * y_penalty - 35.0 * size_penalty
                if score > best_score:
                    best_score = score
                    best_pair = (left_box, right_box)

        return best_pair

    def _heuristic_eye_boxes(
        self,
        face_box: tuple[int, int, int, int],
        frame_shape: tuple[int, int],
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        left_box = _face_relative_box(face_box, frame_shape, 0.33, 0.39, 0.26, 0.16)
        right_box = _face_relative_box(face_box, frame_shape, 0.67, 0.39, 0.26, 0.16)
        return left_box, right_box

    def _extract_features(self, left_eye: np.ndarray, right_eye: np.ndarray) -> np.ndarray:
        left_prepared = _prepare_eye(left_eye, flip_horizontal=False)
        right_prepared = _prepare_eye(right_eye, flip_horizontal=True)

        left_features = _eye_features(left_prepared)
        right_features = _eye_features(right_prepared)
        return np.concatenate([left_features, right_features]).astype(np.float32)


def _load_cascade(filename: str) -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / filename
    if not cascade_path.exists():
        raise FileNotFoundError(f"Could not find OpenCV cascade file: {cascade_path}")
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade classifier: {cascade_path}")
    return cascade


def _clip_box(box: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = box
    x = max(0, min(int(round(x)), frame_w - 1))
    y = max(0, min(int(round(y)), frame_h - 1))
    max_w = max(frame_w - x, 1)
    max_h = max(frame_h - y, 1)
    w = max(1, min(int(round(w)), max_w))
    h = max(1, min(int(round(h)), max_h))
    return x, y, w, h


def _grow_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    x_scale: float = 1.15,
    y_scale: float = 1.25,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    grown_w = w * x_scale
    grown_h = h * y_scale
    grown_x = center_x - grown_w / 2.0
    grown_y = center_y - grown_h / 2.0
    return _clip_box((int(round(grown_x)), int(round(grown_y)), int(round(grown_w)), int(round(grown_h))), frame_shape)


def _face_relative_box(
    face_box: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    center_x_ratio: float,
    center_y_ratio: float,
    width_ratio: float,
    height_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = face_box
    box_w = max(int(round(w * width_ratio)), 1)
    box_h = max(int(round(h * height_ratio)), 1)
    center_x = x + int(round(w * center_x_ratio))
    center_y = y + int(round(h * center_y_ratio))
    return _clip_box((center_x - box_w // 2, center_y - box_h // 2, box_w, box_h), frame_shape)


def _crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return frame[y : y + h, x : x + w]


def _prepare_eye(eye_crop: np.ndarray, flip_horizontal: bool) -> np.ndarray:
    eye = cv2.GaussianBlur(eye_crop, (5, 5), 0)
    eye = cv2.equalizeHist(eye)
    eye = cv2.resize(eye, EYE_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    if flip_horizontal:
        eye = cv2.flip(eye, 1)
    return eye


def _eye_features(eye_image: np.ndarray) -> np.ndarray:
    eye_float = eye_image.astype(np.float32) / 255.0
    darkness = 1.0 - eye_float

    low_res = cv2.resize(darkness, (12, 6), interpolation=cv2.INTER_AREA).reshape(-1)
    col_profile = cv2.resize(darkness, (12, 1), interpolation=cv2.INTER_AREA).reshape(-1)
    row_profile = cv2.resize(darkness, (1, 6), interpolation=cv2.INTER_AREA).reshape(-1)

    col_mass = darkness.sum(axis=0)
    row_mass = darkness.sum(axis=1)
    mass = float(col_mass.sum()) + 1e-6

    x_axis = np.linspace(0.0, 1.0, darkness.shape[1], dtype=np.float32)
    y_axis = np.linspace(0.0, 1.0, darkness.shape[0], dtype=np.float32)
    center_of_mass_x = float(np.dot(col_mass, x_axis) / mass)
    center_of_mass_y = float(np.dot(row_mass, y_axis) / mass)
    contrast = float(eye_float.std())

    return np.concatenate(
        [
            low_res.astype(np.float32),
            col_profile.astype(np.float32),
            row_profile.astype(np.float32),
            np.array([center_of_mass_x, center_of_mass_y, contrast], dtype=np.float32),
        ]
    )


def _draw_text_block(frame: np.ndarray, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def _safe_destroy_window(window_name: str) -> None:
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def initialize_camera(cam: int = 0):
    return cv2.VideoCapture(cam)


def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.rotate(frame, cv2.ROTATE_180)


def init_detector():
    return NeuralGazeDetector()


def run_detector(detector: NeuralGazeDetector, frame: np.ndarray) -> FrameDetection | None:
    return detector.detect(frame)


def train_model(detector: NeuralGazeDetector, feature_rows: list[np.ndarray], labels: list[str]) -> None:
    detector.fit(feature_rows, labels)


def predict(detector: NeuralGazeDetector, frame: np.ndarray) -> GazePrediction | None:
    return detector.predict(frame)


def gaze_xy(detector: NeuralGazeDetector, frame: np.ndarray) -> tuple[float, float] | None:
    prediction = predict(detector, frame)
    if prediction is None:
        return None
    return prediction.gaze_x, prediction.gaze_y


def draw_detection(frame: np.ndarray, detection: FrameDetection | None, prediction: GazePrediction | None = None) -> None:
    if detection is None:
        return

    cv2.rectangle(frame, detection.face_box[:2], (detection.face_box[0] + detection.face_box[2], detection.face_box[1] + detection.face_box[3]), (255, 255, 0), 2)
    cv2.rectangle(frame, detection.left_eye_box[:2], (detection.left_eye_box[0] + detection.left_eye_box[2], detection.left_eye_box[1] + detection.left_eye_box[3]), (0, 255, 0), 2)
    cv2.rectangle(frame, detection.right_eye_box[:2], (detection.right_eye_box[0] + detection.right_eye_box[2], detection.right_eye_box[1] + detection.right_eye_box[3]), (0, 255, 0), 2)

    if prediction is not None:
        text = f"{prediction.label} ({prediction.confidence:.2f})"
        cv2.putText(frame, text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def calibrate(
    detector: NeuralGazeDetector,
    cap,
    prep_seconds: float = 1.5,
    sample_seconds: float = 3.0,
    min_samples_per_label: int = 8,
    window_name: str = "Gaze Calibration",
) -> CalibrationResult | None:
    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    label_counts = {label: 0 for label in LABELS}

    for label, prompt in CALIBRATION_TARGETS:
        if not _run_calibration_stage(detector, cap, label, prompt, prep_seconds, sample_seconds, feature_rows, labels, label_counts, window_name):
            _safe_destroy_window(window_name)
            return None

    if any(label_counts[label] < min_samples_per_label for label in LABELS):
        print(f"Calibration failed. Sample counts were: {label_counts}")
        _safe_destroy_window(window_name)
        return None

    train_model(detector, feature_rows, labels)

    center_features = [row for row, label in zip(feature_rows, labels) if label == CENTER_LABEL]
    center_scores = [detector.predict_from_features(row).gaze_x for row in center_features]
    center_x = float(np.median(center_scores)) if center_scores else 0.0

    _safe_destroy_window(window_name)
    return CalibrationResult(center_x=center_x, center_y=0.0, label_counts=label_counts)


def _run_calibration_stage(
    detector: NeuralGazeDetector,
    cap,
    label: str,
    prompt: str,
    prep_seconds: float,
    sample_seconds: float,
    feature_rows: list[np.ndarray],
    labels: list[str],
    label_counts: dict[str, int],
    window_name: str,
) -> bool:
    prep_start = time.time()
    while time.time() - prep_start < prep_seconds:
        frame = get_frame(cap)
        if frame is None:
            continue

        detection = run_detector(detector, frame)
        draw_detection(frame, detection)
        remaining = max(prep_seconds - (time.time() - prep_start), 0.0)
        _draw_text_block(
            frame,
            [f"Get ready: {prompt}", f"Capture starts in {remaining:0.1f}s"],
            (10, 30),
            (0, 255, 255),
        )
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    sample_start = time.time()
    while time.time() - sample_start < sample_seconds:
        frame = get_frame(cap)
        if frame is None:
            continue

        detection = run_detector(detector, frame)
        if detection is not None:
            feature_rows.append(detection.features.copy())
            labels.append(label)
            label_counts[label] += 1

        draw_detection(frame, detection)
        remaining = max(sample_seconds - (time.time() - sample_start), 0.0)
        status = f"Samples: {label_counts[label]}"
        if detection is None:
            status = "Face / eyes not found"
        _draw_text_block(
            frame,
            [prompt, f"Hold still: {remaining:0.1f}s", status],
            (10, 30),
            (0, 255, 0),
        )
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    return True
