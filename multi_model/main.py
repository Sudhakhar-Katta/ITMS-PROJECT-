from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best.pt"

INPUT_IMAGES_DIR = BASE_DIR / "inputs" / "images"
INPUT_VIDEOS_DIR = BASE_DIR / "inputs" / "videos"

OUTPUT_IMAGES_DIR = BASE_DIR / "outputs" / "images"
OUTPUT_VIDEOS_DIR = BASE_DIR / "outputs" / "videos"

LOGS_DIR = BASE_DIR / "logs"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

DEVICE = "cpu"

# 640 is suitable for CPU testing.
# You can later change this to 768 for more accuracy,
# but CPU processing will become slower.
IMAGE_SIZE = 640

IOU_THRESHOLD = 0.45

# Independent per-class confidence thresholds.
#
# Smoking has lower recall, so its threshold starts lower.
# These can be adjusted after testing real cabin footage.
CLASS_THRESHOLDS = {
    0: 0.20,  # smoking
    1: 0.25,  # phone
    2: 0.25,  # drinking
}

CLASS_NAMES = {
    0: "smoking",
    1: "phone",
    2: "drinking",
}

# OpenCV uses BGR colors.
CLASS_COLORS = {
    0: (0, 140, 255),    # smoking
    1: (255, 0, 255),    # phone
    2: (255, 120, 0),    # drinking
}

ALERT_COLORS = {
    "safe": (60, 180, 75),
    "checking": (0, 215, 255),
    "danger": (0, 0, 255),
}


# ============================================================
# EVENT CONFIGURATION
# ============================================================

# Phone must remain detected for five seconds before the
# final PHONE USAGE DETECTED alert is activated.
PHONE_CONFIRM_SECONDS = 5.0

# Allows a short detection gap so one missed frame does not
# immediately reset the five-second phone timer.
PHONE_GAP_GRACE_SECONDS = 0.75

# Smoking and drinking alerts remain visible briefly to reduce
# visual flickering when a frame is missed.
IMMEDIATE_EVENT_HOLD_SECONDS = 0.75


# ============================================================
# SUPPORTED FILE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
}


# ============================================================
# DETECTION DATA CLASS
# ============================================================

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_required_directories() -> None:
    directories = [
        INPUT_IMAGES_DIR,
        INPUT_VIDEOS_DIR,
        OUTPUT_IMAGES_DIR,
        OUTPUT_VIDEOS_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# MODEL WRAPPER
# ============================================================

class ITMSDetector:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model was not found:\n{model_path}"
            )

        print("=" * 80)
        print("LOADING ITMS MODEL")
        print("=" * 80)

        print("Model:", model_path)
        print("Device:", DEVICE)
        print("Image size:", IMAGE_SIZE)

        self.model = YOLO(str(model_path))

        self.model_names = {
            int(class_id): str(class_name)
            for class_id, class_name
            in self.model.names.items()
        }

        print("Model classes:", self.model_names)
        print("Model loaded successfully.")

    def predict(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:
        minimum_confidence = min(
            CLASS_THRESHOLDS.values()
        )

        results = self.model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=minimum_confidence,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        if len(result.boxes) == 0:
            return []

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        class_ids = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        confidence_values = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        detections: list[Detection] = []

        frame_height, frame_width = frame.shape[:2]

        for box, class_id, confidence in zip(
            boxes,
            class_ids,
            confidence_values,
        ):
            if class_id not in CLASS_THRESHOLDS:
                continue

            required_confidence = (
                CLASS_THRESHOLDS[class_id]
            )

            if float(confidence) < required_confidence:
                continue

            x1, y1, x2, y2 = [
                int(value)
                for value in box
            ]

            x1 = max(0, min(x1, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y2 = max(0, min(y2, frame_height - 1))

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES[
                        class_id
                    ],
                    confidence=float(
                        confidence
                    ),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections


# ============================================================
# DRAW DETECTION BOXES
# ============================================================

def draw_detection_boxes(
    frame: np.ndarray,
    detections: Iterable[Detection],
) -> None:
    for detection in detections:
        color = CLASS_COLORS[
            detection.class_id
        ]

        cv2.rectangle(
            frame,
            (
                detection.x1,
                detection.y1,
            ),
            (
                detection.x2,
                detection.y2,
            ),
            color,
            2,
        )

        label = (
            f"{detection.class_name.upper()} "
            f"{detection.confidence:.2f}"
        )

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )

        label_width = text_size[0] + 12
        label_height = text_size[1] + baseline + 10

        label_y1 = max(
            0,
            detection.y1 - label_height,
        )

        cv2.rectangle(
            frame,
            (
                detection.x1,
                label_y1,
            ),
            (
                detection.x1 + label_width,
                detection.y1,
            ),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (
                detection.x1 + 6,
                detection.y1 - 7,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


# ============================================================
# DRAW PRESENTABLE ALERT PANEL
# ============================================================

def draw_alert_panel(
    frame: np.ndarray,
    alerts: list[tuple[str, str]],
) -> None:
    panel_x = 20
    panel_y = 20
    panel_width = min(
        600,
        frame.shape[1] - 40,
    )

    line_height = 48

    if not alerts:
        alerts = [
            (
                "NO TARGET ACTIVITY DETECTED",
                "safe",
            )
        ]

    panel_height = (
        20
        + line_height * len(alerts)
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            panel_x,
            panel_y,
        ),
        (
            panel_x + panel_width,
            panel_y + panel_height,
        ),
        (15, 15, 15),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.78,
        frame,
        0.22,
        0,
        frame,
    )

    for index, (
        alert_text,
        alert_level,
    ) in enumerate(alerts):
        y_position = (
            panel_y
            + 39
            + index * line_height
        )

        color = ALERT_COLORS[
            alert_level
        ]

        cv2.circle(
            frame,
            (
                panel_x + 18,
                y_position - 7,
            ),
            7,
            color,
            -1,
        )

        cv2.putText(
            frame,
            alert_text,
            (
                panel_x + 38,
                y_position,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )


# ============================================================
# DRAW FRAME INFORMATION
# ============================================================

def draw_frame_information(
    frame: np.ndarray,
    information: str,
) -> None:
    text_size, _ = cv2.getTextSize(
        information,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        1,
    )

    x_position = 15
    y_position = frame.shape[0] - 18

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            8,
            y_position - 25,
        ),
        (
            x_position + text_size[0] + 12,
            y_position + 8,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame,
    )

    cv2.putText(
        frame,
        information,
        (
            x_position,
            y_position,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# IMAGE ALERTS
# ============================================================

def create_image_alerts(
    detections: Iterable[Detection],
) -> list[tuple[str, str]]:
    detected_class_ids = {
        detection.class_id
        for detection in detections
    }

    alerts: list[tuple[str, str]] = []

    if 0 in detected_class_ids:
        alerts.append(
            (
                "SMOKING DETECTED",
                "danger",
            )
        )

    if 1 in detected_class_ids:
        alerts.append(
            (
                "PHONE USAGE DETECTED",
                "danger",
            )
        )

    if 2 in detected_class_ids:
        alerts.append(
            (
                "DRINKING DETECTED",
                "danger",
            )
        )

    return alerts


# ============================================================
# VIDEO EVENT TRACKER
# ============================================================

class IndependentEventTracker:
    def __init__(self) -> None:
        self.phone_start_time: float | None = None
        self.phone_last_seen_time: float | None = None
        self.phone_confirmed = False

        self.smoking_last_seen_time: float | None = None
        self.drinking_last_seen_time: float | None = None

        self.previous_active_states = {
            0: False,
            1: False,
            2: False,
        }

        self.event_counts = Counter()

    def update(
        self,
        timestamp_seconds: float,
        detections: Iterable[Detection],
    ) -> list[tuple[str, str]]:
        detected_class_ids = {
            detection.class_id
            for detection in detections
        }

        smoking_present = 0 in detected_class_ids
        phone_present = 1 in detected_class_ids
        drinking_present = 2 in detected_class_ids

        # ----------------------------------------------------
        # Smoking state
        # ----------------------------------------------------

        if smoking_present:
            self.smoking_last_seen_time = (
                timestamp_seconds
            )

        smoking_active = (
            self.smoking_last_seen_time is not None
            and (
                timestamp_seconds
                - self.smoking_last_seen_time
            )
            <= IMMEDIATE_EVENT_HOLD_SECONDS
        )

        # ----------------------------------------------------
        # Drinking state
        # ----------------------------------------------------

        if drinking_present:
            self.drinking_last_seen_time = (
                timestamp_seconds
            )

        drinking_active = (
            self.drinking_last_seen_time is not None
            and (
                timestamp_seconds
                - self.drinking_last_seen_time
            )
            <= IMMEDIATE_EVENT_HOLD_SECONDS
        )

        # ----------------------------------------------------
        # Phone state
        # ----------------------------------------------------

        if phone_present:
            if self.phone_start_time is None:
                self.phone_start_time = (
                    timestamp_seconds
                )

            self.phone_last_seen_time = (
                timestamp_seconds
            )

        else:
            phone_gap_exceeded = (
                self.phone_last_seen_time is None
                or (
                    timestamp_seconds
                    - self.phone_last_seen_time
                )
                > PHONE_GAP_GRACE_SECONDS
            )

            if phone_gap_exceeded:
                self.phone_start_time = None
                self.phone_last_seen_time = None
                self.phone_confirmed = False

        phone_visible_or_in_grace = (
            self.phone_start_time is not None
            and self.phone_last_seen_time is not None
            and (
                timestamp_seconds
                - self.phone_last_seen_time
            )
            <= PHONE_GAP_GRACE_SECONDS
        )

        phone_elapsed_seconds = 0.0

        if phone_visible_or_in_grace:
            phone_elapsed_seconds = max(
                0.0,
                timestamp_seconds
                - self.phone_start_time,
            )

            if (
                phone_elapsed_seconds
                >= PHONE_CONFIRM_SECONDS
            ):
                self.phone_confirmed = True

        # ----------------------------------------------------
        # Count independent event activations
        # ----------------------------------------------------

        current_active_states = {
            0: smoking_active,
            1: self.phone_confirmed,
            2: drinking_active,
        }

        for class_id, is_active in (
            current_active_states.items()
        ):
            was_active = (
                self.previous_active_states[
                    class_id
                ]
            )

            if is_active and not was_active:
                self.event_counts[class_id] += 1

        self.previous_active_states = (
            current_active_states
        )

        # ----------------------------------------------------
        # Build independent UI alerts
        # ----------------------------------------------------

        alerts: list[tuple[str, str]] = []

        if smoking_active:
            alerts.append(
                (
                    "SMOKING DETECTED",
                    "danger",
                )
            )

        if phone_visible_or_in_grace:
            if self.phone_confirmed:
                alerts.append(
                    (
                        "PHONE USAGE DETECTED",
                        "danger",
                    )
                )
            else:
                alerts.append(
                    (
                        "PHONE DETECTED - "
                        f"CHECKING "
                        f"{phone_elapsed_seconds:.1f}/"
                        f"{PHONE_CONFIRM_SECONDS:.1f}s",
                        "checking",
                    )
                )

        if drinking_active:
            alerts.append(
                (
                    "DRINKING DETECTED",
                    "danger",
                )
            )

        return alerts


# ============================================================
# IMAGE PROCESSING
# ============================================================

def find_files(
    directory: Path,
    extensions: set[str],
) -> list[Path]:
    return sorted(
        file_path
        for file_path in directory.rglob("*")
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in extensions
        )
    )


def process_images(
    detector: ITMSDetector,
    show_boxes: bool,
) -> None:
    image_paths = find_files(
        INPUT_IMAGES_DIR,
        IMAGE_EXTENSIONS,
    )

    print("\n" + "=" * 80)
    print("IMAGE TEST")
    print("=" * 80)

    print("Input folder:", INPUT_IMAGES_DIR)
    print("Images found:", len(image_paths))

    if not image_paths:
        print(
            "No images were found. Add images to:\n"
            f"{INPUT_IMAGES_DIR}"
        )
        return

    csv_path = (
        LOGS_DIR / "image_test_results.csv"
    )

    csv_rows: list[dict[str, object]] = []

    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):
        print(
            f"\n[{image_number}/{len(image_paths)}] "
            f"Processing: {image_path.name}"
        )

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            print(
                "Skipped: OpenCV could not read "
                f"{image_path}"
            )
            continue

        start_time = time.perf_counter()

        detections = detector.predict(frame)

        inference_time = (
            time.perf_counter() - start_time
        )

        output_frame = frame.copy()

        if show_boxes:
            draw_detection_boxes(
                output_frame,
                detections,
            )

        alerts = create_image_alerts(
            detections
        )

        draw_alert_panel(
            output_frame,
            alerts,
        )

        draw_frame_information(
            output_frame,
            (
                f"CPU inference: "
                f"{inference_time:.2f}s | "
                f"Detections: {len(detections)}"
            ),
        )

        output_path = (
            OUTPUT_IMAGES_DIR
            / (
                f"{image_path.stem}_detected"
                f"{image_path.suffix.lower()}"
            )
        )

        saved = cv2.imwrite(
            str(output_path),
            output_frame,
        )

        if not saved:
            raise RuntimeError(
                f"Could not save image: {output_path}"
            )

        detection_counts = Counter(
            detection.class_name
            for detection in detections
        )

        maximum_confidences: dict[str, float] = {}

        for detection in detections:
            current_maximum = (
                maximum_confidences.get(
                    detection.class_name,
                    0.0,
                )
            )

            maximum_confidences[
                detection.class_name
            ] = max(
                current_maximum,
                detection.confidence,
            )

        csv_rows.append(
            {
                "input_image": str(
                    image_path
                ),
                "output_image": str(
                    output_path
                ),
                "smoking_count": (
                    detection_counts[
                        "smoking"
                    ]
                ),
                "phone_count": (
                    detection_counts[
                        "phone"
                    ]
                ),
                "drinking_count": (
                    detection_counts[
                        "drinking"
                    ]
                ),
                "smoking_max_confidence": (
                    maximum_confidences.get(
                        "smoking",
                        0.0,
                    )
                ),
                "phone_max_confidence": (
                    maximum_confidences.get(
                        "phone",
                        0.0,
                    )
                ),
                "drinking_max_confidence": (
                    maximum_confidences.get(
                        "drinking",
                        0.0,
                    )
                ),
                "inference_seconds": (
                    inference_time
                ),
            }
        )

        print("Detections:")

        if not detections:
            print("  None")
        else:
            for detection in detections:
                print(
                    f"  {detection.class_name}: "
                    f"{detection.confidence:.3f}"
                )

        print("Saved:", output_path)

    if csv_rows:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(
                    csv_rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(csv_rows)

        print("\nImage results CSV:", csv_path)

    print("\nIMAGE TEST COMPLETE")


# ============================================================
# VIDEO PROCESSING
# ============================================================

def create_video_writer(
    output_path: Path,
    frames_per_second: float,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    codec = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        codec,
        frames_per_second,
        (
            frame_width,
            frame_height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video:\n"
            f"{output_path}"
        )

    return writer


def process_single_video(
    detector: ITMSDetector,
    video_path: Path,
    show_boxes: bool,
    process_every_n_frames: int,
) -> None:
    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        print(
            "Skipped: Could not open video:\n"
            f"{video_path}"
        )
        return

    frames_per_second = capture.get(
        cv2.CAP_PROP_FPS
    )

    if frames_per_second <= 0:
        frames_per_second = 30.0

    frame_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    output_path = (
        OUTPUT_VIDEOS_DIR
        / f"{video_path.stem}_detected.mp4"
    )

    log_path = (
        LOGS_DIR
        / f"{video_path.stem}_detections.csv"
    )

    writer = create_video_writer(
        output_path=output_path,
        frames_per_second=frames_per_second,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    tracker = IndependentEventTracker()

    frame_number = 0
    last_detections: list[Detection] = []

    log_rows: list[dict[str, object]] = []

    processing_start_time = (
        time.perf_counter()
    )

    print("\n" + "-" * 80)
    print("Processing video:", video_path.name)
    print("Resolution:", frame_width, "x", frame_height)
    print("FPS:", round(frames_per_second, 2))
    print("Frames:", total_frames)
    print(
        "Process every N frames:",
        process_every_n_frames,
    )

    while True:
        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        video_time_seconds = (
            frame_number
            / frames_per_second
        )

        should_run_inference = (
            frame_number == 1
            or (
                frame_number
                % process_every_n_frames
            )
            == 0
        )

        if should_run_inference:
            last_detections = (
                detector.predict(frame)
            )

        detections = last_detections

        alerts = tracker.update(
            timestamp_seconds=(
                video_time_seconds
            ),
            detections=detections,
        )

        output_frame = frame.copy()

        if show_boxes:
            draw_detection_boxes(
                output_frame,
                detections,
            )

        draw_alert_panel(
            output_frame,
            alerts,
        )

        draw_frame_information(
            output_frame,
            (
                f"Video time: "
                f"{video_time_seconds:.1f}s | "
                f"Frame: {frame_number}/"
                f"{total_frames if total_frames > 0 else '?'}"
            ),
        )

        writer.write(output_frame)

        if should_run_inference:
            alert_text = (
                " | ".join(
                    alert[0]
                    for alert in alerts
                )
                if alerts
                else "NONE"
            )

            if detections:
                for detection in detections:
                    log_rows.append(
                        {
                            "frame_number": (
                                frame_number
                            ),
                            "video_time_seconds": (
                                round(
                                    video_time_seconds,
                                    3,
                                )
                            ),
                            "class_id": (
                                detection.class_id
                            ),
                            "class_name": (
                                detection.class_name
                            ),
                            "confidence": (
                                round(
                                    detection.confidence,
                                    6,
                                )
                            ),
                            "x1": detection.x1,
                            "y1": detection.y1,
                            "x2": detection.x2,
                            "y2": detection.y2,
                            "active_alerts": (
                                alert_text
                            ),
                        }
                    )

        if (
            frame_number % 100 == 0
            or (
                total_frames > 0
                and frame_number
                == total_frames
            )
        ):
            print(
                f"Processed "
                f"{frame_number}/"
                f"{total_frames if total_frames > 0 else '?'} "
                f"frames"
            )

    processing_seconds = (
        time.perf_counter()
        - processing_start_time
    )

    capture.release()
    writer.release()

    if log_rows:
        with log_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer_csv = csv.DictWriter(
                csv_file,
                fieldnames=list(
                    log_rows[0].keys()
                ),
            )

            writer_csv.writeheader()
            writer_csv.writerows(
                log_rows
            )

    print("\nVideo completed:", video_path.name)
    print("Output:", output_path)
    print("Log:", log_path)
    print(
        "Processing time:",
        round(processing_seconds, 2),
        "seconds",
    )

    print("Independent event counts:")
    print(
        "  Smoking:",
        tracker.event_counts[0],
    )
    print(
        "  Phone usage:",
        tracker.event_counts[1],
    )
    print(
        "  Drinking:",
        tracker.event_counts[2],
    )


def process_videos(
    detector: ITMSDetector,
    show_boxes: bool,
    process_every_n_frames: int,
) -> None:
    video_paths = find_files(
        INPUT_VIDEOS_DIR,
        VIDEO_EXTENSIONS,
    )

    print("\n" + "=" * 80)
    print("VIDEO TEST")
    print("=" * 80)

    print("Input folder:", INPUT_VIDEOS_DIR)
    print("Videos found:", len(video_paths))

    if not video_paths:
        print(
            "No videos were found. Add videos to:\n"
            f"{INPUT_VIDEOS_DIR}"
        )
        return

    for video_number, video_path in enumerate(
        video_paths,
        start=1,
    ):
        print(
            f"\nVIDEO "
            f"{video_number}/{len(video_paths)}"
        )

        process_single_video(
            detector=detector,
            video_path=video_path,
            show_boxes=show_boxes,
            process_every_n_frames=(
                process_every_n_frames
            ),
        )

    print("\nVIDEO TEST COMPLETE")


# ============================================================
# WEBCAM TEST
# ============================================================

def test_webcam(
    detector: ITMSDetector,
    camera_index: int,
    show_boxes: bool,
    process_every_n_frames: int,
) -> None:
    capture = cv2.VideoCapture(
        camera_index
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera index "
            f"{camera_index}."
        )

    capture.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    tracker = IndependentEventTracker()

    frame_number = 0
    last_detections: list[Detection] = []

    session_start_time = time.monotonic()

    timestamp_text = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        OUTPUT_VIDEOS_DIR
        / f"webcam_test_{timestamp_text}.mp4"
    )

    writer: cv2.VideoWriter | None = None

    print("\n" + "=" * 80)
    print("WEBCAM TEST")
    print("=" * 80)
    print("Press Q to stop.")
    print("Camera index:", camera_index)

    while True:
        success, frame = capture.read()

        if not success:
            print("Could not read webcam frame.")
            break

        frame_number += 1

        if writer is None:
            frame_height, frame_width = (
                frame.shape[:2]
            )

            writer = create_video_writer(
                output_path=output_path,
                frames_per_second=20.0,
                frame_width=frame_width,
                frame_height=frame_height,
            )

        elapsed_seconds = (
            time.monotonic()
            - session_start_time
        )

        should_run_inference = (
            frame_number == 1
            or (
                frame_number
                % process_every_n_frames
            )
            == 0
        )

        if should_run_inference:
            last_detections = (
                detector.predict(frame)
            )

        detections = last_detections

        alerts = tracker.update(
            timestamp_seconds=(
                elapsed_seconds
            ),
            detections=detections,
        )

        output_frame = frame.copy()

        if show_boxes:
            draw_detection_boxes(
                output_frame,
                detections,
            )

        draw_alert_panel(
            output_frame,
            alerts,
        )

        draw_frame_information(
            output_frame,
            (
                f"Live time: "
                f"{elapsed_seconds:.1f}s | "
                "Press Q to stop"
            ),
        )

        writer.write(output_frame)

        cv2.imshow(
            "ITMS Driver Intelligence Test",
            output_frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    capture.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    print("\nWebcam test finished.")
    print("Saved recording:", output_path)

    print("Independent event counts:")
    print(
        "  Smoking:",
        tracker.event_counts[0],
    )
    print(
        "  Phone usage:",
        tracker.event_counts[1],
    )
    print(
        "  Drinking:",
        tracker.event_counts[2],
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the ITMS smoking, phone, "
            "and drinking model on CPU."
        )
    )

    parser.add_argument(
        "mode",
        choices=[
            "images",
            "videos",
            "webcam",
            "all",
        ],
        help=(
            "Choose images, videos, webcam, "
            "or all."
        ),
    )

    parser.add_argument(
        "--hide-boxes",
        action="store_true",
        help=(
            "Hide detection boxes and show "
            "only the alert panel."
        ),
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help=(
            "Webcam index. Default is 0."
        ),
    )

    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help=(
            "Run inference every N frames. "
            "Default is every frame."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    create_required_directories()

    arguments = parse_arguments()

    if arguments.process_every < 1:
        raise ValueError(
            "--process-every must be at least 1."
        )

    # Leave one logical CPU core free for Windows.
    available_threads = os.cpu_count() or 2

    torch.set_num_threads(
        max(
            1,
            available_threads - 1,
        )
    )

    print("=" * 80)
    print("ITMS MULTI-MODEL CPU TEST")
    print("=" * 80)

    print("Project folder:", BASE_DIR)
    print("PyTorch:", torch.__version__)
    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )
    print(
        "PyTorch CPU threads:",
        torch.get_num_threads(),
    )

    show_boxes = not arguments.hide_boxes

    print("Show boxes:", show_boxes)

    detector = ITMSDetector(
        MODEL_PATH
    )

    if arguments.mode == "images":
        process_images(
            detector=detector,
            show_boxes=show_boxes,
        )

    elif arguments.mode == "videos":
        process_videos(
            detector=detector,
            show_boxes=show_boxes,
            process_every_n_frames=(
                arguments.process_every
            ),
        )

    elif arguments.mode == "webcam":
        test_webcam(
            detector=detector,
            camera_index=arguments.camera,
            show_boxes=show_boxes,
            process_every_n_frames=(
                arguments.process_every
            ),
        )

    elif arguments.mode == "all":
        process_images(
            detector=detector,
            show_boxes=show_boxes,
        )

        process_videos(
            detector=detector,
            show_boxes=show_boxes,
            process_every_n_frames=(
                arguments.process_every
            ),
        )


if __name__ == "__main__":
    main()