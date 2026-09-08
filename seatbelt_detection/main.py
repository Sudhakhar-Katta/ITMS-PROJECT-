import argparse
import csv
import datetime as dt
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

OBJECT_DETECTION_MODEL_PATH = BASE_DIR / "models" / "best.pt"
PREDICTOR_MODEL_PATH = BASE_DIR / "models" / "keras_model.h5"

INPUTS_DIR = BASE_DIR / "inputs"
INPUT_VIDEOS_DIR = INPUTS_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
EVIDENCE_DIR = LOG_DIR / "evidence"
METRICS_DIR = LOG_DIR / "metrics"

DEFAULT_INPUT_VIDEO = INPUT_VIDEOS_DIR / "test_1.mp4"

CLASS_NAMES = {
    0: "No Seatbelt worn",
    1: "Seatbelt Worn",
}

DRIVER_SIDE = "right"
CLASSIFICATION_THRESHOLD = 0.95
DETECTION_THRESHOLD = 0.25
NO_SEATBELT_TRIGGER_SECONDS = 5.0
PREDICTION_WINDOW_SECONDS = 1.0
NO_SEATBELT_VOTE_RATIO = 0.60
SEATBELT_VOTE_RATIO = 0.70
UNKNOWN_GRACE_SECONDS = 2.0
WARNING_RESET_GAP_SECONDS = 5.0
SEATBELT_CONFIRM_SECONDS = 2.0
LAST_DRIVER_BOX_GRACE_SECONDS = 1.0
PROCESS_EVERY_N_FRAMES = 1
MAX_FRAMES = 0

STATE_SAFE = "SAFE"
STATE_WARNING = "WARNING"
STATE_VIOLATION = "VIOLATION"

COLOR_GREEN = (0, 210, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_WHITE = (255, 255, 255)


def create_required_directories() -> None:
    for directory in (
        INPUTS_DIR,
        INPUT_VIDEOS_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        EVIDENCE_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def draw_box(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.60,
    thickness: int = 2,
) -> None:
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_status_banner(
    frame: np.ndarray,
    text: str,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.85
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    x1 = 20
    y1 = 20
    x2 = x1 + text_width + 24
    y2 = y1 + text_height + baseline + 24

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 12, y2 - baseline - 10),
        font,
        scale,
        COLOR_WHITE,
        thickness,
        cv2.LINE_AA,
    )


def create_event_log(event_log_path: Path) -> None:
    with event_log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "event_id",
                "event_type",
                "system_timestamp",
                "video_time_seconds",
                "frame_number",
                "no_seatbelt_duration_seconds",
                "raw_prediction",
                "stable_prediction",
                "confidence",
                "evidence_path",
            ]
        )


def append_event(
    event_log_path: Path,
    event_id: str,
    video_time: float,
    frame_number: int,
    no_seatbelt_duration: float,
    raw_prediction: str,
    stable_prediction: str,
    confidence: float,
    evidence_path: Path,
) -> None:
    with event_log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                event_id,
                "No Seatbelt Worn",
                dt.datetime.now().isoformat(timespec="seconds"),
                round(video_time, 3),
                frame_number,
                round(no_seatbelt_duration, 3),
                raw_prediction,
                stable_prediction,
                round(confidence, 4),
                str(evidence_path),
            ]
        )


class SeatbeltDetector:
    def __init__(
        self,
        object_detection_model_path: Path = OBJECT_DETECTION_MODEL_PATH,
        predictor_model_path: Path = PREDICTOR_MODEL_PATH,
        driver_side: str = DRIVER_SIDE,
        classification_threshold: float = CLASSIFICATION_THRESHOLD,
        detection_threshold: float = DETECTION_THRESHOLD,
    ) -> None:
        self.object_detection_model_path = Path(object_detection_model_path)
        self.predictor_model_path = Path(predictor_model_path)
        self.driver_side = driver_side
        self.classification_threshold = classification_threshold
        self.detection_threshold = detection_threshold
        self.prediction_history: deque[dict[str, Any]] = deque()

        try:
            import tensorflow as tf
            import torch
            from keras.models import load_model
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Seatbelt detection dependencies are not installed. "
                "Install the root requirements.txt before creating SeatbeltDetector."
            ) from error

        if not self.object_detection_model_path.exists():
            raise FileNotFoundError(
                f"Object detector model was not found: {self.object_detection_model_path}"
            )

        if not self.predictor_model_path.exists():
            raise FileNotFoundError(
                f"Seatbelt classifier model was not found: {self.predictor_model_path}"
            )

        print("=" * 80)
        print("ITMS Seatbelt Module")
        print("=" * 80)
        print("TensorFlow version:", tf.__version__)
        print("Torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("Driver image side:", self.driver_side)
        print("Classifier:", self.predictor_model_path)
        print("Object detector:", self.object_detection_model_path)

        self.predictor = load_model(str(self.predictor_model_path), compile=False)

        self.detector = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(self.object_detection_model_path),
            force_reload=False,
        )
        self.detector.conf = self.detection_threshold
        self.detector.iou = 0.45

        print("Detector classes:", self.detector.names)
        print("=" * 80)

    def classify_crop(self, rgb_crop: np.ndarray) -> tuple[str, float]:
        if rgb_crop is None or rgb_crop.size == 0:
            return "Unknown", 0.0

        resized = cv2.resize(rgb_crop, (224, 224), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        batch = np.expand_dims(normalized, axis=0)

        model_output = self.predictor.predict(batch, verbose=0)
        class_index = int(np.argmax(model_output[0]))
        confidence = float(model_output[0][class_index])

        if confidence < self.classification_threshold:
            return "Unknown", confidence

        return CLASS_NAMES[class_index], confidence

    def select_driver_box(
        self,
        boxes: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> dict[str, Any] | None:
        if not boxes:
            return None

        minimum_area = frame_width * frame_height * 0.02
        valid_boxes = []

        for candidate in boxes:
            x1, y1, x2, y2 = candidate["bbox"]
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            if box_width <= 0 or box_height <= 0:
                continue

            if box_area < minimum_area:
                continue

            valid_boxes.append(candidate)

        if not valid_boxes:
            return None

        if self.driver_side.lower() == "right":
            side_candidates = [
                candidate
                for candidate in valid_boxes
                if candidate["center_x"] >= frame_width * 0.38
            ]

            if side_candidates:
                return max(side_candidates, key=lambda candidate: candidate["center_x"])

            return max(valid_boxes, key=lambda candidate: candidate["center_x"])

        if self.driver_side.lower() == "left":
            side_candidates = [
                candidate
                for candidate in valid_boxes
                if candidate["center_x"] <= frame_width * 0.62
            ]

            if side_candidates:
                return min(side_candidates, key=lambda candidate: candidate["center_x"])

            return min(valid_boxes, key=lambda candidate: candidate["center_x"])

        raise ValueError("driver_side must be either 'left' or 'right'.")

    def predict(self, frame: np.ndarray) -> dict[str, Any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = self.detector(rgb_frame)
        raw_boxes = detection_results.xyxy[0].cpu().numpy()
        frame_height, frame_width = frame.shape[:2]

        candidates = []

        for raw_box in raw_boxes:
            x1, y1, x2, y2, detection_score, class_id = raw_box

            x1 = max(0, min(int(x1), frame_width - 1))
            x2 = max(0, min(int(x2), frame_width - 1))
            y1 = max(0, min(int(y1), frame_height - 1))
            y2 = max(0, min(int(y2), frame_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            candidates.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "center_x": (x1 + x2) / 2.0,
                    "detection_score": float(detection_score),
                    "class_id": int(class_id),
                }
            )

        driver_candidate = self.select_driver_box(
            candidates,
            frame_width,
            frame_height,
        )

        if driver_candidate is None:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "bbox": None,
                "detection_score": 0.0,
            }

        x1, y1, x2, y2 = driver_candidate["bbox"]
        driver_crop = rgb_frame[y1:y2, x1:x2]
        prediction, confidence = self.classify_crop(driver_crop)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "bbox": driver_candidate["bbox"],
            "detection_score": driver_candidate["detection_score"],
        }

    def update_prediction_history(
        self,
        video_time: float,
        prediction: str,
        confidence: float,
    ) -> tuple[str, dict[str, float]]:
        self.prediction_history.append(
            {
                "time": video_time,
                "prediction": prediction,
                "confidence": confidence,
            }
        )

        oldest_allowed_time = video_time - PREDICTION_WINDOW_SECONDS

        while (
            self.prediction_history
            and self.prediction_history[0]["time"] < oldest_allowed_time
        ):
            self.prediction_history.popleft()

        total_predictions = len(self.prediction_history)

        if total_predictions == 0:
            return "Unknown", {
                "no_seatbelt_ratio": 0.0,
                "seatbelt_ratio": 0.0,
                "unknown_ratio": 1.0,
            }

        no_seatbelt_count = sum(
            item["prediction"] == "No Seatbelt worn"
            for item in self.prediction_history
        )
        seatbelt_count = sum(
            item["prediction"] == "Seatbelt Worn" for item in self.prediction_history
        )
        unknown_count = sum(
            item["prediction"] == "Unknown" for item in self.prediction_history
        )

        vote_details = {
            "no_seatbelt_ratio": no_seatbelt_count / total_predictions,
            "seatbelt_ratio": seatbelt_count / total_predictions,
            "unknown_ratio": unknown_count / total_predictions,
        }

        if vote_details["no_seatbelt_ratio"] >= NO_SEATBELT_VOTE_RATIO:
            return "No Seatbelt worn", vote_details

        if vote_details["seatbelt_ratio"] >= SEATBELT_VOTE_RATIO:
            return "Seatbelt Worn", vote_details

        return "Unknown", vote_details


def process_video(
    input_video: Path,
    detector: SeatbeltDetector,
    output_video_path: Path,
    event_log_path: Path,
    metrics_path: Path,
    evidence_dir: Path,
    process_every_n_frames: int = PROCESS_EVERY_N_FRAMES,
    max_frames: int = MAX_FRAMES,
    show_preview: bool = False,
    save_output_video: bool = True,
) -> None:
    capture = cv2.VideoCapture(str(input_video))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    writer = None

    if save_output_video:
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_video_path}")

    create_event_log(event_log_path)

    print()
    print("Input video:", input_video)
    print("Resolution:", f"{frame_width} x {frame_height}")
    print("Source FPS:", fps)
    print("Total frames:", total_frames)
    print("Driver image side:", detector.driver_side)
    print("Prediction smoothing window:", PREDICTION_WINDOW_SECONDS, "seconds")
    print("Unknown grace:", UNKNOWN_GRACE_SECONDS, "seconds")
    print("Seatbelt confirmation:", SEATBELT_CONFIRM_SECONDS, "seconds")
    print("Red-event threshold:", NO_SEATBELT_TRIGGER_SECONDS, "seconds")
    print("Output video:", output_video_path if save_output_video else "disabled")
    print()
    print("Analysing video on CPU...")
    print("=" * 80)

    current_state = STATE_SAFE
    frame_number = 0
    processed_frames = 0
    event_count = 0
    no_seatbelt_duration = 0.0
    seatbelt_confirm_duration = 0.0
    last_no_seatbelt_evidence_time = None
    last_driver_bbox = None
    last_driver_bbox_time = None
    previous_video_time = None
    last_raw_result = {
        "prediction": "Unknown",
        "confidence": 0.0,
        "bbox": None,
    }
    last_stable_prediction = "Unknown"
    status_counts = {
        STATE_SAFE: 0,
        STATE_WARNING: 0,
        STATE_VIOLATION: 0,
    }
    raw_prediction_counts = {
        "Seatbelt Worn": 0,
        "No Seatbelt worn": 0,
        "Unknown": 0,
    }
    processing_start = time.time()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            frame_number += 1
            video_time = (frame_number - 1) / fps

            if previous_video_time is None:
                frame_delta = 1.0 / fps
            else:
                frame_delta = max(0.0, video_time - previous_video_time)

            previous_video_time = video_time
            inference_performed = frame_number % process_every_n_frames == 0

            if inference_performed:
                last_raw_result = detector.predict(frame)
                processed_frames += 1
                raw_prediction = last_raw_result["prediction"]
                raw_confidence = last_raw_result["confidence"]
                detected_bbox = last_raw_result["bbox"]

                if raw_prediction not in raw_prediction_counts:
                    raw_prediction = "Unknown"

                raw_prediction_counts[raw_prediction] += 1

                last_stable_prediction, vote_details = (
                    detector.update_prediction_history(
                        video_time,
                        raw_prediction,
                        raw_confidence,
                    )
                )

                if detected_bbox is not None:
                    last_driver_bbox = detected_bbox
                    last_driver_bbox_time = video_time

            else:
                raw_prediction = last_raw_result["prediction"]
                raw_confidence = last_raw_result["confidence"]
                vote_details = {
                    "no_seatbelt_ratio": 0.0,
                    "seatbelt_ratio": 0.0,
                    "unknown_ratio": 0.0,
                }

            driver_bbox = last_raw_result["bbox"]

            if (
                driver_bbox is None
                and last_driver_bbox is not None
                and last_driver_bbox_time is not None
                and video_time - last_driver_bbox_time <= LAST_DRIVER_BOX_GRACE_SECONDS
            ):
                driver_bbox = last_driver_bbox

            stable_prediction = last_stable_prediction

            if current_state == STATE_VIOLATION:
                if stable_prediction == "Seatbelt Worn":
                    seatbelt_confirm_duration += frame_delta

                    if seatbelt_confirm_duration >= SEATBELT_CONFIRM_SECONDS:
                        current_state = STATE_SAFE
                        no_seatbelt_duration = 0.0
                        seatbelt_confirm_duration = 0.0
                        last_no_seatbelt_evidence_time = None
                        print(
                            f"\n[EVENT CLEARED] Seatbelt confirmed continuously for "
                            f"{SEATBELT_CONFIRM_SECONDS:.1f}s at video time "
                            f"{video_time:.2f}s"
                        )
                else:
                    seatbelt_confirm_duration = 0.0
            else:
                if stable_prediction == "No Seatbelt worn":
                    current_state = STATE_WARNING
                    no_seatbelt_duration += frame_delta
                    last_no_seatbelt_evidence_time = video_time
                    seatbelt_confirm_duration = 0.0

                elif stable_prediction == "Unknown":
                    seatbelt_confirm_duration = 0.0

                    if (
                        current_state == STATE_WARNING
                        and last_no_seatbelt_evidence_time is not None
                    ):
                        unknown_gap = video_time - last_no_seatbelt_evidence_time

                        if unknown_gap <= UNKNOWN_GRACE_SECONDS:
                            no_seatbelt_duration += frame_delta
                        elif unknown_gap <= WARNING_RESET_GAP_SECONDS:
                            pass
                        else:
                            current_state = STATE_SAFE
                            no_seatbelt_duration = 0.0
                            last_no_seatbelt_evidence_time = None

                elif stable_prediction == "Seatbelt Worn":
                    if current_state == STATE_WARNING:
                        seatbelt_confirm_duration += frame_delta

                        if seatbelt_confirm_duration >= SEATBELT_CONFIRM_SECONDS:
                            current_state = STATE_SAFE
                            no_seatbelt_duration = 0.0
                            seatbelt_confirm_duration = 0.0
                            last_no_seatbelt_evidence_time = None
                    else:
                        current_state = STATE_SAFE
                        no_seatbelt_duration = 0.0
                        seatbelt_confirm_duration = 0.0
                        last_no_seatbelt_evidence_time = None

                if (
                    current_state == STATE_WARNING
                    and no_seatbelt_duration >= NO_SEATBELT_TRIGGER_SECONDS
                ):
                    current_state = STATE_VIOLATION
                    event_count += 1
                    event_id = f"SEATBELT_{output_video_path.stem}_{event_count:04d}"
                    evidence_name = (
                        f"{event_id}_frame_{frame_number}_time_{video_time:.2f}s.jpg"
                    )
                    evidence_path = evidence_dir / evidence_name
                    evidence_frame = frame.copy()

                    draw_status_banner(
                        evidence_frame,
                        "ITMS Seatbelt: NO SEATBELT WORN",
                        COLOR_RED,
                    )

                    if driver_bbox is not None:
                        draw_box(evidence_frame, driver_bbox, COLOR_RED)

                    cv2.imwrite(str(evidence_path), evidence_frame)

                    append_event(
                        event_log_path=event_log_path,
                        event_id=event_id,
                        video_time=video_time,
                        frame_number=frame_number,
                        no_seatbelt_duration=no_seatbelt_duration,
                        raw_prediction=raw_prediction,
                        stable_prediction=stable_prediction,
                        confidence=raw_confidence,
                        evidence_path=evidence_path,
                    )

                    print(
                        f"\n[SEATBELT EVENT] {event_id}"
                        f" | video_time={video_time:.2f}s"
                        f" | duration={no_seatbelt_duration:.2f}s"
                        f" | stable_prediction={stable_prediction}"
                    )

            status_counts[current_state] += 1

            if current_state == STATE_VIOLATION:
                system_status = "NO SEATBELT WORN"
                display_color = COLOR_RED
                driver_label = "Driver: NO SEATBELT WORN"
            elif current_state == STATE_WARNING:
                system_status = "WARNING"
                display_color = COLOR_YELLOW
                driver_label = (
                    "Driver: WARNING "
                    f"{min(no_seatbelt_duration, NO_SEATBELT_TRIGGER_SECONDS):.1f}/"
                    f"{NO_SEATBELT_TRIGGER_SECONDS:.1f}s"
                )
            else:
                if stable_prediction == "Seatbelt Worn":
                    system_status = "SEATBELT OK"
                    display_color = COLOR_GREEN
                    driver_label = f"Driver: Seatbelt Worn {raw_confidence:.2f}"
                else:
                    system_status = "WARNING"
                    display_color = COLOR_YELLOW
                    driver_label = "Driver: WARNING - CHECKING SEATBELT"

            if driver_bbox is not None:
                draw_box(frame, driver_bbox, display_color)
                x1, y1, _, _ = driver_bbox
                draw_text(frame, driver_label, (x1, max(30, y1 - 12)), display_color)

            draw_status_banner(frame, f"ITMS Seatbelt: {system_status}", display_color)
            draw_text(
                frame,
                (
                    f"Video time: {video_time:.2f}s"
                    f" | Frame: {frame_number}"
                    f" | Driver side: {detector.driver_side}"
                ),
                (20, 85),
                COLOR_WHITE,
                scale=0.56,
                thickness=1,
            )
            draw_text(
                frame,
                (
                    f"Raw: {raw_prediction}"
                    f" | Stable: {stable_prediction}"
                    f" | No-seatbelt vote: "
                    f"{vote_details['no_seatbelt_ratio']:.2f}"
                    f" | Seatbelt vote: "
                    f"{vote_details['seatbelt_ratio']:.2f}"
                ),
                (20, 112),
                COLOR_WHITE,
                scale=0.48,
                thickness=1,
            )

            if writer is not None:
                writer.write(frame)

            if show_preview:
                cv2.imshow("ITMS Seatbelt Processing", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Processing stopped by user.")
                    break

            if frame_number % 100 == 0:
                percentage = (
                    (frame_number / total_frames) * 100 if total_frames > 0 else 0.0
                )
                print(
                    f"\rProcessed {frame_number}/{total_frames} "
                    f"frames ({percentage:.1f}%)"
                    f" | State: {current_state}"
                    f" | Timer: {no_seatbelt_duration:.2f}s",
                    end="",
                    flush=True,
                )

            if max_frames > 0 and frame_number >= max_frames:
                break
    finally:
        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    processing_seconds = time.time() - processing_start
    metrics = {
        "module": "ITMS Seatbelt Reminder",
        "version": "temporal_smoothing_v2",
        "run_timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input_video": str(input_video),
        "output_video": str(output_video_path),
        "driver_image_side": detector.driver_side,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "detection_threshold": DETECTION_THRESHOLD,
        "no_seatbelt_trigger_seconds": NO_SEATBELT_TRIGGER_SECONDS,
        "prediction_window_seconds": PREDICTION_WINDOW_SECONDS,
        "no_seatbelt_vote_ratio": NO_SEATBELT_VOTE_RATIO,
        "seatbelt_vote_ratio": SEATBELT_VOTE_RATIO,
        "unknown_grace_seconds": UNKNOWN_GRACE_SECONDS,
        "warning_reset_gap_seconds": WARNING_RESET_GAP_SECONDS,
        "seatbelt_confirm_seconds": SEATBELT_CONFIRM_SECONDS,
        "last_driver_box_grace_seconds": LAST_DRIVER_BOX_GRACE_SECONDS,
        "red_event_latched": True,
        "source_fps": fps,
        "source_total_frames": total_frames,
        "frames_read": frame_number,
        "frames_inferred": processed_frames,
        "processing_seconds": round(processing_seconds, 3),
        "processing_fps": round(frame_number / processing_seconds, 3)
        if processing_seconds > 0
        else 0.0,
        "event_count": event_count,
        "state_counts": status_counts,
        "raw_prediction_counts": raw_prediction_counts,
        "event_log": str(event_log_path),
        "evidence_directory": str(evidence_dir),
        "final_state": current_state,
        "final_status": "COMPLETE",
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print()
    print()
    print("=" * 80)
    print("ITMS SEATBELT ANALYSIS COMPLETE")
    print("=" * 80)
    print("Frames read:", frame_number)
    print("Frames inferred:", processed_frames)
    print("Seatbelt events:", event_count)
    print("Final state:", current_state)
    print("Processing time:", f"{processing_seconds:.2f} seconds")
    print("Output video:", output_video_path if save_output_video else "disabled")
    print("Event log:", event_log_path)
    print("Metrics:", metrics_path)
    print("Evidence folder:", evidence_dir)
    print("=" * 80)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ITMS seatbelt detection.")
    parser.add_argument(
        "--input-video",
        type=Path,
        default=DEFAULT_INPUT_VIDEO,
        help="Video path to process. Defaults to seatbelt_detection/inputs/videos/test_1.mp4.",
    )
    parser.add_argument(
        "--driver-side",
        choices=["left", "right"],
        default=DRIVER_SIDE,
        help="Side of the displayed frame containing the driver.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=PROCESS_EVERY_N_FRAMES,
        help="Run inference every N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES,
        help="Stop after N frames. Use 0 for the full video.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show OpenCV processing preview.",
    )
    parser.add_argument(
        "--no-output-video",
        action="store_true",
        help="Disable writing the annotated output video.",
    )
    return parser.parse_args()


def main() -> None:
    create_required_directories()
    arguments = parse_arguments()

    if arguments.process_every < 1:
        raise ValueError("--process-every must be at least 1.")

    input_video = arguments.input_video

    if not input_video.is_absolute():
        input_video = Path.cwd() / input_video

    run_timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = OUTPUT_DIR / f"itms_seatbelt_result_{run_timestamp}.mp4"
    event_log_path = LOG_DIR / f"seatbelt_events_{run_timestamp}.csv"
    metrics_path = METRICS_DIR / f"seatbelt_metrics_{run_timestamp}.json"

    detector = SeatbeltDetector(driver_side=arguments.driver_side)

    process_video(
        input_video=input_video,
        detector=detector,
        output_video_path=output_video_path,
        event_log_path=event_log_path,
        metrics_path=metrics_path,
        evidence_dir=EVIDENCE_DIR,
        process_every_n_frames=arguments.process_every,
        max_frames=arguments.max_frames,
        show_preview=arguments.show_preview,
        save_output_video=not arguments.no_output_video,
    )


if __name__ == "__main__":
    main()
