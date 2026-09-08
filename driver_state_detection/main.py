import os
import time
import pprint
import csv
import json
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
from tired_filter import TiredDisplayFilter
from attention_scorer import AttentionScorer as AttScorer
from distraction_monitor import DistractionMonitor
from eye_detector import EyeDetector as EyeDet
from parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters
from state_monitor import DriverStateMonitor
from mouth_detector import get_MAR
from gaze_reliability import GazeReliabilityGate
from asleep_monitor import AsleepMonitor


def safe_float(value, decimals=4):
    if value is None:
        return ""

    try:
        value = np.ravel(value)[0]
        return round(float(value), decimals)
    except Exception:
        return ""


def build_event_row(video_name, alert):
    event_name = alert.get("event", "")
    duration_sec = alert.get("duration_sec", "")
    if duration_sec == "":
        duration_sec = alert.get("closed_duration", "")
    if duration_sec == "":
        duration_sec = alert.get("away_duration", "")
    if duration_sec == "":
        duration_sec = alert.get("yawn_duration", "")

    known_fields = {
        "event",
        "time_sec",
        "duration_sec",
        "closed_duration",
        "away_duration",
        "yawn_duration",
        "reason",
        "mode",
        "detection_delay",
    }
    extra = {key: value for key, value in alert.items() if key not in known_fields}
    default_reasons = {
        "driver_drowsiness": "eyes_closed",
        "micro_sleep": "repeated_eye_closures",
        "yawning": "single_yawn",
        "yawning_pattern": "three_yawns",
        "driver_distraction": "looking_away",
    }
    default_modes = {
        "driver_drowsiness": "ear",
        "micro_sleep": "ear",
        "yawning": "mar",
        "yawning_pattern": "mar",
        "driver_distraction": "adaptive_gaze_head_pose",
    }

    return {
        "video": video_name,
        "event": event_name,
        "time_sec": alert.get("time_sec", ""),
        "duration_sec": duration_sec,
        "reason": alert.get("reason", default_reasons.get(event_name, "")),
        "mode": alert.get("mode", default_modes.get(event_name, "")),
        "detection_delay": alert.get("detection_delay", ""),
        "extra": json.dumps(extra, sort_keys=True),
    }


def get_video_codec(output_path):
    extension = os.path.splitext(output_path)[1].lower()
    if extension in {".avi"}:
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


def open_video_writer(output_path, frame_size, fps):
    output_parent = os.path.dirname(output_path)

    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    writer_fps = fps if fps and fps > 0 else 30.0
    writer = cv2.VideoWriter(
        output_path,
        get_video_codec(output_path),
        writer_fps,
        frame_size,
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video for writing: {output_path}")

    return writer


def format_video_timestamp(seconds):
    if seconds is None or seconds < 0:
        seconds = 0.0

    minutes = int(seconds // 60)
    remaining_seconds = seconds - (minutes * 60)
    return f"{minutes:02d}:{remaining_seconds:05.2f}"


def get_landmark_bounds(landmarks, frame_size):
    width, height = frame_size
    xs = landmarks[:, 0] * width
    ys = landmarks[:, 1] * height
    return (
        int(np.min(xs)),
        int(np.min(ys)),
        int(np.max(xs)),
        int(np.max(ys)),
    )


def rect_overlap_area(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])

    if right <= left or bottom <= top:
        return 0

    return (right - left) * (bottom - top)


def choose_panel_position(frame, panel_width, panel_height, avoid_rect=None):
    frame_height, frame_width = frame.shape[:2]
    margin = 10
    candidates = [
        (margin, margin),
        (frame_width - panel_width - margin, margin),
        (margin, frame_height - panel_height - margin),
        (frame_width - panel_width - margin, frame_height - panel_height - margin),
    ]

    candidates = [
        (
            max(margin, min(x, frame_width - panel_width - margin)),
            max(margin, min(y, frame_height - panel_height - margin)),
        )
        for x, y in candidates
    ]

    if avoid_rect is None:
        return candidates[0]

    face_center_x = (avoid_rect[0] + avoid_rect[2]) / 2
    face_center_y = (avoid_rect[1] + avoid_rect[3]) / 2

    def placement_score(point):
        panel_rect = (
            point[0],
            point[1],
            point[0] + panel_width,
            point[1] + panel_height,
        )
        panel_center_x = (panel_rect[0] + panel_rect[2]) / 2
        panel_center_y = (panel_rect[1] + panel_rect[3]) / 2
        distance_from_face = (
            abs(panel_center_x - face_center_x) + abs(panel_center_y - face_center_y)
        )
        return rect_overlap_area(panel_rect, avoid_rect), -distance_from_face

    return min(candidates, key=placement_score)


def draw_live_state_overlay(frame, status_items, avoid_rect=None):
    time_value = None
    visible_items = []
    for item in status_items:
        if item["label"] == "Time":
            time_value = item.get("value")
        else:
            visible_items.append(item)
    status_items = visible_items

    event_alerts = [
        item["label"]
        for item in status_items
        if item.get("active") is True and item.get("severity") == "event"
    ]
    active_alerts = [
        item["label"]
        for item in status_items
        if item.get("active") is True and item["label"] != "Face"
    ]
    banner_height = 28 if active_alerts else 0
    row_height = 25
    content_rows = (len(status_items) + 1) // 2
    panel_width = min(440, frame.shape[1] - 20)
    panel_height = 44 + banner_height + (content_rows * row_height) + 10
    panel_x, panel_y = choose_panel_position(
        frame,
        panel_width,
        panel_height,
        avoid_rect=avoid_rect,
    )
    panel_right = panel_x + panel_width
    panel_bottom = panel_y + panel_height

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_right, panel_bottom),
        (12, 18, 28),
        -1,
    )
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_right, panel_bottom),
        (80, 95, 115),
        1,
    )
    cv2.line(
        frame,
        (panel_x, panel_y + 34),
        (panel_right, panel_y + 34),
        (80, 95, 115),
        1,
    )

    cv2.putText(
        frame,
        "DRIVER STATE",
        (panel_x + 10, panel_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if time_value is not None:
        cv2.putText(
            frame,
            fit_panel_text(f"Time: {time_value}", panel_width - 180),
            (panel_x + 170, panel_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (230, 235, 242),
            1,
            cv2.LINE_AA,
        )

    if active_alerts:
        banner_top = panel_y + 34
        banner_bottom = banner_top + banner_height
        banner_label = "EVENT: " if event_alerts else "ACTIVE: "
        banner_items = event_alerts if event_alerts else active_alerts
        banner_color = (0, 120, 255) if event_alerts else (0, 0, 180)
        cv2.rectangle(
            frame,
            (panel_x + 1, banner_top + 1),
            (panel_right - 1, banner_bottom),
            banner_color,
            -1,
        )
        cv2.putText(
            frame,
            banner_label + ", ".join(banner_items[:3]).upper(),
            (panel_x + 10, banner_top + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        cv2.LINE_AA,
    )

    column_width = (panel_width - 24) // 2
    label_width = 108

    for index, item in enumerate(status_items):
        label = item["label"]
        active = item.get("active")
        value = item.get("value")
        column = index % 2
        row = index // 2
        column_x = panel_x + 12 + (column * column_width)
        y = panel_y + 56 + banner_height + (row * row_height)

        if active is None:
            color = (225, 225, 225)
            status_text = fit_panel_text(str(value), column_width - label_width - 28)
        else:
            color = (20, 40, 255) if active else (80, 220, 130)
            status_text = "ACTIVE" if active else "OFF"
            if value not in (None, ""):
                status_text = f"{status_text}  {value}"
            status_text = fit_panel_text(status_text, column_width - label_width - 28)
            if active:
                is_event = item.get("severity") == "event"
                row_color = (0, 120, 255) if is_event else (18, 28, 80)
                thickness = -1 if is_event else 1
                cv2.rectangle(
                    frame,
                    (column_x - 4, y - 18),
                    (column_x + column_width - 10, y + 4),
                    row_color,
                    thickness,
                )
                if is_event:
                    color = (255, 255, 255)

        cv2.circle(frame, (column_x + 6, y - 6), 5, color, -1)
        cv2.putText(
            frame,
            label.upper() if active else label,
            (column_x + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (230, 235, 242),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            status_text,
            (column_x + label_width, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def fit_panel_text(text, max_width):
    if max_width <= 0:
        return ""

    text = str(text)
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)[0][0] <= max_width:
        return text

    suffix = "..."
    while text:
        candidate = text + suffix
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.47, 1)[0][0]
        if width <= max_width:
            return candidate
        text = text[:-1]

    return suffix


def format_active_states(state_flags):
    active_states = [label for label, active in state_flags if active]
    return ", ".join(active_states) if active_states else "none"


def format_progress(frame_count, total_frames):
    if total_frames and total_frames > 0:
        percent = (frame_count / total_frames) * 100
        return f"Processing: {frame_count} / {total_frames} frames ({percent:.1f}%)"

    return f"Processing: {frame_count} frames"


def main():
    args = get_args()

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except Exception as e:
            print(
                f"OpenCV optimization could not be set to True, the script may be slower than expected.\nError: {e}"
            )

    if args.camera_params:
        camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
    else:
        camera_matrix, dist_coeffs = None, None

    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)
        print("\nCamera Matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("\nDistortion Coefficients:")
        pprint.pp(dist_coeffs, indent=4)
        print("\n")

    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    Head_pose = HeadPoseEst(
        show_axis=args.show_axis,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    prev_time = time.perf_counter()
    fps = 0.0
    t_now = 0.0 if args.video_time else time.perf_counter()

    Scorer = AttScorer(
        t_now=t_now,
        ear_thresh=args.ear_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_thresh=args.gaze_thresh,
        pose_time_thresh=args.pose_time_thresh,
        verbose=args.verbose,
    )

    StateMonitor = DriverStateMonitor(
        ear_thresh=args.ear_thresh,
        drowsy_seconds=args.ear_time_thresh,
        micro_closure_seconds=args.micro_closure_seconds,
        micro_closure_count=args.micro_closure_count,
        micro_window_seconds=args.micro_window_seconds,
        yawn_mar_thresh=args.yawn_mar_thresh,
        yawn_seconds=args.yawn_time_thresh,
        yawn_pattern_count=3,
        yawn_window_seconds=300.0,
    )
    TiredFilter = TiredDisplayFilter(
        tired_on_perclos=0.14,
        tired_off_perclos=0.07,
        critical_perclos=0.30,
        perclos_activation_seconds=1.0,
        slow_closure_seconds=0.55,
        blink_ignore_seconds=0.35,
        heavy_ear_thresh=args.ear_thresh + 0.10,
        heavy_ear_seconds=1.0,
        clear_after_open_seconds=1.0,
        min_active_seconds=0.8,
    )

    GazeGate = GazeReliabilityGate(
        ear_min_for_gaze=args.gaze_ear_min,
        max_gaze_score=args.max_gaze_score,
        max_gaze_jump=args.max_gaze_jump,
        jitter_window=args.gaze_jitter_window,
        max_jitter=args.max_gaze_jitter,
        bad_frames_to_disable=1,
        good_frames_to_enable=8,
        max_dark_ratio=0.65,
        max_glare_ratio=0.35,
        min_eye_contrast=4.0,
        min_eye_to_face_brightness_ratio=0.42,
    )

    Distraction = DistractionMonitor(
        gaze_thresh=args.gaze_thresh,
        lookaway_yaw_thresh=22.0,
        lookaway_pitch_thresh=18.0,
        severe_yaw_thresh=40.0,
        severe_pitch_thresh=28.0,
        severe_roll_thresh=28.0,
        distraction_seconds=5.0,
        max_signal_gap_seconds=0.35,
        smoothing_alpha=0.30,
        calibration_seconds=2.0,
    )
    SleepMonitor = AsleepMonitor(head_down_seconds=3.0)

    source = int(args.source) if str(args.source).isdigit() else args.source

    video_name = getattr(args, "video_name", "")

    if not video_name:
        if isinstance(source, str):
            video_name = os.path.basename(source)
        else:
            video_name = "webcam"

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Cannot open camera/video source: {source}")
        return

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_duration = (
        source_frame_total / source_fps
        if source_frame_total > 0 and source_fps > 0
        else None
    )
    output_video_path = args.output_video
    output_video_writer = None

    if output_video_path and source_width > 0 and source_height > 0:
        output_video_writer = open_video_writer(
            output_video_path,
            (source_width, source_height),
            source_fps,
        )

    save_csv_parent = os.path.dirname(args.save_csv)

    if save_csv_parent:
        os.makedirs(save_csv_parent, exist_ok=True)

    events_csv_path = os.path.join(save_csv_parent or ".", "events.csv")
    csv_file = open(args.save_csv, mode="w", newline="", encoding="utf-8")
    events_csv_file = open(events_csv_path, mode="w", newline="", encoding="utf-8")

    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "video",
            "face_detected",
            "time_sec",
            "ear",
            "mar",
            "perclos",
            "gaze",
            "gaze_used",
            "gaze_reliable",
            "gaze_reason",
            "attention_mode",
            "roll",
            "pitch",
            "yaw",
            "rel_roll",
            "rel_pitch",
            "rel_yaw",
            "eyes_closed",
            "closed_duration",
            "tired_raw",
            "tired",
            "tired_reason",
            "asleep",
            "sleep_reason",
            "sleep_pose_signal",
            "sleep_pose_duration",
            "away_signal",
            "looking_away",
            "distracted",
            "away_duration",
            "distraction_reason",
            "distraction_mode",
            "drowsiness_active",
            "yawning",
            "yawn_alert",
            "yawn_count_in_window",
            "micro_closure_count",
            "events",
        ],
    )
    csv_writer.writeheader()

    events_writer = csv.DictWriter(
        events_csv_file,
        fieldnames=[
            "video",
            "event",
            "time_sec",
            "duration_sec",
            "reason",
            "mode",
            "detection_delay",
            "extra",
        ],
    )
    events_writer.writeheader()

    frame_count = 0
    output_frame_count = 0
    frame_errors = 0
    event_counts = Counter()
    face_missing_start = None

    try:
        while True:
            t_now = time.perf_counter()

            elapsed_time = t_now - prev_time
            prev_time = t_now

            if elapsed_time > 0:
                fps = np.round(1 / elapsed_time, 3)

            ret, frame = cap.read()

            if not ret:
                print("Reached end of source video")
                break

            frame_count += 1
            if output_video_path and output_video_writer is None:
                frame_size = frame.shape[1], frame.shape[0]
                output_video_writer = open_video_writer(
                    output_video_path,
                    frame_size,
                    source_fps,
                )

            if args.video_time and not isinstance(source, int) and source_fps > 0:
                t_now = (frame_count - 1) / source_fps

            if frame_count % 30 == 0:
                print(format_progress(frame_count, source_frame_total))

            timestamp_label = (
                f"{format_video_timestamp(t_now)} / "
                f"{format_video_timestamp(source_duration)}"
                if source_duration is not None
                else f"{safe_float(t_now, 3)}s"
            )

            if source == 0:
                frame = cv2.flip(frame, 1)

            e1 = cv2.getTickCount()

            try:
                gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_size = frame.shape[1], frame.shape[0]
    
                gray = np.expand_dims(gray_raw, axis=2)
                gray = np.concatenate([gray, gray, gray], axis=2)
    
                lms = Detector.process(gray).multi_face_landmarks
    
                if lms:
                    if face_missing_start is not None:
                        if (t_now - face_missing_start) >= 0.5:
                            StateMonitor.reset_active_continuity()
                            Distraction.reset_active_continuity()
                            SleepMonitor.reset_active_continuity()
                            Scorer.reset_active_continuity(t_now)
                        face_missing_start = None
    
                    landmarks = get_landmarks(lms)
                    face_bounds = get_landmark_bounds(landmarks, frame_size)
    
                    Eye_det.show_eye_keypoints(
                        color_frame=frame,
                        landmarks=landmarks,
                        frame_size=frame_size,
                    )
    
                    ear = Eye_det.get_EAR(landmarks=landmarks)
                    mar = get_MAR(landmarks)
    
                    tired_raw, perclos_score = Scorer.get_rolling_PERCLOS(t_now, ear)
    
                    gaze = Eye_det.get_Gaze_Score(
                        frame=gray,
                        landmarks=landmarks,
                        frame_size=frame_size,
                    )
    
                    frame_det, roll, pitch, yaw = Head_pose.get_pose(
                        frame=frame,
                        landmarks=landmarks,
                        frame_size=frame_size,
                    )
    
                    gaze_reliable, gaze_reason, attention_mode = GazeGate.update(
                        ear=ear,
                        gaze_score=gaze,
                        gray_frame=gray_raw,
                        landmarks=landmarks,
                        frame_size=frame_size,
                    )
    
                    gaze_for_scorer = gaze if gaze_reliable else 0.0
    
                    asleep, old_looking_away, old_distracted = Scorer.eval_scores(
                        t_now=t_now,
                        ear_score=ear,
                        gaze_score=gaze_for_scorer,
                        head_roll=roll,
                        head_pitch=pitch,
                        head_yaw=yaw,
                    )
    
                    states, alerts = StateMonitor.update(
                        t_now=t_now,
                        ear=ear,
                        mar=mar,
                    )
                    micro_sleep_active = any(
                        alert.get("event") == "micro_sleep" for alert in alerts
                    )
    
                    tired = TiredFilter.update(
                        t_now=t_now,
                        perclos_score=perclos_score,
                        eyes_closed=states["eyes_closed"],
                        closed_duration=states["closed_duration"],
                        ear=ear,
                    )
    
                    distraction_states, distraction_alerts = Distraction.update(
                        t_now=t_now,
                        roll=roll,
                        pitch=pitch,
                        yaw=yaw,
                        gaze_score=gaze,
                        gaze_reliable=gaze_reliable,
                    )
    
                    sleep_states, sleep_alerts = SleepMonitor.update(
                        t_now=t_now,
                        drowsiness_active=states["drowsiness_active"],
                        tired=tired,
                        rel_pitch=distraction_states["rel_pitch"],
                        rel_roll=distraction_states["rel_roll"],
                        rel_yaw=distraction_states["rel_yaw"],
                    )
                    asleep = asleep or sleep_states["asleep"]
                    alerts.extend(sleep_alerts)
    
                    looking_away = distraction_states["looking_away"]
                    distracted = distraction_states["distracted"]
    
                    alerts.extend(distraction_alerts)
    
                    if frame_det is not None:
                        frame = frame_det
    
                    draw_live_state_overlay(
                        frame,
                        [
                            {"label": "Face", "active": True},
                            {
                                "label": "Time",
                                "active": None,
                                "value": timestamp_label,
                            },
                            {
                                "label": "Active States",
                                "active": None,
                                "value": format_active_states(
                                    [
                                        ("EYES CLOSED", states["eyes_closed"]),
                                        ("TIRED", tired),
                                        ("ASLEEP", asleep),
                                        ("LOOKING AWAY", looking_away),
                                        ("DISTRACTED", distracted),
                                        (
                                            "DROWSINESS",
                                            states["drowsiness_active"],
                                        ),
                                        ("YAWNING", states["yawning"]),
                                        (
                                            "YAWN PATTERN",
                                            states.get("yawn_alert", False),
                                        ),
                                        ("MICRO SLEEP", micro_sleep_active),
                                    ]
                                ),
                            },
                            {
                                "label": "EYES CLOSED",
                                "active": states["eyes_closed"],
                            },
                            {"label": "TIRED", "active": tired},
                            {"label": "YAWNING", "active": states["yawning"]},
                            {
                                "label": "DROWSINESS",
                                "active": states["drowsiness_active"],
                            },
                            {"label": "ASLEEP", "active": asleep},
                            {"label": "LOOKING AWAY", "active": looking_away},
                            {"label": "DISTRACTED", "active": distracted},
                            {
                                "label": "YAWN PATTERN",
                                "active": states.get("yawn_alert", False),
                                "severity": "event",
                            },
                            {
                                "label": "MICRO SLEEP",
                                "active": micro_sleep_active,
                            },
                            {
                                "label": "Mode",
                                "active": None,
                                "value": attention_mode,
                            },
                            {
                                "label": "EAR",
                                "active": None,
                                "value": safe_float(ear, 3),
                            },
                            {
                                "label": "Eyes closed",
                                "active": None,
                                "value": f"{safe_float(states['closed_duration'], 3)}s",
                            },
                            {
                                "label": "MAR",
                                "active": None,
                                "value": safe_float(mar, 3),
                            },
                            {
                                "label": "PERCLOS",
                                "active": None,
                                "value": safe_float(perclos_score, 3),
                            },
                            {
                                "label": "Tired Reason",
                                "active": None,
                                "value": TiredFilter.last_reason,
                            },
                            {
                                "label": "Sleep Reason",
                                "active": None,
                                "value": sleep_states["sleep_reason"],
                            },
                            {
                                "label": "Gaze Score",
                                "active": None,
                                "value": safe_float(gaze, 3),
                            },
                            {
                                "label": "Gaze Reliable",
                                "active": None,
                                "value": f"{gaze_reliable} {gaze_reason}",
                            },
                            {
                                "label": "Roll",
                                "active": None,
                                "value": safe_float(roll, 1),
                            },
                            {
                                "label": "Pitch",
                                "active": None,
                                "value": safe_float(pitch, 1),
                            },
                            {
                                "label": "Yaw",
                                "active": None,
                                "value": safe_float(yaw, 1),
                            },
                            {
                                "label": "Rel Roll",
                                "active": None,
                                "value": distraction_states["rel_roll"],
                            },
                            {
                                "label": "Rel Pitch",
                                "active": None,
                                "value": distraction_states["rel_pitch"],
                            },
                            {
                                "label": "Rel Yaw",
                                "active": None,
                                "value": distraction_states["rel_yaw"],
                            },
                            {
                                "label": "Away seconds",
                                "active": None,
                                "value": safe_float(
                                    distraction_states["away_duration"], 2
                                ),
                            },
                            {
                                "label": "Away Reason",
                                "active": None,
                                "value": distraction_states["distraction_reason"],
                            },
                        ],
                        avoid_rect=face_bounds,
                    )
    
                    for alert in alerts:
                        print("ALERT:", alert)
                        events_writer.writerow(build_event_row(video_name, alert))
                        events_csv_file.flush()
                        event_counts[alert.get("event", "unknown")] += 1
    
                    csv_writer.writerow(
                        {
                            "video": video_name,
                            "face_detected": True,
                            "time_sec": round(t_now, 3),
                            "ear": safe_float(ear, 4),
                            "mar": safe_float(mar, 4),
                            "perclos": safe_float(perclos_score, 4),
                            "gaze": safe_float(gaze, 4),
                            "gaze_used": safe_float(gaze_for_scorer, 4),
                            "gaze_reliable": gaze_reliable,
                            "gaze_reason": gaze_reason,
                            "attention_mode": attention_mode,
                            "roll": safe_float(roll, 3),
                            "pitch": safe_float(pitch, 3),
                            "yaw": safe_float(yaw, 3),
                            "rel_roll": distraction_states["rel_roll"],
                            "rel_pitch": distraction_states["rel_pitch"],
                            "rel_yaw": distraction_states["rel_yaw"],
                            "eyes_closed": states["eyes_closed"],
                            "closed_duration": states["closed_duration"],
                            "tired_raw": tired_raw,
                            "tired": tired,
                            "tired_reason": TiredFilter.last_reason,
                            "asleep": asleep,
                            "sleep_reason": sleep_states["sleep_reason"],
                            "sleep_pose_signal": sleep_states["sleep_pose_signal"],
                            "sleep_pose_duration": sleep_states["sleep_pose_duration"],
                            "away_signal": distraction_states.get("away_signal", False),
                            "looking_away": looking_away,
                            "distracted": distracted,
                            "away_duration": distraction_states["away_duration"],
                            "distraction_reason": distraction_states[
                                "distraction_reason"
                            ],
                            "distraction_mode": distraction_states["distraction_mode"],
                            "drowsiness_active": states["drowsiness_active"],
                            "yawning": states["yawning"],
                            "yawn_alert": states.get("yawn_alert", False),
                            "yawn_count_in_window": states.get(
                                "yawn_count_in_window", states.get("yawn_count", 0)
                            ),
                            "micro_closure_count": states["micro_closure_count"],
                            "events": "|".join([a["event"] for a in alerts]),
                        }
                    )
                    if frame_count % 10 == 0:
                        csv_file.flush()
    
                else:
                    if face_missing_start is None:
                        face_missing_start = t_now
    
                    if (t_now - face_missing_start) >= 0.5:
                        StateMonitor.reset_active_continuity()
                        Distraction.reset_active_continuity()
                        SleepMonitor.reset_active_continuity()
                        Scorer.reset_active_continuity(t_now)
    
                    cv2.putText(
                        frame,
                        "NO FACE DETECTED",
                        (10, 50),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.5,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
    
                    draw_live_state_overlay(
                        frame,
                        [
                            {"label": "Face", "active": False},
                            {
                                "label": "Time",
                                "active": None,
                                "value": timestamp_label,
                            },
                            {
                                "label": "Active States",
                                "active": None,
                                "value": "no_face",
                            },
                            {"label": "Mode", "active": None, "value": "no_face"},
                            {"label": "EAR", "active": None, "value": "n/a"},
                            {"label": "Closed For", "active": None, "value": "0.0s"},
                            {"label": "MAR", "active": None, "value": "n/a"},
                            {"label": "PERCLOS", "active": None, "value": "n/a"},
                            {"label": "Tired Reason", "active": None, "value": "n/a"},
                            {"label": "Gaze Score", "active": None, "value": "n/a"},
                            {"label": "Gaze Reliable", "active": None, "value": "n/a"},
                            {"label": "Roll", "active": None, "value": "n/a"},
                            {"label": "Pitch", "active": None, "value": "n/a"},
                            {"label": "Yaw", "active": None, "value": "n/a"},
                            {"label": "Rel Roll", "active": None, "value": "n/a"},
                            {"label": "Rel Pitch", "active": None, "value": "n/a"},
                            {"label": "Rel Yaw", "active": None, "value": "n/a"},
                            {"label": "EYES CLOSED", "active": False},
                            {"label": "TIRED", "active": False},
                            {"label": "YAWNING", "active": False},
                            {"label": "DROWSINESS", "active": False},
                            {"label": "ASLEEP", "active": False},
                            {"label": "LOOKING AWAY", "active": False},
                            {"label": "DISTRACTED", "active": False},
                            {
                                "label": "YAWN PATTERN",
                                "active": False,
                                "severity": "event",
                            },
                            {"label": "MICRO SLEEP", "active": False},
                        ],
                    )
    
                    csv_writer.writerow(
                        {
                            "video": video_name,
                            "face_detected": False,
                            "time_sec": round(t_now, 3),
                            "sleep_reason": "not_asleep",
                            "sleep_pose_signal": False,
                            "sleep_pose_duration": 0.0,
                            "events": "",
                        }
                    )
                    if frame_count % 10 == 0:
                        csv_file.flush()

            except Exception as exc:
                frame_errors += 1
                print(f"Frame {frame_count} analysis error: {exc}")
                StateMonitor.reset_active_continuity()
                Distraction.reset_active_continuity()
                SleepMonitor.reset_active_continuity()
                Scorer.reset_active_continuity(t_now)
                cv2.putText(
                    frame,
                    "FRAME ANALYSIS ERROR",
                    (10, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                draw_live_state_overlay(
                    frame,
                    [
                        {"label": "Face", "active": False},
                        {
                            "label": "Time",
                            "active": None,
                            "value": timestamp_label,
                        },
                        {
                            "label": "Active States",
                            "active": None,
                            "value": "frame_error",
                        },
                        {"label": "Mode", "active": None, "value": "frame_error"},
                        {"label": "EAR", "active": None, "value": "n/a"},
                        {"label": "Closed For", "active": None, "value": "n/a"},
                        {"label": "MAR", "active": None, "value": "n/a"},
                        {"label": "PERCLOS", "active": None, "value": "n/a"},
                        {"label": "Tired Reason", "active": None, "value": "n/a"},
                        {"label": "Gaze Score", "active": None, "value": "n/a"},
                        {"label": "Gaze Reliable", "active": None, "value": "n/a"},
                        {"label": "Roll", "active": None, "value": "n/a"},
                        {"label": "Pitch", "active": None, "value": "n/a"},
                        {"label": "Yaw", "active": None, "value": "n/a"},
                        {"label": "Rel Roll", "active": None, "value": "n/a"},
                        {"label": "Rel Pitch", "active": None, "value": "n/a"},
                        {"label": "Rel Yaw", "active": None, "value": "n/a"},
                        {"label": "EYES CLOSED", "active": False},
                        {"label": "TIRED", "active": False},
                        {"label": "YAWNING", "active": False},
                        {"label": "DROWSINESS", "active": False},
                        {"label": "ASLEEP", "active": False},
                        {"label": "LOOKING AWAY", "active": False},
                        {"label": "DISTRACTED", "active": False},
                        {
                            "label": "YAWN PATTERN",
                            "active": False,
                            "severity": "event",
                        },
                        {"label": "MICRO SLEEP", "active": False},
                    ],
                )
                csv_writer.writerow(
                    {
                        "video": video_name,
                        "face_detected": False,
                        "time_sec": round(t_now, 3),
                        "sleep_reason": "frame_analysis_error",
                        "sleep_pose_signal": False,
                        "sleep_pose_duration": 0.0,
                        "events": "",
                    }
                )
                if frame_count % 10 == 0:
                    csv_file.flush()
            e2 = cv2.getTickCount()
            proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000

            if args.show_fps:
                cv2.putText(
                    frame,
                    "FPS:" + str(round(fps)),
                    (10, 400),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 0, 255),
                    1,
                )

            if args.show_proc_time:
                cv2.putText(
                    frame,
                    "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + "ms",
                    (10, 430),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 0, 255),
                    1,
                )

            if output_video_writer is not None:
                output_video_writer.write(frame)
                output_frame_count += 1

            if not args.no_display:
                cv2.imshow("Press 'q' to terminate", frame)

                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break

    finally:
        csv_file.close()
        events_csv_file.close()
        if output_video_writer is not None:
            output_video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("\nAnalysis complete")
        print(f"Input frames: {frame_count}")
        print(f"Output frames: {output_frame_count}")
        print(f"Input FPS: {source_fps if source_fps > 0 else 'unknown'}")
        print(f"Output video: {output_video_path or 'not requested'}")
        print(f"Predictions: {args.save_csv}")
        print(f"Events: {events_csv_path}")
        if output_video_path:
            frame_delta = abs(output_frame_count - frame_count)
            if frame_delta <= 1:
                print("Frame count check: output approximately equals input")
            else:
                print(
                    "Frame count check: WARNING output video frames do not match "
                    f"input frames ({output_frame_count} != {frame_count})"
                )
        print(f"frame analysis errors recovered: {frame_errors}")
        print("number of events by type:")
        if event_counts:
            for event_name, count in sorted(event_counts.items()):
                print(f"  {event_name}: {count}")
        else:
            print("  none: 0")


if __name__ == "__main__":
    main()
