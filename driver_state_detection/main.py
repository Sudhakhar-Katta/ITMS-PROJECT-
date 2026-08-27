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
    t_now = time.perf_counter()

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
                print("Can't receive frame from camera/stream end")
                break

            frame_count += 1

            if source == 0:
                frame = cv2.flip(frame, 1)

            e1 = cv2.getTickCount()

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

                if ear is not None:
                    cv2.putText(
                        frame,
                        "EAR:" + str(round(float(ear), 3)),
                        (10, 50),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if mar is not None:
                    cv2.putText(
                        frame,
                        "MAR:" + str(round(float(mar), 3)),
                        (10, 80),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if gaze is not None:
                    cv2.putText(
                        frame,
                        "Gaze Score:" + str(round(float(gaze), 3)),
                        (10, 110),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame,
                    "PERCLOS:" + str(round(float(perclos_score), 3)),
                    (10, 140),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Mode: " + attention_mode,
                    (10, 170),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Gaze reliable: " + str(gaze_reliable) + " | " + gaze_reason,
                    (10, 195),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Away duration: " + str(distraction_states["away_duration"]),
                    (10, 220),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Distraction reason: "
                    + distraction_states["distraction_reason"],
                    (10, 245),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Tired reason: " + str(TiredFilter.last_reason),
                    (20, 420),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Sleep reason: " + sleep_states["sleep_reason"],
                    (20, 445),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Sleep pose duration: " + str(sleep_states["sleep_pose_duration"]),
                    (20, 470),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

                if distraction_states.get("away_signal", False):
                    cv2.putText(
                        frame,
                        "AWAY SIGNAL",
                        (20, 455),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.5,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if roll is not None:
                    cv2.putText(
                        frame,
                        "roll:" + str(roll.round(1)[0]),
                        (450, 40),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.5,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if pitch is not None:
                    cv2.putText(
                        frame,
                        "pitch:" + str(pitch.round(1)[0]),
                        (450, 70),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.5,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if yaw is not None:
                    cv2.putText(
                        frame,
                        "yaw:" + str(yaw.round(1)[0]),
                        (450, 100),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.5,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if tired:
                    cv2.putText(
                        frame,
                        "TIRED!",
                        (10, 260),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if asleep:
                    cv2.putText(
                        frame,
                        "ASLEEP!",
                        (10, 280),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if looking_away:
                    cv2.putText(
                        frame,
                        "LOOKING AWAY!",
                        (10, 300),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if distracted:
                    cv2.putText(
                        frame,
                        "DISTRACTED!",
                        (10, 320),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if states["drowsiness_active"]:
                    cv2.putText(
                        frame,
                        "DROWSINESS: EYES CLOSED >= 3s",
                        (10, 340),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if states["yawning"]:
                    cv2.putText(
                        frame,
                        "YAWNING!",
                        (10, 360),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                for alert in alerts:
                    print("ALERT:", alert)
                    events_writer.writerow(build_event_row(video_name, alert))
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
                    "FACE NOT DETECTED",
                    (10, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
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

            if not args.no_display:
                cv2.imshow("Press 'q' to terminate", frame)

                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break

    finally:
        csv_file.close()
        events_csv_file.close()
        cap.release()
        cv2.destroyAllWindows()
        print("\nRun summary")
        print(f"total frames processed: {frame_count}")
        print(f"output frame CSV path: {args.save_csv}")
        print(f"output events CSV path: {events_csv_path}")
        print("number of events by type:")
        if event_counts:
            for event_name, count in sorted(event_counts.items()):
                print(f"  {event_name}: {count}")
        else:
            print("  none: 0")


if __name__ == "__main__":
    main()
