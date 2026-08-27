import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Real-time driver state detection with adaptive gaze reliability."
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Legacy camera index. Use --source instead.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index, video path, or stream path.",
    )

    parser.add_argument(
        "--video_name",
        type=str,
        default="",
        help="Video name written into prediction CSV.",
    )

    parser.add_argument(
        "--save_csv",
        type=str,
        default="predictions.csv",
        help="Path to save frame-level prediction CSV.",
    )

    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Run without showing OpenCV window.",
    )

    parser.add_argument(
        "--camera_params",
        type=str,
        default="",
        help="Path to camera calibration parameters file.",
    )

    parser.add_argument(
        "--show_fps",
        action="store_true",
        help="Show FPS on video.",
    )

    parser.add_argument(
        "--show_proc_time",
        action="store_true",
        help="Show per-frame processing time.",
    )

    parser.add_argument(
        "--show_eye_proc",
        action="store_true",
        help="Show eye processing landmarks.",
    )

    parser.add_argument(
        "--show_axis",
        action="store_true",
        help="Show head pose axis.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information.",
    )

    # Eye / sleep thresholds
    parser.add_argument(
        "--ear_thresh",
        type=float,
        default=0.22,
        help="EAR threshold. Eyes are closed when EAR <= this value.",
    )

    parser.add_argument(
        "--ear_time_thresh",
        type=float,
        default=3.0,
        help="Seconds eyes must stay closed before asleep/drowsiness alert.",
    )

    # Gaze thresholds
    parser.add_argument(
        "--gaze_thresh",
        type=float,
        default=0.2,
        help="Gaze score threshold for looking away.",
    )

    parser.add_argument(
        "--gaze_time_thresh",
        type=float,
        default=2.0,
        help="Seconds gaze must stay away before looking-away alert.",
    )

    # Head pose thresholds
    parser.add_argument(
        "--pose_time_thresh",
        type=float,
        default=2.0,
        help="Seconds head pose must be bad before distracted alert.",
    )

    parser.add_argument(
        "--roll_thresh",
        type=float,
        default=20.0,
        help="Head roll threshold.",
    )

    parser.add_argument(
        "--pitch_thresh",
        type=float,
        default=20.0,
        help="Head pitch threshold.",
    )

    parser.add_argument(
        "--yaw_thresh",
        type=float,
        default=25.0,
        help="Head yaw threshold.",
    )

    # Microsleep
    parser.add_argument(
        "--micro_closure_seconds",
        type=float,
        default=2.0,
        help="Minimum eye-closure duration for one microsleep closure event.",
    )

    parser.add_argument(
        "--micro_window_seconds",
        type=float,
        default=60.0,
        help="Window for microsleep pattern detection.",
    )

    parser.add_argument(
        "--micro_closure_count",
        type=int,
        default=3,
        help="Number of closure events needed for microsleep alert.",
    )

    # Yawn
    parser.add_argument(
        "--yawn_mar_thresh",
        type=float,
        default=0.55,
        help="MAR threshold for yawn detection.",
    )

    parser.add_argument(
        "--yawn_time_thresh",
        type=float,
        default=1.2,
        help="Seconds MAR must stay high before yawn alert.",
    )

    # Adaptive gaze reliability gate
    parser.add_argument(
        "--gaze_ear_min",
        type=float,
        default=0.18,
        help="Minimum EAR required to trust gaze. If eyes are too closed, ignore gaze.",
    )

    parser.add_argument(
        "--max_gaze_score",
        type=float,
        default=1.20,
        help="Maximum valid gaze score before considering gaze unreliable.",
    )

    parser.add_argument(
        "--max_gaze_jump",
        type=float,
        default=0.45,
        help="Maximum allowed frame-to-frame gaze jump.",
    )

    parser.add_argument(
        "--gaze_jitter_window",
        type=int,
        default=8,
        help="Number of frames used to estimate gaze jitter.",
    )

    parser.add_argument(
        "--max_gaze_jitter",
        type=float,
        default=0.35,
        help="Maximum gaze jitter before falling back to head pose only.",
    )

    parser.add_argument(
        "--gaze_bad_frames",
        type=int,
        default=3,
        help="Bad consecutive gaze frames needed to disable gaze.",
    )

    parser.add_argument(
        "--gaze_good_frames",
        type=int,
        default=5,
        help="Good consecutive gaze frames needed to re-enable gaze.",
    )

    args = parser.parse_args()
    return args