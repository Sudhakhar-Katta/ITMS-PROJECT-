from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
DRIVER_STATE_DIR = PROJECT_DIR / "driver_state_detection"
SEATBELT_DIR = PROJECT_DIR / "seatbelt_detection"
CAMERA_PARAMS_PATH = DRIVER_STATE_DIR / "camera_params.json"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "videos"

DEFAULT_INPUT_VIDEO = PROJECT_DIR / "demo" / "driver_demo_1min.mp4"
DEFAULT_DRIVER_SECONDS = 30.0
DEFAULT_PROCESS_EVERY = 1
DEFAULT_DRIVER_SIDE = "right"


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


def open_capture(video_path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def read_video_info(video_path: Path) -> VideoInfo:
    capture = open_capture(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    if fps <= 0:
        fps = 30.0
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Video has invalid dimensions: {video_path}")

    duration_seconds = frame_count / fps if frame_count > 0 else 0.0
    return VideoInfo(
        path=video_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
    )


def create_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    fourcc_text: str = "mp4v",
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc_text),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")
    return writer


def split_video_by_time(
    input_video: Path,
    first_output: Path,
    second_output: Path,
    split_seconds: float,
) -> tuple[VideoInfo, int, int]:
    info = read_video_info(input_video)
    split_frame = max(1, min(info.frame_count, int(round(split_seconds * info.fps))))

    capture = open_capture(input_video)
    first_writer = create_writer(first_output, info.fps, info.width, info.height)
    second_writer = create_writer(second_output, info.fps, info.width, info.height)

    first_count = 0
    second_count = 0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1

            if frame_index <= split_frame:
                first_writer.write(frame)
                first_count += 1
            else:
                second_writer.write(frame)
                second_count += 1
    finally:
        capture.release()
        first_writer.release()
        second_writer.release()

    if first_count == 0:
        raise RuntimeError("Driver-state segment was empty.")
    if second_count == 0:
        raise RuntimeError("Seatbelt segment was empty.")

    return info, first_count, second_count


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}. Log: {log_path}"
        )


def run_driver_state_stage(
    input_segment: Path,
    output_video: Path,
    predictions_csv: Path,
    log_path: Path,
    camera_params: Path | None,
) -> Path:
    command = [
        sys.executable,
        "main.py",
        "--source",
        str(input_segment),
        "--video_name",
        input_segment.name,
        "--save_csv",
        str(predictions_csv),
        "--output_video",
        str(output_video),
        "--no_display",
        "--video_time",
    ]
    if camera_params is not None and camera_params.exists():
        command.extend(["--camera_params", str(camera_params)])

    run_command(command, DRIVER_STATE_DIR, log_path)
    events_csv = predictions_csv.parent / "events.csv"
    if not output_video.exists():
        raise RuntimeError(f"Driver-state stage did not create: {output_video}")
    return events_csv


def import_seatbelt_module():
    sys.path.insert(0, str(SEATBELT_DIR))
    try:
        import main as seatbelt_main
    finally:
        try:
            sys.path.remove(str(SEATBELT_DIR))
        except ValueError:
            pass
    return seatbelt_main


def run_seatbelt_stage(
    input_segment: Path,
    output_video: Path,
    event_log: Path,
    metrics_path: Path,
    evidence_dir: Path,
    driver_side: str,
    process_every: int,
    max_frames: int,
) -> None:
    seatbelt_main = import_seatbelt_module()
    seatbelt_main.create_required_directories()
    detector = seatbelt_main.SeatbeltDetector(driver_side=driver_side)
    seatbelt_main.process_video(
        input_video=input_segment,
        detector=detector,
        output_video_path=output_video,
        event_log_path=event_log,
        metrics_path=metrics_path,
        evidence_dir=evidence_dir,
        process_every_n_frames=process_every,
        max_frames=max_frames,
        show_preview=False,
        save_output_video=True,
    )
    if not output_video.exists():
        raise RuntimeError(f"Seatbelt stage did not create: {output_video}")


def iter_video_frames(video_path: Path) -> Iterable:
    capture = open_capture(video_path)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def concatenate_videos(video_paths: list[Path], output_path: Path, fps: float) -> int:
    if not video_paths:
        raise ValueError("No videos were provided for concatenation.")

    first_info = read_video_info(video_paths[0])
    writer = create_writer(output_path, fps, first_info.width, first_info.height)
    written = 0

    try:
        for video_path in video_paths:
            for frame in iter_video_frames(video_path):
                if frame.shape[1] != first_info.width or frame.shape[0] != first_info.height:
                    frame = cv2.resize(frame, (first_info.width, first_info.height))
                writer.write(frame)
                written += 1
    finally:
        writer.release()

    if written == 0:
        raise RuntimeError(f"Concatenation wrote no frames: {output_path}")
    return written


def transcode_with_ffmpeg(input_video: Path, output_video: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_video),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_video),
    ]
    subprocess.run(command, check=True)
    return True


def transcode_with_opencv_h264(input_video: Path, output_video: Path, fps: float) -> str:
    info = read_video_info(input_video)
    failures: list[str] = []

    for fourcc_text in ("avc1", "H264", "x264"):
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*fourcc_text),
            fps,
            (info.width, info.height),
        )
        if not writer.isOpened():
            failures.append(fourcc_text)
            continue

        written = 0
        try:
            for frame in iter_video_frames(input_video):
                writer.write(frame)
                written += 1
        finally:
            writer.release()

        if written > 0 and output_video.exists() and output_video.stat().st_size > 0:
            return fourcc_text

        failures.append(fourcc_text)

    raise RuntimeError(
        "Could not encode H.264 output. Install ffmpeg or use an OpenCV build with "
        f"H.264 support. Tried codecs: {', '.join(failures)}"
    )


def make_h264_output(input_video: Path, output_video: Path, fps: float) -> str:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if transcode_with_ffmpeg(input_video, output_video):
        return "ffmpeg/libx264"
    return f"opencv/{transcode_with_opencv_h264(input_video, output_video, fps)}"


def probe_output(video_path: Path) -> dict[str, object]:
    info = read_video_info(video_path)
    capture = open_capture(video_path)
    try:
        ok, _ = capture.read()
    finally:
        capture.release()

    if not ok:
        raise RuntimeError(f"Output video has no readable first frame: {video_path}")

    return {
        "path": str(video_path),
        "size_bytes": video_path.stat().st_size,
        "fps": round(info.fps, 3),
        "width": info.width,
        "height": info.height,
        "frame_count": info.frame_count,
        "duration_seconds": round(info.duration_seconds, 3),
    }


def count_csv_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return max(0, sum(1 for _ in csv.reader(file)) - 1)


def run_pipeline(arguments: argparse.Namespace) -> Path:
    input_video = arguments.input_video.resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video was not found: {input_video}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (arguments.output_dir or OUTPUT_ROOT / f"pipeline_{run_id}").resolve()
    segments_dir = run_dir / "segments"
    logs_dir = run_dir / "logs"
    driver_dir = run_dir / "driver_state_detection"
    seatbelt_dir = run_dir / "seatbelt_detection"
    evidence_dir = seatbelt_dir / "evidence"

    for directory in (segments_dir, logs_dir, driver_dir, seatbelt_dir, evidence_dir):
        directory.mkdir(parents=True, exist_ok=True)

    driver_segment = segments_dir / "driver_state_000_030.mp4"
    seatbelt_segment = segments_dir / "seatbelt_030_end.mp4"
    driver_annotated = driver_dir / "driver_state_annotated.mp4"
    driver_predictions = driver_dir / "predictions.csv"
    seatbelt_annotated = seatbelt_dir / "seatbelt_annotated.mp4"
    seatbelt_events = seatbelt_dir / "seatbelt_events.csv"
    seatbelt_metrics = seatbelt_dir / "seatbelt_metrics.json"
    combined_mp4v = run_dir / "driver_demo_pipeline_combined_mp4v.mp4"
    h264_output = run_dir / "driver_demo_pipeline_h264.mp4"
    summary_path = run_dir / "pipeline_summary.json"

    input_info, driver_frames, seatbelt_frames = split_video_by_time(
        input_video=input_video,
        first_output=driver_segment,
        second_output=seatbelt_segment,
        split_seconds=arguments.driver_seconds,
    )

    driver_events = run_driver_state_stage(
        input_segment=driver_segment,
        output_video=driver_annotated,
        predictions_csv=driver_predictions,
        log_path=logs_dir / "driver_state_detection.log",
        camera_params=arguments.camera_params,
    )

    run_seatbelt_stage(
        input_segment=seatbelt_segment,
        output_video=seatbelt_annotated,
        event_log=seatbelt_events,
        metrics_path=seatbelt_metrics,
        evidence_dir=evidence_dir,
        driver_side=arguments.driver_side,
        process_every=arguments.process_every,
        max_frames=arguments.max_frames,
    )

    combined_frames = concatenate_videos(
        [driver_annotated, seatbelt_annotated],
        combined_mp4v,
        input_info.fps,
    )
    encoder = make_h264_output(combined_mp4v, h264_output, input_info.fps)
    output_probe = probe_output(h264_output)

    summary = {
        "input_video": str(input_video),
        "input": {
            "fps": input_info.fps,
            "width": input_info.width,
            "height": input_info.height,
            "frame_count": input_info.frame_count,
            "duration_seconds": input_info.duration_seconds,
        },
        "split": {
            "driver_state_seconds": arguments.driver_seconds,
            "driver_state_frames": driver_frames,
            "seatbelt_frames": seatbelt_frames,
        },
        "outputs": {
            "run_dir": str(run_dir),
            "driver_segment": str(driver_segment),
            "seatbelt_segment": str(seatbelt_segment),
            "driver_annotated": str(driver_annotated),
            "seatbelt_annotated": str(seatbelt_annotated),
            "combined_mp4v": str(combined_mp4v),
            "h264_video": str(h264_output),
            "driver_predictions": str(driver_predictions),
            "driver_events": str(driver_events),
            "seatbelt_events": str(seatbelt_events),
            "seatbelt_metrics": str(seatbelt_metrics),
            "seatbelt_evidence_dir": str(evidence_dir),
        },
        "events": {
            "driver_state_event_count": count_csv_rows(driver_events),
            "seatbelt_event_count": count_csv_rows(seatbelt_events),
        },
        "encoding": {
            "requested": "h264",
            "encoder": encoder,
        },
        "verification": output_probe,
        "combined_frame_count": combined_frames,
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("=" * 80)
    print("ITMS PIPELINE COMPLETE")
    print("=" * 80)
    print("Final H.264 video:", h264_output)
    print("Summary:", summary_path)
    print("Driver-state events:", summary["events"]["driver_state_event_count"])
    print("Seatbelt events:", summary["events"]["seatbelt_event_count"])
    print("Encoder:", encoder)
    print("Frames:", output_probe["frame_count"])
    print("Resolution:", f"{output_probe['width']}x{output_probe['height']}")
    print("=" * 80)

    return h264_output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ITMS 30s driver-state + remaining seatbelt video pipeline."
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=DEFAULT_INPUT_VIDEO,
        help="Input video to process.",
    )
    parser.add_argument(
        "--driver-seconds",
        type=float,
        default=DEFAULT_DRIVER_SECONDS,
        help="Seconds from the beginning of the video assigned to driver-state detection.",
    )
    parser.add_argument(
        "--driver-side",
        choices=["left", "right"],
        default=DEFAULT_DRIVER_SIDE,
        help="Driver side used by the seatbelt detector.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=DEFAULT_PROCESS_EVERY,
        help="Run seatbelt inference every N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional seatbelt max frame cap. Use 0 for the full seatbelt segment.",
    )
    parser.add_argument(
        "--camera-params",
        type=Path,
        default=CAMERA_PARAMS_PATH,
        help="Camera calibration JSON used by driver-state head-pose estimation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for the pipeline run.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.driver_seconds <= 0:
        raise ValueError("--driver-seconds must be positive.")
    if arguments.process_every < 1:
        raise ValueError("--process-every must be at least 1.")
    run_pipeline(arguments)


if __name__ == "__main__":
    main()
