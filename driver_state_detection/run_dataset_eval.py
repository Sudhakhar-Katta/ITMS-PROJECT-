import argparse
import csv
import subprocess
import sys
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"}


def find_videos(videos_dir):
    videos = []

    for path in Path(videos_dir).rglob("*"):
        if path.is_file() and path.suffix in VIDEO_EXTS:
            videos.append(path)

    return sorted(videos)


def combine_prediction_files(prediction_files, combined_csv):
    prediction_files = [p for p in prediction_files if p.exists() and p.stat().st_size > 0]

    if not prediction_files:
        print("No prediction CSV files to combine.")
        return

    with open(combined_csv, "w", newline="", encoding="utf-8") as fout:
        writer = None

        for file_path in prediction_files:
            with open(file_path, "r", newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)

                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()

                for row in reader:
                    writer.writerow(row)

    print(f"\nCombined predictions saved to: {combined_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--combined_csv", required=True)

    parser.add_argument("--ear_thresh", type=float, default=0.22)
    parser.add_argument("--ear_time_thresh", type=float, default=3.0)
    parser.add_argument("--yawn_mar_thresh", type=float, default=0.55)
    parser.add_argument("--yawn_time_thresh", type=float, default=1.2)

    args = parser.parse_args()

    videos_dir = Path(args.videos)
    out_dir = Path(args.out_dir)
    combined_csv = Path(args.combined_csv)

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Looking for videos in: {videos_dir}")

    video_files = find_videos(videos_dir)
    print(f"Found {len(video_files)} video files.")

    if len(video_files) == 0:
        print("No videos found.")
        return

    all_prediction_files = []

    for video_path in video_files:
        video_name = video_path.name
        pred_csv = out_dir / f"{video_path.stem}_predictions.csv"
        all_prediction_files.append(pred_csv)

        if pred_csv.exists() and pred_csv.stat().st_size > 0:
            print(f"Skipping already processed: {video_name}")
            continue

        print(f"\nRunning detector on: {video_name}")
        print(f"Saving to: {pred_csv}")

        cmd = [
            sys.executable,
            "main.py",
            "--source",
            str(video_path),
            "--video_name",
            video_name,
            "--save_csv",
            str(pred_csv),
            "--ear_thresh",
            str(args.ear_thresh),
            "--ear_time_thresh",
            str(args.ear_time_thresh),
            "--yawn_mar_thresh",
            str(args.yawn_mar_thresh),
            "--yawn_time_thresh",
            str(args.yawn_time_thresh),
            "--no_display",
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"ERROR: main.py failed on {video_name}")
            continue

    print("\nCombining available prediction CSV files...")
    combine_prediction_files(all_prediction_files, combined_csv)

    completed = [
        p for p in all_prediction_files
        if p.exists() and p.stat().st_size > 0
    ]

    print(f"\nCompleted prediction files: {len(completed)} / {len(video_files)}")


if __name__ == "__main__":
    main()