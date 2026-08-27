import argparse
import csv
import json
import shutil
from pathlib import Path


VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]


def safe_name(path: Path, root: Path):
    rel = path.relative_to(root)
    parts = rel.parts
    name = "__".join(parts)
    return name.replace(" ", "_")


def find_json_for_video(video_path: Path):
    folder = video_path.parent

    same_stem = folder / f"{video_path.stem}.json"
    if same_stem.exists():
        return same_stem

    json_files = list(folder.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True, help="Raw downloaded dataset folder")
    parser.add_argument("--out_root", required=True, help="Clean output dataset folder")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    videos_out = out_root / "videos"
    json_out = out_root / "json"
    predictions_out = out_root / "predictions"

    videos_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)
    predictions_out.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.csv"

    video_files = []
    for ext in VIDEO_EXTS:
        video_files.extend(raw_root.rglob(f"*{ext}"))

    print(f"Found {len(video_files)} videos")

    rows = []

    for video_path in video_files:
        new_video_name = safe_name(video_path, raw_root)
        new_video_path = videos_out / new_video_name

        shutil.copy2(video_path, new_video_path)

        json_path = find_json_for_video(video_path)

        new_json_name = ""
        new_json_path = ""

        if json_path is not None:
            new_json_name = safe_name(json_path, raw_root)
            new_json_path_obj = json_out / new_json_name
            shutil.copy2(json_path, new_json_path_obj)
            new_json_path = str(new_json_path_obj)

        rows.append(
            {
                "video": new_video_name,
                "json": new_json_name,
                "original_video_path": str(video_path),
                "original_json_path": str(json_path) if json_path else "",
                "clean_video_path": str(new_video_path),
                "clean_json_path": new_json_path,
            }
        )

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "json",
                "original_video_path",
                "original_json_path",
                "clean_video_path",
                "clean_json_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done.")
    print(f"Videos copied to: {videos_out}")
    print(f"JSON copied to: {json_out}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()