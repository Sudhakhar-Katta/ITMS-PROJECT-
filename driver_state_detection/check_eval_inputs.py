import csv
from collections import Counter
from pathlib import Path


PRED_PATH = Path(r"C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\completed_predictions.csv")
LABEL_PATH = Path(r"C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\labels.csv")


def load_csv(path):
    if not path.exists():
        print(f"Missing file: {path}")
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    preds = load_csv(PRED_PATH)
    labels = load_csv(LABEL_PATH)

    print("\n=== Prediction file ===")
    print(f"Prediction rows: {len(preds)}")

    pred_videos = sorted(set(row.get("video", "") for row in preds))
    print(f"Prediction videos: {len(pred_videos)}")
    print("Sample prediction videos:")
    for v in pred_videos[:5]:
        print(" ", v)

    print("\n=== Labels file ===")
    print(f"Label rows: {len(labels)}")

    label_videos = sorted(set(row.get("video", "") for row in labels))
    print(f"Label videos: {len(label_videos)}")
    print("Sample label videos:")
    for v in label_videos[:5]:
        print(" ", v)

    label_counts = Counter(row.get("label", "") for row in labels)
    print("\nLabel counts:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")

    matching_videos = set(pred_videos) & set(label_videos)
    print("\n=== Matching ===")
    print(f"Videos appearing in both predictions and labels: {len(matching_videos)}")

    if len(matching_videos) == 0:
        print("\nPROBLEM: Video names in labels.csv do not match video names in completed_predictions.csv")

    unmatched_labels = sorted(set(label_videos) - set(pred_videos))
    unmatched_preds = sorted(set(pred_videos) - set(label_videos))

    print("\nLabel videos not found in predictions:")
    for v in unmatched_labels[:10]:
        print(" ", v)

    print("\nPrediction videos not found in labels:")
    for v in unmatched_preds[:10]:
        print(" ", v)

    print("\n=== Prediction positive frame counts ===")
    pred_cols = [
        "drowsiness_active",
        "tired",
        "asleep",
        "looking_away",
        "distracted",
        "yawning",
    ]

    for col in pred_cols:
        count = sum(str(row.get(col, "")).lower() == "true" for row in preds)
        print(f"{col}: {count}")


if __name__ == "__main__":
    main()