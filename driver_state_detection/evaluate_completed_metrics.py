import argparse
import csv
from collections import defaultdict


def bool_from_csv(value):
    return str(value).strip().lower() in ["true", "1", "yes"]


def load_predictions(predictions_path):
    rows = []

    with open(predictions_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def load_labels(labels_path, completed_videos):
    labels = []

    with open(labels_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            video = row["video"]

            # Important: ignore labels for videos that are not completed yet
            if video not in completed_videos:
                continue

            labels.append(
                {
                    "video": video,
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "label": row["label"],
                }
            )

    return labels


def label_active(labels, video, time_sec, label_name):
    for label in labels:
        if label["video"] != video:
            continue

        if label["label"] != label_name:
            continue

        if label["start_sec"] <= time_sec <= label["end_sec"]:
            return True

    return False


def compute_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
    }


def evaluate_frame_level(pred_rows, labels, label_name, pred_column):
    tp = fp = fn = tn = 0

    for row in pred_rows:
        video = row["video"]
        time_sec = float(row["time_sec"])

        y_true = label_active(labels, video, time_sec, label_name)
        y_pred = bool_from_csv(row.get(pred_column, ""))

        if y_true and y_pred:
            tp += 1
        elif not y_true and y_pred:
            fp += 1
        elif y_true and not y_pred:
            fn += 1
        else:
            tn += 1

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        **compute_metrics(tp, fp, fn, tn),
    }


def evaluate_event_level(pred_rows, labels, label_name, event_name):
    true_events = [
        label for label in labels
        if label["label"] == label_name
    ]

    matched_true_events = set()
    fp = 0

    for row in pred_rows:
        events = row.get("events", "")

        if event_name not in events:
            continue

        video = row["video"]
        time_sec = float(row["time_sec"])

        matched = False

        for idx, label in enumerate(true_events):
            if idx in matched_true_events:
                continue

            if label["video"] != video:
                continue

            if label["start_sec"] <= time_sec <= label["end_sec"]:
                matched_true_events.add(idx)
                matched = True
                break

        if not matched:
            fp += 1

    tp = len(matched_true_events)
    fn = len(true_events) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": "N/A event-level",
        "accuracy": "N/A event-level",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": "Use FP per hour for event-level",
    }


def print_result(name, result):
    print(f"\n=== {name} ===")

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def prediction_summary(pred_rows):
    print("\n=== Prediction Summary ===")

    videos = sorted(set(row["video"] for row in pred_rows))
    print(f"Completed videos in predictions: {len(videos)}")

    frame_count = len(pred_rows)
    print(f"Total prediction rows/frames: {frame_count}")

    columns_to_count = [
        "drowsiness_active",
        "tired",
        "asleep",
        "looking_away",
        "distracted",
        "yawning",
    ]

    for col in columns_to_count:
        count = sum(bool_from_csv(row.get(col, "")) for row in pred_rows)
        print(f"{col}: {count} frames")

    event_counts = defaultdict(int)

    for row in pred_rows:
        events = row.get("events", "")

        if not events:
            continue

        for event in events.split("|"):
            if event.strip():
                event_counts[event.strip()] += 1

    print("\nEvent counts:")
    if not event_counts:
        print("No events found.")
    else:
        for event, count in sorted(event_counts.items()):
            print(f"{event}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)

    args = parser.parse_args()

    pred_rows = load_predictions(args.predictions)
    completed_videos = set(row["video"] for row in pred_rows)

    prediction_summary(pred_rows)

    labels = load_labels(args.labels, completed_videos)

    print(f"\nLoaded labels for completed videos only: {len(labels)}")

    modules = [
        {
            "name": "Driver Drowsiness",
            "label": "drowsiness",
            "pred_column": "drowsiness_active",
        },
        {
            "name": "Tired / PERCLOS",
            "label": "tired",
            "pred_column": "tired",
        },
        {
            "name": "Distracted",
            "label": "distracted",
            "pred_column": "distracted",
        },
        {
            "name": "Yawning",
            "label": "yawning",
            "pred_column": "yawning",
        },
    ]

    for module in modules:
        result = evaluate_frame_level(
            pred_rows=pred_rows,
            labels=labels,
            label_name=module["label"],
            pred_column=module["pred_column"],
        )

        print_result(module["name"], result)

    micro_result = evaluate_event_level(
        pred_rows=pred_rows,
        labels=labels,
        label_name="micro_sleep",
        event_name="micro_sleep",
    )

    print_result("Micro Sleep", micro_result)


if __name__ == "__main__":
    main()