import argparse
import csv


def bool_from_csv(value):
    return str(value).strip().lower() in ["true", "1", "yes"]


def load_labels(labels_path):
    labels = []

    with open(labels_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            labels.append(
                {
                    "video": row["video"],
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
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
    }


def evaluate_frame_level(predictions_path, labels, label_name, pred_column):
    tp = fp = fn = tn = 0

    with open(predictions_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            video = row["video"]
            time_sec = float(row["time_sec"])

            y_true = label_active(labels, video, time_sec, label_name)
            y_pred = bool_from_csv(row[pred_column])

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


def evaluate_event_level(predictions_path, labels, label_name, event_name):
    """
    Event-level evaluation is better for micro_sleep because it fires once,
    not continuously every frame.
    """
    true_events = [
        label for label in labels
        if label["label"] == label_name
    ]

    matched_true_events = set()
    fp = 0

    with open(predictions_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
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
        "TN": "N/A for event-level",
        "accuracy": "N/A for event-level",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": "Use FP per hour instead",
    }


def print_metrics(name, result):
    print(f"\n=== {name} ===")

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)

    args = parser.parse_args()

    labels = load_labels(args.labels)

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
            predictions_path=args.predictions,
            labels=labels,
            label_name=module["label"],
            pred_column=module["pred_column"],
        )

        print_metrics(module["name"], result)

    micro_result = evaluate_event_level(
        predictions_path=args.predictions,
        labels=labels,
        label_name="micro_sleep",
        event_name="micro_sleep",
    )

    print_metrics("Micro Sleep", micro_result)


if __name__ == "__main__":
    main()