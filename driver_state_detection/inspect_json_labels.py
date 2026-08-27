import json
from pathlib import Path
from collections import Counter


JSON_DIR = Path(r"C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\json")


def walk(obj, depth=0, max_depth=5):
    if depth > max_depth:
        return

    indent = "  " * depth

    if isinstance(obj, dict):
        print(f"{indent}dict keys: {list(obj.keys())[:20]}")
        for key, value in list(obj.items())[:5]:
            print(f"{indent}- {key}: {type(value).__name__}")
            walk(value, depth + 1, max_depth)

    elif isinstance(obj, list):
        print(f"{indent}list length: {len(obj)}")
        if obj:
            print(f"{indent}first item type: {type(obj[0]).__name__}")
            walk(obj[0], depth + 1, max_depth)

    else:
        print(f"{indent}{repr(obj)[:120]}")


def collect_possible_labels(obj, labels):
    label_keys = {
        "label",
        "class",
        "category",
        "activity",
        "action",
        "behavior",
        "state",
        "name",
        "type",
    }

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in label_keys and isinstance(v, str):
                labels.append(v)
            collect_possible_labels(v, labels)

    elif isinstance(obj, list):
        for item in obj:
            collect_possible_labels(item, labels)


def main():
    json_files = sorted(JSON_DIR.glob("*.json"))

    print(f"Found JSON files: {len(json_files)}")

    if not json_files:
        print("No JSON files found.")
        return

    for json_file in json_files[:3]:
        print("\n" + "=" * 100)
        print(json_file.name)
        print("=" * 100)

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        walk(data, max_depth=4)

        labels = []
        collect_possible_labels(data, labels)

        print("\nPossible label values found:")
        counts = Counter(labels)
        for label, count in counts.most_common(30):
            print(f"  {label}: {count}")


if __name__ == "__main__":
    main()