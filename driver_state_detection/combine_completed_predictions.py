import csv
from pathlib import Path


PRED_DIR = Path(r"C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\predictions")
OUT_CSV = Path(r"C:\Users\valla\ITMS-SYSTEM1\datasets\cabin_eval\completed_predictions.csv")


def main():
    pred_files = sorted(PRED_DIR.glob("*_predictions.csv"))

    pred_files = [
        p for p in pred_files
        if p.exists() and p.stat().st_size > 0
    ]

    print(f"Found completed prediction files: {len(pred_files)}")

    if not pred_files:
        print("No prediction CSV files found.")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = None

        for file_path in pred_files:
            print(f"Adding: {file_path.name}")

            with open(file_path, "r", newline="", encoding="utf-8") as fin:
                reader = csv.DictReader(fin)

                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
                    writer.writeheader()

                for row in reader:
                    writer.writerow(row)

    print(f"\nSaved combined completed predictions to:")
    print(OUT_CSV)


if __name__ == "__main__":
    main()