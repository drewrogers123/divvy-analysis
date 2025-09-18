import pandas as pd
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="List unique values and counts from the 'events' column.")
    parser.add_argument(
        "--input", "-i",
        default="/Users/drewrogers/Downloads/divvyandweather.csv",
        help="Path to input CSV (Divvy + weather)"
    )
    parser.add_argument(
        "--output", "-o",
        default="weather_events_summary.txt",
        help="Path to write the text summary (default: weather_events_summary.txt)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading: {input_path}")
    # Only load the events column for speed and memory efficiency
    df = pd.read_csv(input_path, usecols=["events"])  # Assumes 'events' column exists

    print("Computing unique events and counts...")
    counts = df["events"].value_counts(dropna=False)

    # Prepare pretty output
    lines = []
    total = int(counts.sum())
    lines.append("Weather events summary\n")
    lines.append(f"Total rows: {total:,}\n")
    lines.append("Event\tCount\tPercent\n")
    for event, count in counts.items():
        label = "<NaN>" if pd.isna(event) else str(event)
        pct = (count / total) * 100 if total else 0
        lines.append(f"{label}\t{count:,}\t{pct:0.2f}%\n")

    # Write to file
    out_path = Path(args.output)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Summary written to: {out_path.resolve()}")

    # Also print to stdout
    print("\n=== Unique 'events' values (sorted by count) ===")
    for line in lines[2:]:  # skip header lines
        print(line.strip())


if __name__ == "__main__":
    main()
