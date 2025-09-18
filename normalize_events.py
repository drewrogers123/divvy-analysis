import argparse
from pathlib import Path
import pandas as pd


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    if 'events' not in df.columns:
        raise KeyError("Input CSV must contain an 'events' column")
    if 'temperature' not in df.columns:
        raise KeyError("Input CSV must contain a 'temperature' column for rain/snow disambiguation")

    # Work on a copy
    out = df.copy()

    # Standardize spacing/case for events
    out['events'] = out['events'].astype(str).str.strip().str.lower()

    # Map variants of not clear -> cloudy
    out['events'] = out['events'].replace({
        'not clear': 'cloudy',
        'not_clear': 'cloudy'
    })

    # Temperature-based disambiguation for 'rain or snow'
    ros_mask = out['events'].str.contains('rain or snow', na=False)
    # > 41 F => rain
    out.loc[ros_mask & (out['temperature'] > 41), 'events'] = 'rain'
    # < 30 F => snow
    out.loc[ros_mask & (out['temperature'] < 30), 'events'] = 'snow'
    # 30..41 F (inclusive) => stays 'rain or snow'
    out.loc[ros_mask & out['temperature'].between(30, 41, inclusive='both'), 'events'] = 'rain or snow'

    return out


def main():
    parser = argparse.ArgumentParser(description="Normalize the 'events' column per defined weather rules.")
    parser.add_argument(
        "--input", "-i",
        default="/Users/drewrogers/Downloads/divvyandweather.csv",
        help="Path to input CSV"
    )
    parser.add_argument(
        "--output", "-o",
        default="divvyandweather_cleaned.csv",
        help="Path to write cleaned CSV"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a before/after value counts summary"
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"Loading: {in_path}")
    usecols = None
    # Load only necessary columns if present to save memory; else load all
    try:
        df = pd.read_csv(in_path, usecols=['events', 'temperature'])
        # But we need to write the whole CSV back out; reload all if we succeeded above
        df_full = pd.read_csv(in_path)
        df = df_full
    except ValueError:
        # 'events'/'temperature' not in usecols, load entire file
        df = pd.read_csv(in_path)

    if args.summary:
        print("\nBefore normalization (top 10):")
        print(df['events'].value_counts(dropna=False).head(10))

    cleaned = normalize_events(df)

    out_path = Path(args.output)
    cleaned.to_csv(out_path, index=False)
    print(f"\nWrote cleaned CSV to: {out_path.resolve()}")

    if args.summary:
        print("\nAfter normalization (top 10):")
        print(cleaned['events'].value_counts(dropna=False).head(10))


if __name__ == "__main__":
    main()
