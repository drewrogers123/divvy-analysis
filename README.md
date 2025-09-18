# Divvy + Weather Analysis

This project analyzes Divvy bike trips (2014–2017) with hourly weather to produce normalized 0–1 impact metrics for factors like time of day, weekday, weather, temperature, station popularity, distance, and user type. The goal is to understand how these factors affect trip duration. The analysis is based on the public domain Divvy dataset available on Kaggle: https://www.kaggle.com/datasets/yingwurenjian/chicago-divvy-bicycle-sharing-data/data

## Repository layout

- `divvy_analysis.py` — Main analysis script. Computes distance, speeds, impact metrics, and saves charts + an enriched CSV.
- `normalize_events.py` — Preprocesses the `events` column using custom rules (rain/snow thresholds, "not clear" → "cloudy").
- `list_events.py` — Lists unique values and counts in the `events` column for quick inspection.
- `analyze_divvy.py` — Lightweight data inspection helper (basic summary/stats).

## Requirements

- Python 3.9+
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`

### Optional: create and use a virtual environment

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install pandas numpy matplotlib seaborn tqdm
```

## Input data

Provide a Divvy+weather CSV including at least these columns:

- `trip_id, starttime, stoptime, tripduration`
- `year, month, week, day, hour`
- `usertype, gender`
- `temperature, events`
- `from_station_id, from_station_name, latitude_start, longitude_start, dpcapacity_start`
- `to_station_id, to_station_name, latitude_end, longitude_end, dpcapacity_end`

Example location used in commands below:

- `/Users/drewrogers/Downloads/divvyandweather.csv`

## Step 1: Normalize weather event labels (recommended)

Apply your rules to clean the `events` column, then write a cleaned CSV:

```bash
python normalize_events.py \
  --input "/Users/drewrogers/Downloads/divvyandweather.csv" \
  --output "divvyandweather_cleaned.csv" \
  --summary
```

Rules applied:

- "not clear" or "not_clear" → "cloudy"
- "rain or snow":
  - temperature > 41 → "rain"
  - 30 ≤ temperature ≤ 41 → "rain or snow"
  - temperature < 30 → "snow"

## Step 2: Inspect event categories (optional)

```bash
python list_events.py \
  --input "divvyandweather_cleaned.csv" \
  --output "weather_events_summary.txt"
```

This prints a summary and writes it to `weather_events_summary.txt`.

## Step 3: Run the analysis

Full dataset:

```bash
python divvy_analysis.py \
  --input "divvyandweather_cleaned.csv" \
  --output "divvy_analysis_output"
```

Quick test on the first N rows (fast feedback):

```bash
python divvy_analysis.py \
  --input "divvyandweather_cleaned.csv" \
  --output "divvy_analysis_output" \
  --sample 10000
```

## Outputs

- CSV: `divvy_analysis_output/divvy_trips_with_impact_metrics.csv`
  - Adds: `tripduration_min, distance_km, speed_kmh`
  - Impact metrics (0–1):
    - `time_of_day_impact, day_of_week_impact, weather_impact, temp_impact`
    - `start_station_impact, end_station_impact, distance_impact`
    - `user_type_impact, seasonal_impact, combined_impact`
- Charts (PNG) in `divvy_analysis_output/`:
  - `time_of_day_impact.png, day_of_week_impact.png`
  - `weather_impact.png, temperature_impact.png, seasonal_impact.png`
  - `impact_correlation.png`

## Notes on methodology

- Distances are computed with the vectorized Haversine formula (km).
- Impact metrics are normalized to 0–1 using median trip duration by category.
- `combined_impact` is a weighted average of individual impacts (weights are editable in `divvy_analysis.py`).
- For small samples, some visualizations are skipped automatically.

## Troubleshooting

- If you see Matplotlib/seaborn style errors, the script falls back to built-in styles. You can safely ignore style warnings.
- If you get a categorical/numeric error on `combined_impact`, ensure you’re running the latest script; impacts are coerced to numeric with neutral fill (0.5).
- Memory: 9.5M rows may require 2–4GB RAM. Use `--sample` for quick iteration.

## Next steps and ideas

- Add CLI flags to disable visualizations (e.g., `--no-viz`).
- Persist station popularity metrics for reuse across runs.
- Use Dask/Modin for parallel/out-of-core scaling if needed.
- Add geographic clustering for “neighborhood” factors.
