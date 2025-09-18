import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2

# Load the data
print("Loading data...")
df = pd.read_csv('/Users/drewrogers/Downloads/divvyandweather.csv')

# Display basic info
print("\n=== Data Overview ===")
print(f"Number of trips: {len(df):,}")
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# Convert tripduration to minutes if it's in seconds
if df['tripduration'].mean() > 1000:  # assuming if average > 1000, it's in seconds
    df['tripduration_min'] = df['tripduration'] / 60
else:
    df['tripduration_min'] = df['tripduration']

# Display basic statistics
print("\n=== Trip Duration Statistics ===")
print(f"Average trip duration: {df['tripduration_min'].mean():.1f} minutes")
print(f"Median trip duration: {df['tripduration_min'].median():.1f} minutes")
print(f"Maximum trip duration: {df['tripduration_min'].max():.1f} minutes")
print(f"Minimum trip duration: {df['tripduration_min'].min():.1f} minutes")

# Check date range
if 'starttime' in df.columns:
    df['datetime'] = pd.to_datetime(df['starttime'])
    print(f"\nDate range: {df['datetime'].min()} to {df['datetime'].max()}")

# Check weather data
if 'events' in df.columns:
    print("\n=== Weather Conditions ===")
    print(df['events'].value_counts(dropna=False))

print("\nInitial data exploration complete. Ready to proceed with analysis.")
