import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2
import os
from tqdm import tqdm
import argparse

# Set up visualization style
try:
    plt.style.use('ggplot')  # guaranteed built-in style
except Exception:
    # Fall back to default style if ggplot is unavailable
    plt.style.use('default')

# Try to set a seaborn palette, but don't fail if seaborn style/palette isn't available
try:
    sns.set_palette('viridis')
except Exception:
    pass
plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['font.size'] = 12

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on the earth."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def normalize_metric(series, invert=False):
    """Normalize a series to 0-1 range"""
    if len(series) == 0:
        return pd.Series()
    if invert:
        return 1 - ((series - series.min()) / (series.max() - series.min() + 1e-10))
    return (series - series.min()) / (series.max() - series.min() + 1e-10)

def analyze_divvy_data(input_file, output_dir='output', sample_size=None, visualize=False):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    if sample_size:
        print(f"  [TEST MODE] Loading only first {sample_size} rows...")
        df = pd.read_csv(input_file, nrows=sample_size)
    else:
        df = pd.read_csv(input_file)

    # Note: 'events' normalization is handled by normalize_events.py before analysis
    
    # Convert tripduration to minutes if it's in seconds
    if df['tripduration'].mean() > 1000:  # assuming if average > 1000, it's in seconds
        df['tripduration_min'] = df['tripduration'] / 60
    else:
        df['tripduration_min'] = df['tripduration']
    
    # Convert to datetime
    df['datetime'] = pd.to_datetime(df['starttime'])
    
    # Create time-based features
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['datetime'].dt.month
    df['season'] = ((df['month'] % 12 + 3) // 3).replace({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})
    
    # Calculate distance between stations (vectorized for speed)
    print("Calculating distances between stations (vectorized)...")
    R = 6371.0
    lat1 = np.radians(df['latitude_start'].to_numpy())
    lon1 = np.radians(df['longitude_start'].to_numpy())
    lat2 = np.radians(df['latitude_end'].to_numpy())
    lon2 = np.radians(df['longitude_end'].to_numpy())
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df['distance_km'] = R * c
    
    # Calculate speed (km/h)
    df['speed_kmh'] = (df['distance_km'] / (df['tripduration_min'] / 60)).replace([np.inf, -np.inf], np.nan)
    
    # Create impact metrics (0-1)
    print("\nCalculating impact metrics...")
    
    # 1. Time of Day Impact
    time_impact = df.groupby('hour_of_day')['tripduration_min'].median()
    df['time_of_day_impact'] = df['hour_of_day'].map(
        lambda x: normalize_metric(time_impact, invert=False).get(x, 0.5)
    )
    
    # 2. Day of Week Impact
    dow_impact = df.groupby('day_of_week')['tripduration_min'].median()
    df['day_of_week_impact'] = df['day_of_week'].map(
        lambda x: normalize_metric(dow_impact, invert=False).get(x, 0.5)
    )
    
    # 3. Weather Impact
    weather_impact = df.groupby('events')['tripduration_min'].median()
    df['weather_impact'] = df['events'].map(
        lambda x: normalize_metric(weather_impact, invert=False).get(x, 0.5) if pd.notnull(x) else 0.5
    )
    
    # 4. Temperature Impact
    temp_bins = pd.cut(df['temperature'], bins=10, duplicates='drop')
    temp_impact = df.groupby(temp_bins)['tripduration_min'].median()
    temp_impact_dict = {interval: val for interval, val in zip(temp_impact.index, normalize_metric(temp_impact, invert=False))}
    df['temp_impact'] = temp_bins.map(lambda x: temp_impact_dict.get(x, 0.5))
    
    # 5. Station Popularity Impact
    print("Analyzing station popularity...")
    station_popularity = pd.concat([
        df['from_station_id'].value_counts(),
        df['to_station_id'].value_counts()
    ]).groupby(level=0).sum()
    
    df['start_station_impact'] = df['from_station_id'].map(
        lambda x: normalize_metric(station_popularity, invert=True).get(x, 0.5)
    )
    df['end_station_impact'] = df['to_station_id'].map(
        lambda x: normalize_metric(station_popularity, invert=True).get(x, 0.5)
    )
    
    # 6. Distance Impact
    df['distance_impact'] = normalize_metric(df['distance_km'], invert=False)
    
    # 7. User Type Impact
    user_impact = df.groupby('usertype')['tripduration_min'].median()
    df['user_type_impact'] = df['usertype'].map(
        lambda x: normalize_metric(user_impact, invert=False).get(x, 0.5)
    )
    
    # 8. Gender Impact (if applicable)
    if 'gender' in df.columns:
        gender_impact = df.groupby('gender')['tripduration_min'].median()
        df['gender_impact'] = df['gender'].map(
            lambda x: normalize_metric(gender_impact, invert=False).get(x, 0.5) if pd.notnull(x) else 0.5
        )
    
    # 9. Seasonal Impact
    season_impact = df.groupby('season')['tripduration_min'].median()
    df['seasonal_impact'] = df['season'].map(
        lambda x: normalize_metric(season_impact, invert=False).get(x, 0.5)
    )
    
    # 10. Combined Impact Score (weighted average of all impacts)
    weights = {
        'time_of_day_impact': 0.25,
        'day_of_week_impact': 0.15,
        'weather_impact': 0.15,
        'temp_impact': 0.1,
        'distance_impact': 0.15,
        'seasonal_impact': 0.1,
        'user_type_impact': 0.1
    }
    
    # Add gender weight if available
    if 'gender_impact' in df.columns:
        weights['gender_impact'] = 0.05
        # Reduce other weights proportionally
        total_weight = sum(weights.values())
        for k in weights:
            if k != 'gender_impact':
                weights[k] = weights[k] * (1 - 0.05) / (total_weight - 0.05)
    
    # Ensure all impact columns are numeric
    for col in weights.keys():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.5)  # Fill missing with neutral 0.5
    
    # Calculate combined impact
    df['combined_impact'] = sum(df[col] * weight for col, weight in weights.items())
    
    # Create visualizations only if enabled and we have sufficient data
    if visualize and len(df) >= 10:  # Minimum rows needed for meaningful visualizations
        print("\nGenerating visualizations...")
        
        try:
            # 1. Time of Day Impact
            plt.figure(figsize=(14, 7))
            time_impact_plot = df.groupby('hour_of_day')['tripduration_min'].median()
            if len(time_impact_plot) > 1:  # Need at least 2 points for a line plot
                ax = sns.lineplot(x=time_impact_plot.index, y=normalize_metric(time_impact_plot))
                plt.title('Normalized Trip Duration Impact by Hour of Day', fontsize=16)
                plt.xlabel('Hour of Day', fontsize=14)
                plt.ylabel('Impact (0-1)', fontsize=14)
                plt.xticks(range(0, 24, 2))
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/time_of_day_impact.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Day of Week Impact
            plt.figure(figsize=(12, 6))
            dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_impact_plot = df.groupby('day_of_week')['tripduration_min'].median()
            if len(dow_impact_plot) > 1:  # Need at least 2 days
                ax = sns.barplot(x=dow_labels[:len(dow_impact_plot)], y=normalize_metric(dow_impact_plot))
                plt.title('Normalized Trip Duration Impact by Day of Week', fontsize=16)
                plt.xlabel('Day of Week', fontsize=14)
                plt.ylabel('Impact (0-1)', fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/day_of_week_impact.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Weather Impact
            if 'events' in df.columns and len(df['events'].unique()) > 1:
                plt.figure(figsize=(12, 6))
                weather_impact_plot = df.groupby('events')['tripduration_min'].median().sort_values(ascending=False)
                if len(weather_impact_plot) > 1:  # Need at least 2 weather conditions
                    ax = sns.barplot(y=weather_impact_plot.index, x=normalize_metric(weather_impact_plot), orient='h')
                    plt.title('Normalized Trip Duration Impact by Weather Condition', fontsize=16)
                    plt.xlabel('Impact (0-1)', fontsize=14)
                    plt.ylabel('Weather Condition', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/weather_impact.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 4. Temperature Impact
            if 'temperature' in df.columns and len(df['temperature'].unique()) > 1:
                plt.figure(figsize=(12, 6))
                temp_bins = pd.cut(df['temperature'], bins=min(10, len(df)//2))  # Fewer bins for small samples
                temp_impact_plot = df.groupby(temp_bins)['tripduration_min'].median()
                if len(temp_impact_plot) > 1:  # Need at least 2 temperature ranges
                    temp_impact_plot.index = [f"{i.left:.1f}-{i.right:.1f}°C" for i in temp_impact_plot.index]
                    ax = sns.barplot(y=temp_impact_plot.index, x=normalize_metric(temp_impact_plot), orient='h')
                    plt.title('Normalized Trip Duration Impact by Temperature Range', fontsize=16)
                    plt.xlabel('Impact (0-1)', fontsize=14)
                    plt.ylabel('Temperature Range', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/temperature_impact.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 5. Seasonal Impact
            if 'season' in df.columns and len(df['season'].unique()) > 1:
                plt.figure(figsize=(10, 6))
                season_order = ['Winter', 'Spring', 'Summer', 'Fall']
                seasonal_impact_plot = df.groupby('season')['tripduration_min'].median().reindex(season_order).dropna()
                if len(seasonal_impact_plot) > 1:  # Need at least 2 seasons
                    ax = sns.barplot(x=seasonal_impact_plot.index, y=normalize_metric(seasonal_impact_plot), order=season_order)
                    plt.title('Normalized Trip Duration Impact by Season', fontsize=16)
                    plt.xlabel('Season', fontsize=14)
                    plt.ylabel('Impact (0-1)', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/seasonal_impact.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 6. Correlation Heatmap (only if we have enough impact metrics)
            impact_columns = [col for col in df.columns if 'impact' in col and col != 'combined_impact']
            if len(impact_columns) >= 2:  # Need at least 2 impact metrics
                plt.figure(figsize=(12, 10))
                corr = df[impact_columns].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, fmt=".2f")
                plt.title('Correlation Between Different Impact Factors', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/impact_correlation.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"\nWarning: Could not generate some visualizations: {str(e)}")
    else:
        print("\nSkipping visualizations - insufficient data (need at least 10 rows)")
    
    # Save the enhanced dataset
    print("\nSaving results...")
    output_columns = [
        'trip_id', 'starttime', 'stoptime', 'tripduration_min', 'distance_km', 'speed_kmh',
        'from_station_name', 'to_station_name', 'temperature', 'events',
        'time_of_day_impact', 'day_of_week_impact', 'weather_impact', 'temp_impact',
        'start_station_impact', 'end_station_impact', 'distance_impact',
        'user_type_impact', 'seasonal_impact', 'combined_impact'
    ]
    
    # Add gender impact if it exists
    if 'gender_impact' in df.columns:
        output_columns.append('gender_impact')
    
    # Save to CSV
    output_file = f'{output_dir}/divvy_trips_with_impact_metrics.csv'
    df[output_columns].to_csv(output_file, index=False)
    
    print(f"\nAnalysis complete! Results saved to {output_file}")
    print(f"Visualizations saved to the '{output_dir}' directory.")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Divvy trips with weather and compute 0-1 impact metrics.")
    parser.add_argument("--input", "-i", default="/Users/drewrogers/Downloads/divvyandweather.csv", help="Path to input CSV (Divvy + weather)")
    parser.add_argument("--output", "-o", default="divvy_analysis_output", help="Directory to write results")
    parser.add_argument("--sample", type=int, default=None, 
                       help="Process only the first N rows (for testing)")
    parser.add_argument("--viz", action="store_true", help="Enable chart generation (disabled by default)")
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output

    print(f"Starting Divvy Trip Analysis for {input_file}")
    print("=" * 60)

    try:
        df = analyze_divvy_data(input_file, output_dir, sample_size=args.sample, visualize=args.viz)
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred during analysis: {str(e)}")
        raise
