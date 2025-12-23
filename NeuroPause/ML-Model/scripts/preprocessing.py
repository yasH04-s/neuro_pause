# File: /NeuroPause/NeuroPause/ML-Model/scripts/preprocessing.py

"""
preprocessing.py

Full preprocessing pipeline for NeuroPause.

Expected input: CSV in `ML-Model/data/raw/` with columns:
    - timestamp
    - scroll_speed
    - swipe_count
    - touches_per_minute
    - session_duration
    - screen_on_off
    - active_app

Outputs saved to `ML-Model/data/processed/`:
    - `X_processed.npy` : numpy array of shape (N_windows, timesteps, features)
    - `y_labels.npy`    : numpy binary labels (N_windows,)

This file implements:
    - missing value handling
    - StandardScaler normalization
    - sliding window generation (30-second windows, adaptive to timestamp sampling)
    - simple heuristic label generation (0 mindful, 1 compulsive)

Usage: run this script from the project root or as a module. Paths are resolved
relative to the script location to avoid path issues.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw CSV into a DataFrame. Raises a clear error if file missing.

    Args:
        file_path: Path to raw CSV file.
    Returns:
        pd.DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values using forward/backward fill and drop remaining nulls.

    Args:
        df: Raw DataFrame
    Returns:
        Cleaned DataFrame
    """
    # Basic logging
    missing_before = df.isnull().sum()

    # Fill simple gaps
    df = df.sort_values('timestamp').reset_index(drop=True)
    # use recommended forward/backward fill APIs
    df = df.ffill().bfill()

    # If any critical columns are still missing, drop those rows
    critical = ['scroll_speed', 'swipe_count', 'touches_per_minute']
    df = df.dropna(subset=critical)

    missing_after = df.isnull().sum()
    print(f"Missing before:\n{missing_before}\nMissing after:\n{missing_after}")
    return df


def normalize_features(df: pd.DataFrame, scaler: StandardScaler = None):
    """Normalize numeric columns using StandardScaler. Returns scaler for reuse.

    Excludes 'timestamp', 'screen_on_off', 'active_app'.
    """
    exclude = ['timestamp', 'screen_on_off', 'active_app']
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler


def compute_sampling_interval_seconds(df: pd.DataFrame) -> float:
    """Estimate median sampling interval (in seconds) from timestamp column.
    If timestamps are not present or invalid, default to 1.0s.
    """
    if 'timestamp' not in df.columns:
        return 1.0
    diffs = df['timestamp'].diff().dt.total_seconds().dropna()
    if len(diffs) == 0:
        return 1.0
    median = float(diffs.median())
    if median <= 0:
        return 1.0
    return median


def create_sliding_windows(df: pd.DataFrame, window_seconds: int = 30, step_seconds: int = 10):
    """Create sliding windows based on timestamps. Returns list of DataFrames.

    Steps:
        - compute median sampling interval -> convert seconds to timesteps
        - slide over data producing windows as DataFrames
    """
    sampling_sec = compute_sampling_interval_seconds(df)
    window_size = max(1, int(round(window_seconds / sampling_sec)))
    step_size = max(1, int(round(step_seconds / sampling_sec)))

    windows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        win = df.iloc[start:start + window_size].reset_index(drop=True)
        windows.append(win)

    print(f"Created {len(windows)} windows (window_size={window_size}, step={step_size}, sampling_sec={sampling_sec:.2f}s)")
    return windows


def generate_labels(windows, threshold: float = 0.7):
    """Generate binary labels using a simple heuristic (compulsive_score).

    compulsive_score computed from avg_scroll_speed, avg_swipe_count, avg_touches.
    Returns numpy array of 0/1 labels.
    """
    labels = []
    for w in windows:
        avg_scroll = float(w['scroll_speed'].mean())
        avg_swipes = float(w['swipe_count'].mean())
        avg_touches = float(w['touches_per_minute'].mean())

        # heuristic (values expected to be normalized already)
        compulsive_score = (abs(avg_scroll) + abs(avg_swipes) + abs(avg_touches)) / 3.0
        label = 1 if compulsive_score > threshold else 0
        labels.append(label)
    labels = np.array(labels, dtype=np.int64)
    print(f"Labels distribution: {{0: {int((labels==0).sum())}, 1: {int((labels==1).sum())}}}")
    return labels


def windows_to_numpy(windows, feature_columns=None):
    """Convert list of DataFrame windows to numpy array (N, timesteps, features).

    If feature_columns is None, infer numeric columns (exclude timestamp and categorical).
    """
    if len(windows) == 0:
        return np.zeros((0, 0, 0))

    if feature_columns is None:
        candidate = [c for c in windows[0].columns if c not in ('timestamp', 'screen_on_off', 'active_app')]
        feature_columns = [c for c in candidate if pd.api.types.is_numeric_dtype(windows[0][c])]

    arr = np.array([w[feature_columns].to_numpy(dtype=np.float32) for w in windows])
    return arr, feature_columns


def save_numpy_arrays(X, y, output_dir: str):
    # If a file exists at the output_dir path (common on Windows), rename it first
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        backup = output_dir + '.bak'
        print(f"Path {output_dir} exists and is a file. Renaming to {backup} to proceed.")
        os.rename(output_dir, backup)
    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, 'X_processed.npy')
    y_path = os.path.join(output_dir, 'y_labels.npy')
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"Saved X -> {x_path} (shape={X.shape})")
    print(f"Saved y -> {y_path} (shape={y.shape})")


def main(raw_csv_path: str = None, output_dir: str = None):
    script_dir = Path(__file__).parent.parent  # ML-Model/
    if raw_csv_path is None:
        raw_csv_path = script_dir / 'data' / 'raw' / 'neuropause_raw.csv'
    if output_dir is None:
        output_dir = script_dir / 'data' / 'processed'

    raw_csv_path = str(raw_csv_path)
    output_dir = str(output_dir)

    print(f"Loading raw data from: {raw_csv_path}")
    df = load_raw_data(raw_csv_path)
    df = handle_missing_values(df)
    df, scaler = normalize_features(df)

    windows = create_sliding_windows(df, window_seconds=30, step_seconds=10)
    y = generate_labels(windows, threshold=0.7)

    X, feature_cols = windows_to_numpy(windows)

    # Save outputs
    save_numpy_arrays(X, y, output_dir)

    # Save feature columns for later reference
    feat_path = Path(output_dir) / 'feature_columns.txt'
    with open(feat_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"Saved feature column list to: {feat_path}")


if __name__ == '__main__':
    main()