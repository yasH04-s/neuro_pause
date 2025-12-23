import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

"""
generate_sample_data.py

Creates synthetic raw interaction data for NeuroPause to allow quick testing of preprocessing and
feature extraction pipelines.

Output: ML-Model/data/raw/neuropause_raw.csv
Columns:
    - timestamp
    - scroll_speed
    - swipe_count
    - touches_per_minute
    - session_duration
    - screen_on_off
    - active_app

Usage:
    python scripts/generate_sample_data.py --num-records 2000 --compulsive-fraction 0.05
"""


def generate_sample_data(num_records: int = 2000, compulsive_fraction: float = 0.02, output_dir: str = None):
    script_dir = Path(__file__).parent.parent
    if output_dir is None:
        output_dir = script_dir / 'data' / 'raw'
    else:
        output_dir = Path(output_dir)

    # Safety: if a file exists at the directory path, rename it
    if output_dir.exists() and not output_dir.is_dir():
        backup_path = output_dir.with_name(output_dir.name + '.bak')
        print(f"Path {output_dir} exists and is a file. Renaming to {backup_path} to proceed.")
        output_dir.rename(backup_path)

    os.makedirs(output_dir, exist_ok=True)

    start_time = datetime.now() - timedelta(days=1)
    sampling_sec = 3
    timestamps = [start_time + timedelta(seconds=i * sampling_sec) for i in range(num_records)]

    # base (non-compulsive) behavior
    scroll_speed = np.abs(np.random.normal(loc=0.5, scale=0.4, size=num_records))
    swipe_count = np.random.poisson(lam=2.5, size=num_records)
    touches_per_minute = np.abs(np.random.normal(loc=18, scale=7, size=num_records))
    session_duration = np.random.uniform(low=10, high=600, size=num_records)
    screen_on_off = np.random.choice([0, 1], size=num_records, p=[0.05, 0.95])
    active_app = np.random.choice(['TikTok', 'Instagram', 'YouTube', 'Twitter', 'News'], size=num_records)

    # Introduce compulsive records.
    # Option A: random isolated rows (from compulsive_fraction)
    num_compulsive = int(num_records * float(compulsive_fraction))
    if num_compulsive > 0:
        comp_idx = np.random.choice(num_records, size=num_compulsive, replace=False)
        for i in comp_idx:
            scroll_speed[i] = abs(scroll_speed[i] * np.random.uniform(3.0, 6.0))
            swipe_count[i] = int(swipe_count[i] + np.random.poisson(lam=6))
            touches_per_minute[i] = abs(touches_per_minute[i] * np.random.uniform(2.0, 4.0))
            active_app[i] = np.random.choice(['TikTok', 'YouTube'])

    # Option B: contiguous compulsive blocks (more realistic). If user passed
    # environment variable COMPULSIVE_BLOCKS or set via CLI, those will be handled
    # in the CLI wrapper below. The default CLI will not create blocks unless
    # arguments are provided.

    data = {
        'timestamp': timestamps,
        'scroll_speed': scroll_speed,
        'swipe_count': swipe_count,
        'touches_per_minute': touches_per_minute,
        'session_duration': session_duration,
        'screen_on_off': screen_on_off,
        'active_app': active_app
    }

    df = pd.DataFrame(data)
    out_path = output_dir / 'neuropause_raw.csv'
    df.to_csv(out_path, index=False)
    print(f"Generated sample data: {out_path} (shape={df.shape}), compulsive_fraction={compulsive_fraction}")
    return out_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic NeuroPause raw data')
    parser.add_argument('--num-records', type=int, default=2000, help='Number of records to generate')
    parser.add_argument('--compulsive-fraction', type=float, default=0.02, help='Fraction of rows to make compulsive (0-1)')
    parser.add_argument('--compulsive-blocks', type=int, default=0, help='Number of contiguous compulsive blocks to insert')
    parser.add_argument('--compulsive-block-size', type=int, default=30, help='Size (rows) of each compulsive block')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory')
    args = parser.parse_args()

    # Generate base data first
    out = generate_sample_data(num_records=args.num_records, compulsive_fraction=args.compulsive_fraction, output_dir=args.output_dir)

    # If blocks requested, post-process the CSV to create contiguous compulsive blocks
    if args.compulsive_blocks and args.compulsive_blocks > 0:
        import csv
        from random import randint

        csv_path = Path(out)
        df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        n = len(df)
        blocks = min(args.compulsive_blocks, max(1, n // args.compulsive_block_size))
        starts = []
        for _ in range(blocks):
            start = randint(0, max(0, n - args.compulsive_block_size - 1))
            starts.append(start)
            end = start + args.compulsive_block_size
            indices = range(start, min(end, n))
            for i in indices:
                # amplify
                df.at[i, 'scroll_speed'] = abs(df.at[i, 'scroll_speed'] * np.random.uniform(3.0, 6.0))
                df.at[i, 'swipe_count'] = int(df.at[i, 'swipe_count'] + np.random.poisson(lam=6))
                df.at[i, 'touches_per_minute'] = abs(df.at[i, 'touches_per_minute'] * np.random.uniform(2.0, 4.0))
                df.at[i, 'active_app'] = np.random.choice(['TikTok', 'YouTube'])

        df.to_csv(csv_path, index=False)
        print(f"Inserted {blocks} compulsive blocks (size={args.compulsive_block_size}) into {csv_path}")
