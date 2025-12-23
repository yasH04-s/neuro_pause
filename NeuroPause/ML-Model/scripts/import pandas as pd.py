import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Generate sample raw data for NeuroPause
def generate_sample_data(num_records=1000, output_dir='./data/raw/'):
    """
    Generate synthetic raw sensor data for testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp range
    start_time = datetime.now() - timedelta(days=7)
    timestamps = [start_time + timedelta(seconds=i*3) for i in range(num_records)]
    
    # Generate synthetic data
    data = {
        'timestamp': timestamps,
        'scroll_speed': np.random.normal(loc=0.5, scale=0.3, size=num_records),
        'swipe_count': np.random.poisson(lam=3, size=num_records),
        'touches_per_minute': np.random.normal(loc=20, scale=8, size=num_records),
        'session_duration': np.random.uniform(low=30, high=600, size=num_records),
        'screen_on_off': np.random.randint(0, 2, size=num_records),
        'active_app': np.random.choice(['Instagram', 'TikTok', 'Twitter', 'YouTube', 'News'], size=num_records)
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'neuropause_raw.csv')
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated sample data: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")


if __name__ == "__main__":
    generate_sample_data(num_records=2000)