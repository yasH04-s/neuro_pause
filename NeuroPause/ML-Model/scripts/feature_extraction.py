# Contents of /NeuroPause/NeuroPause/ML-Model/scripts/feature_extraction.py

"""
feature_extraction.py

Extract behavioral features from sliding windows produced by preprocessing.

Input: `ML-Model/data/processed/X_processed.npy` (N, timesteps, features)
Output: `ML-Model/data/processed/features_extracted.npy` (N, 6)

Produces six features per window:
    - avg_scroll_speed
    - burst_scrolls
    - swipe_intensity
    - touch_variation
    - engagement_spike
    - compulsive_score

Each feature function expects a DataFrame for a single window with numeric columns
matching those used in preprocessing (e.g. 'scroll_speed', 'swipe_count', 'touches_per_minute', 'session_duration').
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import os


class FeatureExtractor:
    """Extracts features from a single window (DataFrame).

    Expected window columns: ['scroll_speed','swipe_count','touches_per_minute','session_duration']
    """
    def __init__(self, df_window: pd.DataFrame):
        self.window = df_window
        self.features = {}

    def extract_avg_scroll_speed(self):
        self.features['avg_scroll_speed'] = float(self.window['scroll_speed'].mean())
        return self.features['avg_scroll_speed']

    def extract_burst_scrolls(self):
        threshold = self.window['scroll_speed'].quantile(0.75)
        burst_count = (self.window['scroll_speed'] > threshold).sum()
        self.features['burst_scrolls'] = float(burst_count) / max(1, len(self.window))
        return self.features['burst_scrolls']

    def extract_swipe_intensity(self):
        avg_swipes = float(self.window['swipe_count'].mean())
        avg_touches = float(self.window['touches_per_minute'].mean())
        self.features['swipe_intensity'] = (abs(avg_swipes) + abs(avg_touches)) / 2.0
        return self.features['swipe_intensity']

    def extract_touch_variation(self):
        mean_touches = float(self.window['touches_per_minute'].mean())
        std_touches = float(self.window['touches_per_minute'].std())
        if mean_touches == 0:
            self.features['touch_variation'] = 0.0
        else:
            self.features['touch_variation'] = std_touches / abs(mean_touches)
        return self.features['touch_variation']

    def extract_engagement_spike(self):
        scroll_speed = self.window['scroll_speed'].values
        if len(scroll_speed) < 2:
            self.features['engagement_spike'] = 0.0
            return 0.0
        z_scores = np.abs(stats.zscore(scroll_speed))
        spike_ratio = float((z_scores > 2).sum()) / len(scroll_speed)
        self.features['engagement_spike'] = spike_ratio
        return self.features['engagement_spike']

    def extract_compulsive_score(self):
        # call other extractors (they cache values)
        avg_scroll = abs(self.extract_avg_scroll_speed())
        bursts = self.extract_burst_scrolls()
        intensity = self.extract_swipe_intensity()
        variation = self.extract_touch_variation()
        spikes = self.extract_engagement_spike()

        # weighted combination
        compulsive_score = (
            0.25 * avg_scroll + 0.20 * bursts + 0.20 * intensity + 0.20 * variation + 0.15 * spikes
        )
        self.features['compulsive_score'] = float(np.clip(compulsive_score, 0.0, 1.0))
        return self.features['compulsive_score']

    def extract_all(self):
        self.extract_avg_scroll_speed()
        self.extract_burst_scrolls()
        self.extract_swipe_intensity()
        self.extract_touch_variation()
        self.extract_engagement_spike()
        self.extract_compulsive_score()
        return self.features


def extract_features_batch(X: np.ndarray, column_names=None):
    """Given X (N, timesteps, features) extract 6-feature vector per window.

    Returns (N,6) numpy array and header list.
    """
    features = []
    if X.size == 0:
        return np.zeros((0, 6)), ['avg_scroll_speed','burst_scrolls','swipe_intensity','touch_variation','engagement_spike','compulsive_score']

    for i in range(X.shape[0]):
        dfw = pd.DataFrame(X[i], columns=column_names)
        fe = FeatureExtractor(dfw)
        d = fe.extract_all()
        row = [d['avg_scroll_speed'], d['burst_scrolls'], d['swipe_intensity'], d['touch_variation'], d['engagement_spike'], d['compulsive_score']]
        features.append(row)
    return np.array(features, dtype=np.float32), ['avg_scroll_speed','burst_scrolls','swipe_intensity','touch_variation','engagement_spike','compulsive_score']


def main(x_path=None, output_dir=None):
    script_dir = Path(__file__).parent.parent
    if x_path is None:
        x_path = script_dir / 'data' / 'processed' / 'X_processed.npy'
    if output_dir is None:
        output_dir = script_dir / 'data' / 'processed'

    x_path = str(x_path)
    output_dir = str(output_dir)

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Preprocessed data not found: {x_path}")

    X = np.load(x_path)
    # read feature column names if present
    feat_file = Path(output_dir) / 'feature_columns.txt'
    if feat_file.exists():
        with open(feat_file, 'r') as f:
            cols = [l.strip() for l in f.read().splitlines() if l.strip()]
    else:
        # fallback column names (must match preprocessing selection)
        cols = ['scroll_speed','swipe_count','touches_per_minute','session_duration']

    feature_matrix, headers = extract_features_batch(X, column_names=cols)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'features_extracted.npy')
    np.save(out_path, feature_matrix)
    print(f"Saved features to {out_path} (shape={feature_matrix.shape})")


if __name__ == '__main__':
    main()