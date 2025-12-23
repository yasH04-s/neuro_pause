# Contents of /NeuroPause/NeuroPause/ML-Model/scripts/train_model.py

"""
train_model.py

Train an LSTM model on time-series windows produced by preprocessing.

Inputs (expected in `ML-Model/data/processed/`):
    - `X_processed.npy` shape (N, timesteps, features)
    - `y_labels.npy` shape (N,)

Outputs:
    - saved Keras model `model/saved/neuro_model.h5` (and best model `best_model.h5`)
    - evaluation metrics CSV

The script uses EarlyStopping and reports Accuracy, Precision, Recall, F1.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import csv


def _import_tf():
    try:
        import tensorflow as tf
        from tensorflow import keras
        return tf, keras
    except Exception as e:
        raise ImportError(
            "TensorFlow is required to run training. Install it with: `pip install tensorflow`\n"
            f"Original error: {e}"
        )


def load_preprocessed_data(data_dir: str):
    x_path = Path(data_dir) / 'X_processed.npy'
    y_path = Path(data_dir) / 'y_labels.npy'
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Preprocessed files not found in {data_dir}. Expected X_processed.npy and y_labels.npy")
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"Loaded X shape={X.shape}, y shape={y.shape}")
    return X, y


def reshape_for_lstm(X: np.ndarray):
    # ensure shape (samples, timesteps, features)
    if X.ndim == 2:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X


def train_test_split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Split X, y into train/val/test.

    If any class in `y` has fewer than 2 samples, stratified splitting is not possible
    (scikit-learn requires at least 2 members per class). In that case we fall back
    to a non-stratified split and emit a warning.
    """
    import numpy as _np

    # Check class counts
    try:
        unique, counts = _np.unique(y, return_counts=True)
    except Exception:
        # If y is not a simple array, coerce
        import pandas as _pd
        counts = _pd.Series(y).value_counts().values

    min_count = int(_np.min(counts))
    use_stratify = min_count >= 2

    if not use_stratify:
        print(f"Warning: class imbalance detected: counts={dict(zip(unique, counts))}.\n"
              "Falling back to non-stratified train/test split.")
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative, random_state=random_state)
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=random_state)

    print(f"Train/Val/Test sizes: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_lstm_model(input_shape, lstm_units=64, dropout_rate=0.3):
    tf, keras = _import_tf()
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.LSTM(lstm_units, activation='tanh', return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout_rate),
        layers.LSTM(max(8, lstm_units // 2), activation='tanh'),
        layers.Dropout(dropout_rate),
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_and_save(model, X_train, X_val, X_test, y_train, y_val, y_test, model_save_path='./model/saved', epochs=100, batch_size=32):
    tf, keras = _import_tf()
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # If a file exists at the intended model_save_path, rename it to avoid WinError 183
    if os.path.exists(model_save_path) and not os.path.isdir(model_save_path):
        backup = model_save_path + '.bak'
        print(f"Path {model_save_path} exists and is a file. Renaming to {backup} to proceed.")
        os.rename(model_save_path, backup)
    os.makedirs(model_save_path, exist_ok=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(os.path.join(model_save_path, 'best_model.h5'), monitor='val_accuracy', save_best_only=True, verbose=1)

    # Compute class weights to mitigate class imbalance
    try:
        from sklearn.utils import class_weight
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        print(f"Using class_weight: {class_weight_dict}")
    except Exception:
        class_weight_dict = None

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save final model
    final_path = os.path.join(model_save_path, 'neuro_model.h5')
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    # Evaluate
    # Get prediction probabilities for validation and test sets
    y_val_probs = model.predict(X_val).flatten()
    y_test_probs = model.predict(X_test).flatten()

    # Find best threshold on validation set by F1
    best_t = 0.5
    best_f1 = -1.0
    from sklearn.metrics import f1_score as _f1
    for t in [i / 100 for i in range(10, 91)]:
        yv = (y_val_probs > t).astype(int)
        f = _f1(y_val, yv, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t

    print(f"Best threshold on val set by F1: {best_t} (F1={best_f1:.4f})")

    # Apply threshold to test set
    y_test_pred = (y_test_probs > best_t).astype(int)

    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print(f"Test Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame({'metric': ['accuracy','precision','recall','f1','best_threshold'],'value':[acc,prec,rec,f1,best_t]})
    metrics_df.to_csv(os.path.join(model_save_path, 'evaluation_metrics.csv'), index=False)
    print(f"Saved metrics to {model_save_path}/evaluation_metrics.csv")

    # Save per-sample predictions
    preds_out = os.path.join(model_save_path, 'test_predictions.csv')
    with open(preds_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['y_true', 'y_prob', 'y_pred'])
        for yt, p in zip(y_test.tolist(), y_test_probs.tolist()):
            writer.writerow([int(yt), float(p), int(p > best_t)])
    print(f"Saved per-sample predictions to {preds_out}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    cm_path = os.path.join(model_save_path, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")

    return history, y_test_pred, y_test


def main(data_dir=None, model_save_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    if model_save_dir is None:
        model_save_dir = Path(__file__).parent.parent / 'model' / 'saved'

    data_dir = str(data_dir)
    model_save_dir = str(model_save_dir)

    X, y = load_preprocessed_data(data_dir)
    X = reshape_for_lstm(X)
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_data(X, y)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    history, y_pred, y_test = train_and_save(model, X_train, X_val, X_test, y_train, y_val, y_test, model_save_path=model_save_dir)


if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print(str(e))