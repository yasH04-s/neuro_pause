# Contents of /NeuroPause/NeuroPause/ML-Model/scripts/evaluate_model.py

"""
This file contains functions to evaluate the trained model's performance.

Expected input:
- trained_model: The model that has been trained and is ready for evaluation.
- test_data: The data on which the model will be evaluated, typically containing features and labels.

Expected output:
- evaluation_metrics: A dictionary containing metrics such as accuracy, precision, recall, etc., that summarize the model's performance.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(trained_model, test_data, test_labels):
    """
    Evaluates the performance of the trained model on the test data.

    Parameters:
    - trained_model: The trained machine learning model.
    - test_data: The features of the test dataset.
    - test_labels: The true labels of the test dataset.

    Returns:
    - evaluation_metrics: A dictionary containing evaluation metrics.
    """
    # Make predictions using the trained model
    predictions = trained_model.predict(test_data)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')

    # Compile metrics into a dictionary
    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return evaluation_metrics