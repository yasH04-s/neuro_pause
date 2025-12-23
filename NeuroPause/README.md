# NeuroPause: Real-Time AI-Based Nudge System

## Overview
NeuroPause is a real-time AI-based nudge system designed to enhance user engagement and promote healthier behaviors through personalized nudges. This project leverages machine learning and reinforcement learning techniques to analyze user interactions and provide timely nudges based on individual profiles.

## Project Structure
The project is organized into two main components: **ML-Model** and **Personalization**.

### ML-Model
- **data/**: Contains raw and processed data files.
  - **raw/**: Directory for raw data files used for training the model.
  - **processed/**: Directory for processed data files ready for model training.
- **model/**: Stores trained models.
  - **saved/**: Directory for saved model files after training.
  - **tflite/**: Directory for TensorFlow Lite model files for deployment.
- **scripts/**: Contains scripts for various stages of the ML pipeline.
  - **preprocessing.py**: Functions for data preprocessing.
  - **feature_extraction.py**: Functions for feature extraction from processed data.
  - **train_model.py**: Code to train the machine learning model.
  - **evaluate_model.py**: Functions to evaluate the trained model's performance.
  - **convert_to_tflite.py**: Code to convert the trained model to TensorFlow Lite format.
- **utils/**: Utility functions and configuration settings.
  - **helpers.py**: Utility functions for use across different scripts.
  - **config.py**: Configuration settings for the project.
- **requirements.txt**: Lists required Python packages for the project.

### Personalization
- **rl_policy.py**: Implements the reinforcement learning policy for the nudge system.
- **age_profile.py**: Defines user age profiles and associated nudge strategies.
- **nudge_selector.py**: Logic to select appropriate nudges based on user profiles and interactions.
- **logs/**: Contains logs of user interactions and rewards.
  - **interactions.csv**: Logs of user interactions with the nudge system.
  - **rewards.csv**: Logs of rewards received by users based on their interactions.
- **__init__.py**: Marks the Personalization directory as a Python package.

## Setup Instructions
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages using the following command:
   ```
   pip install -r ML-Model/requirements.txt
   ```
4. Prepare your raw data files and place them in the `ML-Model/data/raw/` directory.
5. Run the preprocessing script to generate processed data:
   ```
   python ML-Model/scripts/preprocessing.py
   ```
6. Follow the subsequent scripts in the `scripts/` directory to train and evaluate your model.

## Contribution
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.