Voice Analysis to Detect Emotional Disorder
📌 Project Overview

This project applies Machine Learning (XGBoost, Random Forest) and Deep Learning (LSTM) models to analyze synthetic voice features (pitch, jitter, shimmer, MFCCs, formants, speech rate) and classify individuals as Healthy or having an Emotional Disorder.

It demonstrates how AI-driven voice biomarkers can support mental health monitoring and early detection of emotional disorders.

📊 Dataset

Synthetic dataset with 500 samples

Features include:

Pitch (Hz)

Energy (dB)

Jitter, Shimmer

MFCC1, MFCC2, MFCC3

Formants (Hz)

Speech rate (WPM)

Target label:

0 → Healthy

1 → Emotional Disorder

📂 File: synthetic_voice_data.csv

🛠️ Models Used

Random Forest Classifier – baseline ensemble learning

XGBoost Classifier – gradient boosting for improved accuracy

LSTM (Long Short-Term Memory) – deep learning for sequential voice data

🚀 Installation
# Clone repository
git clone https://github.com/your-username/Voice-Analysis-to-Detect-Emotional-Disorder.git
cd Voice-Analysis-to-Detect-Emotional-Disorder

# Install dependencies
pip install -r requirements.txt

▶️ Usage

Run the Python script:

python voice_emotion_disorder_prediction.py

📈 Output

Accuracy and classification reports for Random Forest, XGBoost, and LSTM

Confusion matrix heatmap for Random Forest

LSTM model evaluation on synthetic data

📌 Future Work

Collect and analyze real-world clinical datasets

Incorporate prosody features and spectrogram analysis

Build a real-time detection system

👤 Author

Your Name: Okes Imoni

🌐 GitHub: @Okes2025
