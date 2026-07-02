# Multimodal Multi-Label Emotion Recognition

A deep learning-based multimodal emotion recognition system that analyzes **text**, **facial images**, and **speech audio** to identify human emotions. The application integrates state-of-the-art transformer models with machine learning classifiers to provide accurate emotion prediction through an interactive web interface.

---

## Overview

Emotion recognition plays a significant role in Human-Computer Interaction (HCI), affective computing, healthcare, education, and intelligent virtual assistants. Unlike unimodal systems, this project combines multiple information sources to improve emotion understanding by analyzing textual, visual, and speech-based cues.

The system enables users to perform emotion analysis using any individual modality or a combination of multiple modalities. It further compares the predicted emotions using Decision Tree and K-Nearest Neighbors (KNN) algorithms and presents the confidence scores through interactive visualizations.

---

## Key Features

- Multimodal Emotion Recognition using Text, Image, and Audio
- Independent or Combined Modality Analysis
- Transformer-based Deep Learning Models
- Decision Tree Classification
- K-Nearest Neighbor (KNN) Classification
- Confidence Score Visualization
- Automatic Prediction Logging
- Interactive Gradio Web Application
- CSV Export of Prediction History
- Professional User Interface

---

## System Architecture

```text
                   User Input
                        │
        ┌───────────────┼───────────────┐
        │               │               │
      Text           Image          Audio
        │               │               │
        ▼               ▼               ▼
 Transformer      Vision Transformer   Wav2Vec2
        │               │               │
        └───────────────┼───────────────┘
                        │
                Emotion Prediction
                        │
        ┌───────────────┼───────────────┐
        │                               │
 Decision Tree                    KNN Classifier
        │                               │
        └───────────────┼───────────────┘
                        │
             Confidence Visualization
                        │
              CSV Prediction Storage
```

---

## Technologies Used

### Programming Language

- Python 3.11

### Frameworks

- Gradio
- Hugging Face Transformers
- PyTorch
- Scikit-learn

### Libraries

- Pandas
- NumPy
- Matplotlib
- Pillow

---

## Deep Learning Models

| Modality | Model |
|----------|-------|
| Text Emotion Recognition | j-hartmann/emotion-english-distilroberta-base |
| Facial Emotion Recognition | trpakov/vit-face-expression |
| Speech Emotion Recognition | ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition |

---

## Machine Learning Models

- Decision Tree Classifier
- K-Nearest Neighbors (KNN)

---

## Project Structure

```text
Multimodal_Emotion_Recognition/
│
├── app.py
├── predict.py
├── model_loader.py
├── requirements.txt
├── README.md
├── emotion_results.csv
│
├── assets/
├── docs/
├── models/
├── outputs/
├── sample_data/
└── venv/
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/preeti33-codeer/Multimodal-Multilabel-Emotion-Recognition-.git
```

### Navigate to the Project Directory

```bash
cd Multimodal-Multilabel-Emotion-Recognition-
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Launch the Application

```bash
python app.py
```

---

## Application Workflow

1. Launch the Gradio application.
2. Select one or more input modalities.
3. Upload or provide the required input.
4. Execute emotion analysis.
5. View predicted emotions and confidence scores.
6. Compare predictions using Decision Tree and KNN.
7. Download prediction history as a CSV file.

---

## Output

The application generates:

- Emotion Label
- Confidence Score
- Decision Tree Prediction
- KNN Prediction
- Confidence Visualization
- Recent Prediction History
- CSV Export

---

## Applications

- Human–Computer Interaction
- Mental Health Monitoring
- Intelligent Virtual Assistants
- Educational Analytics
- Customer Sentiment Analysis
- Healthcare Support Systems
- Emotion-Aware AI Systems

---

## Future Enhancements

- Real-time Webcam Emotion Recognition
- Live Speech Processing
- Multilingual Emotion Detection
- Emotion Trend Dashboard
- PDF Report Generation
- Cloud Deployment
- User Authentication
- Database Integration

---

## Research Significance

This project demonstrates a multimodal approach to emotion recognition by integrating Natural Language Processing, Computer Vision, and Speech Processing within a unified framework. The combination of transformer-based deep learning models with traditional machine learning classifiers provides an effective decision-support mechanism for emotion analysis across multiple modalities.

---

## Author

**Preeti Katti**

Artificial Intelligence & Machine Learning Engineer

- GitHub: https://github.com/preeti33-codeer

---

## License

This project is intended for academic, educational, and research purposes.
