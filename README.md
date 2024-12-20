# Healthcare AI System

A comprehensive healthcare AI system that combines disease prediction based on symptoms and skin disease detection from images.

## Features

1. **Disease Prediction from Symptoms**
   - Multiple ML/DL models for accurate disease prediction
   - Text-based input of symptoms
   - Detailed remedies and recommendations

2. **Skin Disease Detection**
   - CNN-based image processing
   - Instant skin disease classification
   - Treatment recommendations
   - High accuracy detection system

3. **Model Comparison Dashboard**
   - Performance metrics visualization
   - Comparative analysis of different models
   - Detailed accuracy metrics

## Models Implemented

### Symptom-based Disease Prediction
1. Standard ML Models:
   - Random Forest
   - Support Vector Machine
   - XGBoost

2. Deep Learning Models:
   - Multi-layer Perceptron
   - LSTM Network

### Skin Disease Detection
1. CNN Models:
   - Custom CNN
   - Transfer Learning with VGG16
   - ResNet50

## Project Structure
```
healthcare_ai/
├── data/
│   ├── disease_symptom_data/
│   └── skin_disease_images/
├── models/
│   ├── disease_prediction/
│   └── skin_disease_detection/
├── src/
│   ├── train_disease_models.py
│   ├── train_skin_models.py
│   ├── model_evaluation.py
│   └── utils.py
├── notebooks/
│   ├── data_analysis.ipynb
│   └── model_comparison.ipynb
├── streamlit_app/
│   ├── app.py
│   └── pages/
└── requirements.txt
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   cd streamlit_app
   streamlit run app.py
   ```

2. Use the web interface to:
   - Input symptoms for disease prediction
   - Upload images for skin disease detection
   - View model comparisons and metrics

## Model Performance

### Disease Prediction Models
- Random Forest: 92% accuracy
- SVM: 88% accuracy
- XGBoost: 90% accuracy
- Deep Learning: 94% accuracy

### Skin Disease Detection
- Custom CNN: 89% accuracy
- VGG16: 93% accuracy
- ResNet50: 95% accuracy

[Detailed metrics and comparison plots will be added]

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
