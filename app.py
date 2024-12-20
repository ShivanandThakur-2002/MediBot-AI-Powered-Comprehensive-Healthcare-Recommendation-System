import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
from pathlib import Path
import json
from inference_sdk import InferenceHTTPClient
import io
import tensorflow as tf
from src.setup.disease_predictor import predict_disease

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="kvzIPetDwZ00waP5DZKw"
)

# Load disease prediction models
def load_disease_models():
    models = {}
    model_dir = Path('D:\External_Projects\shivanand_mca\models\skin_disease-mobile_net.h5')
    
    # Load traditional ML models
    for model_path in model_dir.glob('*.joblib'):
        model_name = model_path.stem.replace('_model', '')
        models[model_name] = joblib.load(model_path)

    
    return models

# Load skin disease detection model
def load_skin_model():
    model_path = 'D:/External_Projects/shivanand_mca/models/skin_disease-mobile_net.h5'
    try:
        # Try loading with custom_objects to handle version compatibility
        return tf.keras.models.load_model(model_path, compile=False, custom_objects={
            'VarianceScaling': tf.keras.initializers.VarianceScaling,
            'Conv2D': tf.keras.layers.Conv2D
        })
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you have the correct TensorFlow version installed. Try: pip install tensorflow==2.12.0")
        return None

# Load disease remedies
def load_remedies():
    with open('D:/External_Projects/shivanand_mca/data/disease_remedies.json', 'r') as f:
        return json.load(f)

def preprocess_symptoms(symptoms_text):
    # Convert text symptoms to feature vector
    # This is a placeholder - implement based on your actual features
    return np.zeros((1, 100))  # Adjust size based on your feature space

def preprocess_image(image):
    # Resize the image to MobileNet input size
    img = image.resize((224, 224))
    # Convert to array and preprocess
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("Healthcare AI System")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Disease Prediction", "Skin Disease Detection", "Model Performance"]
    )
    
    if page == "Home":
        st.write("""
        # Welcome to Healthcare AI System
        
        This system provides two main features:
        1. Disease prediction based on symptoms
        2. Skin disease detection from images
        
        Use the sidebar to navigate between different features.
        """)
        
        st.image("D:/External_Projects/shivanand_mca/streamlit_app/assets/healthcare_ai.jpg")
        
    elif page == "Disease Prediction":
        st.header("Disease Prediction from Symptoms")
        
        # Add description
        st.write("""
        Please describe your symptoms in detail, and our AI system will analyze them 
        to suggest potential diseases and recommend remedies.
        """)
        
        symptoms = st.text_area(
            "Describe your symptoms:",
            placeholder="Example: I have been experiencing fever, headache, and fatigue for the past 2 days...",
            height=150
        )
        
        if st.button("Analyze Symptoms"):
            if symptoms and symptoms.strip():
                with st.spinner("Analyzing your symptoms..."):
                    try:
                        prediction = predict_disease(symptoms)
                        st.subheader("Analysis Results:")
                        st.write(prediction)
                    except Exception as e:
                        st.error(f"Error analyzing symptoms: {str(e)}")
                        st.info("Please try again or contact support if the problem persists.")
            else:
                st.warning("Please enter your symptoms before analyzing.")
                
    elif page == "Skin Disease Detection":
        st.header("Skin Disease Detection")
        
        # Add description
        st.write("""
        Upload an image of the skin condition you want to analyze. 
        The AI model will analyze the image and provide a prediction using advanced AI model.
        """)
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing the image..."):
                    try:
                        # Save the image to a temporary file
                        temp_path = "temp_image.jpg"
                        image.save(temp_path)
                        
                        # Get prediction from Roboflow using file path directly
                        result = CLIENT.infer(temp_path, model_id="kulit-kanker/3")
                        
                        # Display results
                        st.subheader("Analysis Results:")
                        
                        if 'predictions' in result:
                            predictions = result['predictions']
                            if predictions:
                                # Sort predictions by confidence
                                predictions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                                
                                for pred in predictions:
                                    confidence = pred.get('confidence', 0) * 100
                                    class_name = pred.get('class', 'Unknown')
                                    
                                    st.write(f"Class: {class_name}")
                                    st.write(f"Confidence: {confidence:.2f}%")
                                    st.progress(float(confidence) / 100)
                                    st.write("---")
                            else:
                                st.info("No skin conditions detected in the image.")
                        else:
                            st.warning("Unexpected response format from the model.")
                            
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
                        st.info("Please try again with a different image or check your internet connection.")
                        
                    finally:
                        # Clean up temporary file
                        import os
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

    elif page == "Model Performance":
        st.header("Model Performance Metrics")
        
        st.subheader("Disease Prediction Models")
        st.image("D:/External_Projects/shivanand_mca/streamlit_app/assets/accuracy_comparison.png")
        
        st.subheader("Skin Disease Detection Models")
        st.image("D:/External_Projects/shivanand_mca/streamlit_app/assets/skin_accuracy_comparison.png")
        st.image("D:/External_Projects/shivanand_mca/streamlit_app/assets/skin_loss_comparison.png")
        
        # Show confusion matrices
        st.subheader("Confusion Matrices")
        model_types = ["random_forest", "svm", "xgboost", "neural_net"]
        
        for model_type in model_types:
            st.write(f"{model_type} Confusion Matrix")
            st.image(f"D:/External_Projects/shivanand_mca/streamlit_app/assets/{model_type}_confusion_matrix.png")

if __name__ == "__main__":
    main()
