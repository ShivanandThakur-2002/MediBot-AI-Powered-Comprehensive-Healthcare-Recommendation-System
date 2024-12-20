import os
import json
import zipfile
import pandas as pd
from pathlib import Path

def setup_kaggle_credentials():
    """
    Create kaggle.json file with API credentials.
    You need to get these credentials from kaggle.com/account
    """
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Note: Replace these with your actual Kaggle credentials
    kaggle_json = {
        "username": "amanpramodv",
        "key": "859bf399d722ed2a92d3bf02ecf64c95"
    }
    
    kaggle_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(kaggle_path, 'w') as f:
        json.dump(kaggle_json, f)
    
    # Set permissions
    try:
        os.chmod(kaggle_path, 0o600)
    except Exception as e:
        print(f"Warning: Could not set file permissions: {e}")

    print("Kaggle credentials set up successfully!")

# Set up Kaggle credentials before importing kaggle
setup_kaggle_credentials()

# Now import kaggle after credentials are set
import kaggle

def download_disease_dataset():
    """
    Download the Disease Symptom Prediction dataset
    """
    print("Downloading Disease Symptom dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data/disease_symptom_data', exist_ok=True)
    
    # Download dataset
    kaggle.api.dataset_download_files(
        'itachi9604/disease-symptom-description-dataset',
        path='data/disease_symptom_data',
        unzip=True
    )
    
    print("Disease dataset downloaded successfully!")

def download_skin_dataset():
    """
    Download the HAM10000 skin cancer dataset
    """
    print("Downloading HAM10000 skin disease dataset...")
    
    # Create directories
    os.makedirs('data/skin_disease_images/train', exist_ok=True)
    os.makedirs('data/skin_disease_images/validation', exist_ok=True)
    
    # Download dataset
    kaggle.api.dataset_download_files(
        'kmader/skin-cancer-mnist-ham10000',
        path='data/skin_disease_images',
        unzip=True
    )
    
    print("Skin disease dataset downloaded successfully!")

def prepare_disease_data():
    """
    Prepare the disease symptom dataset
    """
    # Read the datasets
    symptom_description = pd.read_csv('data/disease_symptom_data/symptom_Description.csv')
    symptom_precaution = pd.read_csv('data/disease_symptom_data/symptom_precaution.csv')
    symptom_severity = pd.read_csv('data/disease_symptom_data/Symptom-severity.csv')
    
    # Create a mapping of diseases to their descriptions and precautions
    disease_info = {}
    
    for _, row in symptom_description.iterrows():
        disease = row['Disease']
        description = row['Description']
        precautions = symptom_precaution[symptom_precaution['Disease'] == disease].iloc[0, 1:].tolist()
        
        disease_info[disease] = {
            'description': description,
            'precautions': [p for p in precautions if isinstance(p, str)]
        }
    
    # Save the processed data
    with open('data/disease_symptom_data/disease_info.json', 'w') as f:
        json.dump(disease_info, f, indent=4)

def prepare_skin_data():
    """
    Prepare the HAM10000 dataset
    Split into train and validation sets
    """
    import shutil
    from sklearn.model_selection import train_test_split
    
    # Read the metadata
    metadata = pd.read_csv('data/skin_disease_images/HAM10000_metadata.csv')
    
    # Create train and validation splits
    train_df, val_df = train_test_split(metadata, test_size=0.2, random_state=42)
    
    # Move images to appropriate directories
    for df, split in [(train_df, 'train'), (val_df, 'validation')]:
        for _, row in df.iterrows():
            image_id = row['image_id']
            dx = row['dx']  # diagnosis
            
            # Create class directory if it doesn't exist
            os.makedirs(f'data/skin_disease_images/{split}/{dx}', exist_ok=True)
            
            # Move image to appropriate directory
            src = f'data/skin_disease_images/HAM10000_images/{image_id}.jpg'
            dst = f'data/skin_disease_images/{split}/{dx}/{image_id}.jpg'
            
            if os.path.exists(src):
                shutil.copy(src, dst)

def main():
    try:
        # Download datasets
        download_disease_dataset()
        download_skin_dataset()
        
        # Prepare datasets
        prepare_disease_data()
        prepare_skin_data()
        
        print("Dataset preparation completed!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()