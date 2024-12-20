import os
import json

def setup_kaggle():
    """Setup Kaggle configuration"""
    # Create .kaggle directory in user's home directory
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Create kaggle.json with credentials
    kaggle_cred = {
        "username": "amanpramodv",
        "key": "859bf399d722ed2a92d3bf02ecf64c95"
    }
    
    # Save credentials to kaggle.json
    kaggle_path = os.path.join(kaggle_dir, 'kaggle.json')
    with open(kaggle_path, 'w') as f:
        json.dump(kaggle_cred, f)
    
    # Set appropriate permissions
    try:
        os.chmod(kaggle_path, 0o600)
    except Exception as e:
        print(f"Warning: Could not set file permissions: {e}")
    
    print("Kaggle configuration has been set up successfully!")

if __name__ == "__main__":
    setup_kaggle()
