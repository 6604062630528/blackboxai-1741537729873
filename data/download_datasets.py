import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_heart_disease_data():
    """
    Download the Heart Disease dataset from UCI ML Repository
    """
    try:
        # URL for the Heart Disease dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        # Download the data
        logger.info("Downloading Heart Disease dataset...")
        response = requests.get(url)
        
        if response.status_code == 200:
            # Save raw data
            raw_data_path = os.path.join('data', 'heart_disease_raw.data')
            with open(raw_data_path, 'w') as f:
                f.write(response.text)
            
            # Process the data
            columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            
            data = pd.read_csv(raw_data_path, names=columns, na_values='?')
            
            # Clean the data
            data = data.dropna()  # Remove rows with missing values
            
            # Save as CSV
            csv_path = os.path.join('data', 'heart.csv')
            data.to_csv(csv_path, index=False)
            
            # Remove raw data file
            os.remove(raw_data_path)
            
            logger.info("Heart Disease dataset downloaded and processed successfully")
            
        else:
            logger.error("Failed to download Heart Disease dataset")
            
    except Exception as e:
        logger.error(f"Error downloading Heart Disease dataset: {str(e)}")
        raise

def download_fashion_mnist():
    """
    Download Fashion MNIST dataset using TensorFlow
    """
    try:
        logger.info("Downloading Fashion MNIST dataset...")
        # Load Fashion MNIST dataset
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        
        logger.info("Fashion MNIST dataset downloaded successfully")
        
        # Save some sample images for testing
        logger.info("Saving sample images...")
        sample_images_dir = os.path.join('data', 'sample_images')
        if not os.path.exists(sample_images_dir):
            os.makedirs(sample_images_dir)
            
        # Save 5 sample images from each class
        for class_idx in range(10):
            class_images = train_images[train_labels == class_idx][:5]
            for i, img in enumerate(class_images):
                filename = os.path.join(sample_images_dir, f'class_{class_idx}_sample_{i}.png')
                tf.keras.preprocessing.image.save_img(filename, img.reshape(28, 28, 1))
                
        logger.info("Sample images saved successfully")
        
    except Exception as e:
        logger.error(f"Error downloading Fashion MNIST dataset: {str(e)}")
        raise

def main():
    """
    Download and prepare both datasets
    """
    try:
        # Create data directory if it doesn't exist
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Download datasets
        download_heart_disease_data()
        download_fashion_mnist()
        
        logger.info("All datasets downloaded and prepared successfully")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
