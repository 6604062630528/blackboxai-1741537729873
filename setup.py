import os
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_requirements():
    """
    Check and install required packages
    """
    try:
        logger.info("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Required packages installed successfully")
    except Exception as e:
        logger.error(f"Error installing requirements: {str(e)}")
        raise

def setup_directories():
    """
    Create necessary directories
    """
    try:
        directories = ['data', 'models', 'uploads']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise

def download_datasets():
    """
    Download and prepare datasets
    """
    try:
        logger.info("Downloading datasets...")
        subprocess.check_call([sys.executable, "data/download_datasets.py"])
        logger.info("Datasets downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading datasets: {str(e)}")
        raise

def train_models():
    """
    Train both ML and NN models
    """
    try:
        # Train ML model
        logger.info("Training Machine Learning model...")
        subprocess.check_call([sys.executable, "models/train_ml_model.py"])
        logger.info("ML model trained successfully")

        # Train NN model
        logger.info("Training Neural Network model...")
        subprocess.check_call([sys.executable, "models/train_nn_model.py"])
        logger.info("NN model trained successfully")
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

def main():
    """
    Main setup function
    """
    try:
        logger.info("Starting setup process...")
        
        # Check and install requirements
        check_requirements()
        
        # Create necessary directories
        setup_directories()
        
        # Download datasets
        download_datasets()
        
        # Train models
        train_models()
        
        logger.info("""
Setup completed successfully!

To run the application:
1. Execute 'python app.py'
2. Open your web browser and navigate to http://localhost:5000

The application includes:
- Machine Learning model for heart disease prediction
- Neural Network model for fashion item classification
- Interactive web interface for both models
- Detailed theory pages explaining the implementation

Note: The models have been trained on sample datasets. For production use,
you may want to retrain them on more comprehensive datasets.
        """)
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        logger.error("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
