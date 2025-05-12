import os
import numpy as np
from image_input import ImageInput
from preprocessing import ImagePreprocessor
from gan_generator import GANGenerator
from cnn_features import CNNFeatureExtractor
from regression_model import InsurancePredictor

class InsurancePredictionPipeline:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.image_input = ImageInput()
        self.preprocessor = ImagePreprocessor()
        self.gan = GANGenerator()
        self.feature_extractor = CNNFeatureExtractor()
        self.predictor = InsurancePredictor(input_dim=25088)  # VGG16 feature dimension
    
    def train_pipeline(self, train_images, train_labels, val_images=None, val_labels=None):
        """Train the complete pipeline"""
        # Preprocess training images
        processed_images = self.preprocessor.batch_preprocess(train_images)
        
        # Generate synthetic images using GAN
        synthetic_images = self.gan.generate_images(n_samples=len(train_images))
        processed_synthetic = self.preprocessor.batch_preprocess(synthetic_images)
        
        # Combine real and synthetic images
        combined_images = np.vstack([processed_images, processed_synthetic])
        combined_labels = np.concatenate([train_labels, train_labels])  # Use same labels for synthetic
        
        # Extract features
        features = self.feature_extractor.batch_extract_features(combined_images)
        
        # Train regression model
        history = self.predictor.train(
            features, combined_labels,
            X_val=None if val_images is None else self.feature_extractor.batch_extract_features(
                self.preprocessor.batch_preprocess(val_images)
            ),
            y_val=val_labels
        )
        
        return history
    
    def predict_insurance(self, image_path):
        """Predict insurance amount for a single image"""
        # Load and preprocess image
        image = self.image_input.load_image(image_path)
        processed_image = self.preprocessor.preprocess_pipeline(image)
        
        # Extract features
        features = self.feature_extractor.extract_features(processed_image)
        
        # Make prediction
        prediction = self.predictor.predict(features)
        
        return float(prediction[0][0])
    
    def save_models(self):
        """Save all models"""
        self.gan.save_models(self.model_dir)
        self.feature_extractor.save_model(self.model_dir)
        self.predictor.save_model(self.model_dir)
    
    def load_models(self):
        """Load all models"""
        self.gan.load_models(self.model_dir)
        self.feature_extractor.load_model(self.model_dir)
        self.predictor.load_model(self.model_dir)

def main():
    # Example usage
    pipeline = InsurancePredictionPipeline()
    
    # Train pipeline (if you have training data)
    # train_images = ...  # Load your training images
    # train_labels = ...  # Load your training labels
    # pipeline.train_pipeline(train_images, train_labels)
    
    # Save models after training
    # pipeline.save_models()
    
    # Load pre-trained models
    pipeline.load_models()
    
    # Make prediction on a new image
    image_path = "path/to/your/car/damage/image.jpg"
    prediction = pipeline.predict_insurance(image_path)
    print(f"Predicted insurance amount: ${prediction:.2f}")

if __name__ == "__main__":
    main() 