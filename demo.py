import os
import numpy as np
import matplotlib.pyplot as plt
from image_input import ImageInput
from preprocessing import ImagePreprocessor
from gan_generator import GANGenerator
from cnn_features import CNNFeatureExtractor
from regression_model import InsurancePredictor

def plot_images(images, titles, rows=1):
    """Plot a list of images with titles"""
    fig, axes = plt.subplots(rows, len(images), figsize=(15, 5*rows))
    if rows == 1:
        axes = [axes]
    
    for row in range(rows):
        for col, (img, title) in enumerate(zip(images[row], titles)):
            axes[row][col].imshow(img)
            axes[row][col].set_title(title)
            axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    # Initialize components
    image_input = ImageInput()
    preprocessor = ImagePreprocessor()
    gan = GANGenerator()
    feature_extractor = CNNFeatureExtractor()
    predictor = InsurancePredictor(input_dim=25088)
    
    print("1. Loading and preprocessing a sample image...")
    # Load a sample image from our dataset
    sample_image_path = "train/severe car damaged/0051.JPEG"
    if not os.path.exists(sample_image_path):
        print(f"Please place a sample image at {sample_image_path}")
        return
    
    # Load and preprocess image
    original_image = image_input.load_image(sample_image_path)
    processed_image = preprocessor.preprocess_pipeline(original_image)
    
    print("2. Generating synthetic image using GAN...")
    # Generate synthetic image
    synthetic_image = gan.generate_images(n_samples=1)[0]
    
    print("3. Extracting features and making prediction...")
    # Extract features and make prediction
    features = feature_extractor.extract_features(processed_image)
    prediction = predictor.predict(features)
    
    # Display results
    print("\nResults:")
    print(f"Predicted insurance amount: ${float(prediction[0][0]):.2f}")
    
    # Plot images
    print("\nDisplaying images...")
    plot_images(
        [[original_image, processed_image, synthetic_image]],
        ['Original Image', 'Processed Image', 'Synthetic Image']
    )

if __name__ == "__main__":
    main() 