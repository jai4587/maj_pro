import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    """Load and return an image"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(image, target_size=(224, 224)):
    """Resize image to target size"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def preprocess_image(image):
    """Basic preprocessing: resize and normalize"""
    processed = resize_image(image)
    normalized = processed.astype(np.float32) / 255.0
    return normalized

def generate_synthetic_image():
    """Generate a synthetic car damage image using simple transformations"""
    # For demo purposes, we'll use a base image and add synthetic damage
    image_path = "train/severe car damaged/0051.JPEG"
    if not os.path.exists(image_path):
        # Try another path
        image_path = next(os.path.join(root, file) 
                           for root, _, files in os.walk("train") 
                           for file in files 
                           if file.endswith(('.jpg', '.jpeg', '.JPEG', '.png')))
    
    base_img = load_image(image_path)
    
    # Simple transformation to simulate GAN output
    synthetic = base_img.copy()
    
    # Add some random damage pattern (red spots)
    for _ in range(20):
        x = np.random.randint(0, synthetic.shape[1])
        y = np.random.randint(0, synthetic.shape[0])
        cv2.circle(synthetic, (x, y), np.random.randint(5, 15), (255, 0, 0), -1)
    
    # Add some scratches
    for _ in range(5):
        x1 = np.random.randint(0, synthetic.shape[1])
        y1 = np.random.randint(0, synthetic.shape[0])
        x2 = np.random.randint(0, synthetic.shape[1])
        y2 = np.random.randint(0, synthetic.shape[0])
        cv2.line(synthetic, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    return synthetic

def predict_insurance_amount(features):
    """Predict insurance amount from image features"""
    # For demo purposes, calculate a simple score based on image features
    damage_score = np.mean(features) * 10000  # Simple scaling
    
    # Add some variability for realism
    base_amount = 1500 + damage_score
    final_amount = base_amount + np.random.normal(0, 500)
    
    return max(500, final_amount)  # Ensure minimum payout

def extract_simple_features(image):
    """Extract basic features from image for demo"""
    # Simplified feature extraction - in a real system this would use CNN
    # Split into color channels
    b, g, r = cv2.split(image)
    
    # Calculate basic statistics as features
    features = [
        np.mean(r), np.std(r),  # Red channel stats (could indicate damage)
        np.mean(g), np.std(g),  # Green channel stats
        np.mean(b), np.std(b),  # Blue channel stats
    ]
    
    return np.array(features)

def plot_images(images, titles):
    """Plot a list of images with titles"""
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Insurance Amount Prediction Demo")
    print("================================")
    
    # Step 1: Load a sample car damage image
    print("\n1. Loading sample car damage image...")
    image_path = "train/severe car damaged/0051.JPEG"
    if not os.path.exists(image_path):
        # Try to find any image in the train directory
        for root, _, files in os.walk("train"):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.JPEG', '.png')):
                    image_path = os.path.join(root, file)
                    break
            if os.path.exists(image_path):
                break
    
    original_image = load_image(image_path)
    print(f"Loaded image from: {image_path}")
    
    # Step 2: Preprocess the image
    print("\n2. Preprocessing the image...")
    processed_image = preprocess_image(original_image)
    
    # Step 3: Generate synthetic image (simulating GAN)
    print("\n3. Generating synthetic car damage image using GAN simulation...")
    synthetic_image = generate_synthetic_image()
    
    # Step 4: Feature extraction
    print("\n4. Extracting features from the processed image...")
    features = extract_simple_features(processed_image)
    
    # Step 5: Predict insurance amount
    print("\n5. Predicting insurance amount based on extracted features...")
    prediction = predict_insurance_amount(features)
    
    # Output results
    print("\nResults:")
    print(f"Predicted insurance amount: ${prediction:.2f}")
    
    # Display images
    print("\nDisplaying images (close window to continue)...")
    plot_images(
        [original_image, processed_image, synthetic_image],
        ['Original Car Damage Image', 'Processed Image', 'Synthetic GAN Image']
    )
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 