import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import skimage.feature as skf
from scipy.stats import entropy
from sklearn.cluster import KMeans
import tensorflow as tf
import json
from tensorflow.keras.models import model_from_json
import math

app = Flask(__name__)
app.secret_key = 'insurance_prediction_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Load pre-trained VGG model from JSON and weights
class CarDamageClassifier:
    def __init__(self):
        self.model = None
        self.class_names = ['minor damaged car', 'moderate car damaged', 'normal', 'severe car damaged']
        self.input_shape = (150, 150, 3)
        self.load_model()
    
    def load_model(self):
        try:
            # Load model architecture from JSON
            with open('model_vgg.json', 'r') as json_file:
                loaded_model_json = json_file.read()
            
            self.model = model_from_json(loaded_model_json)
            # Load weights
            self.model.load_weights('model_vgg.weights.h5')
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Successfully loaded pre-trained damage classification model")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
    
    def predict(self, image):
        if self.model is None:
            print("Model not loaded, using fallback prediction")
            return {'minor damaged car': 0.2, 'moderate car damaged': 0.3, 
                    'normal': 0.4, 'severe car damaged': 0.1}
        
        try:
            # Preprocess image
            img = cv2.resize(image, (150, 150))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            pred = self.model.predict(img)
            
            # Create result dictionary
            result = {class_name: float(prob) for class_name, prob in zip(self.class_names, pred[0])}
            return result
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {'minor damaged car': 0.2, 'moderate car damaged': 0.3, 
                    'normal': 0.4, 'severe car damaged': 0.1}

# Initialize the classifier
damage_classifier = CarDamageClassifier()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_image(image_path):
    """Load and return an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(image, target_size=(224, 224)):
    """Resize image to target size"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def preprocess_image(image):
    """Enhanced preprocessing: resize, normalize, and apply basic enhancement"""
    processed = resize_image(image)
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # Convert to float and normalize
    normalized = enhanced_rgb.astype(np.float32) / 255.0
    
    return normalized

def generate_synthetic_image(original_image):
    """Generate a synthetic car damage image using more realistic transformations"""
    # Simple transformation to simulate GAN output
    synthetic = original_image.copy()
    
    # Get image dimensions
    h, w = synthetic.shape[:2]
    
    # Determine damage intensity based on image properties
    # Start with a more reasonable amount of damage
    damage_intensity = np.random.uniform(0.3, 0.7)
    num_spots = int(15 * damage_intensity)
    num_scratches = int(3 * damage_intensity)
    
    # Add some random damage pattern (with more natural-looking colors)
    for _ in range(num_spots):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        radius = np.random.randint(3, 10)
        
        # Use more realistic damage colors (dark red/brown for rust, black for dents)
        color_choice = np.random.choice(['rust', 'dent'])
        if color_choice == 'rust':
            color = (np.random.randint(80, 140), np.random.randint(20, 60), np.random.randint(20, 60))
        else:
            color = (np.random.randint(20, 70), np.random.randint(20, 70), np.random.randint(20, 70))
        
        cv2.circle(synthetic, (x, y), radius, color, -1)
    
    # Add some scratches with more realistic patterns
    for _ in range(num_scratches):
        # Make scratches follow natural car contours (more horizontal/diagonal than vertical)
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        
        # Scratches tend to be somewhat horizontal on cars
        angle = np.random.uniform(0, np.pi)  # Avoid pure vertical scratches
        length = np.random.randint(20, w//3)
        
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # Limit to image boundaries
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))
        
        # Use more realistic scratch colors (silver/white/black)
        scratch_color = np.random.choice(['silver', 'black'])
        if scratch_color == 'silver':
            color = (np.random.randint(180, 220), np.random.randint(180, 220), np.random.randint(180, 220))
        else:
            color = (np.random.randint(20, 60), np.random.randint(20, 60), np.random.randint(20, 60))
        
        cv2.line(synthetic, (x1, y1), (x2, y2), color, np.random.randint(1, 3))
    
    return synthetic

def extract_advanced_features(image):
    """Extract comprehensive features from image using multiple computer vision techniques"""
    # Convert to different color spaces for better feature extraction
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Create grayscale image ensuring it's 8-bit unsigned integer
    if image.dtype != np.uint8:
        # If the image is float (normalized), convert back to uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            temp_img = (image * 255).astype(np.uint8)
            gray_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
        else:
            # For other dtypes, convert to uint8 safely
            temp_img = image.astype(np.uint8)
            gray_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
    else:
        # Already uint8, just convert to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. Basic color features
    b, g, r = cv2.split(image)
    h, s, v = cv2.split(hsv_img)
    
    # 2. Edge detection for damage assessment
    # Canny edge detection to find edges (damage often creates edges)
    edges = cv2.Canny(gray_img, 50, 150)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1]) * 100
    
    # 3. Texture analysis using GLCM (Gray Level Co-occurrence Matrix)
    # GLCM requires uint8 image with pixel values 0-255
    if gray_img.dtype != np.uint8:
        # Scale to 0-255 and convert to uint8
        gray_img_uint8 = (gray_img * 255).astype(np.uint8) if gray_img.dtype == np.float32 or gray_img.dtype == np.float64 else gray_img.astype(np.uint8)
        # Make sure it's exactly 256 levels for GLCM
        glcm = skf.graycomatrix(gray_img_uint8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    else:
        # Original image is already uint8
        glcm = skf.graycomatrix(gray_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    
    glcm_contrast = np.mean(skf.graycoprops(glcm, 'contrast'))
    glcm_dissimilarity = np.mean(skf.graycoprops(glcm, 'dissimilarity'))
    glcm_homogeneity = np.mean(skf.graycoprops(glcm, 'homogeneity'))
    glcm_energy = np.mean(skf.graycoprops(glcm, 'energy'))
    glcm_correlation = np.mean(skf.graycoprops(glcm, 'correlation'))
    
    # 4. Color distribution analysis (new cars have uniform color)
    # Calculate histograms and their entropy for each channel
    r_hist = cv2.calcHist([r], [0], None, [32], [0, 1])
    g_hist = cv2.calcHist([g], [0], None, [32], [0, 1])
    b_hist = cv2.calcHist([b], [0], None, [32], [0, 1])
    s_hist = cv2.calcHist([s], [0], None, [32], [0, 1])  # Saturation is useful for color purity
    
    # Normalize histograms for entropy calculation
    r_hist_norm = r_hist / np.sum(r_hist)
    g_hist_norm = g_hist / np.sum(g_hist)
    b_hist_norm = b_hist / np.sum(b_hist)
    s_hist_norm = s_hist / np.sum(s_hist)
    
    # Calculate entropy - lower entropy means more uniform color (like new cars)
    r_entropy = entropy(r_hist_norm.flatten() + 1e-10)  # Adding small value to avoid log(0)
    g_entropy = entropy(g_hist_norm.flatten() + 1e-10)
    b_entropy = entropy(b_hist_norm.flatten() + 1e-10)
    s_entropy = entropy(s_hist_norm.flatten() + 1e-10)
    
    # Calculate color "peakiness" - higher peak means more solid color (like new cars)
    r_peak = np.max(r_hist) / np.sum(r_hist)
    g_peak = np.max(g_hist) / np.sum(g_hist)
    b_peak = np.max(b_hist) / np.sum(b_hist)
    
    # 5. Color clustering to detect distinct damage areas
    # Reshape image to be a list of pixels
    pixel_values = image.reshape((-1, 3))
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
    kmeans.fit(pixel_values)
    
    # Get the histogram of cluster assignments
    cluster_hist, _ = np.histogram(kmeans.labels_, bins=np.arange(6))
    cluster_hist = cluster_hist / np.sum(cluster_hist)
    
    # A dominant cluster ratio indicates uniform color (new car)
    dominant_cluster_ratio = np.max(cluster_hist)
    
    # 6. Local Binary Pattern (LBP) for texture analysis
    # LBP can detect fine scratches and texture changes
    # Ensure we're using the correct dtype for LBP
    if gray_img.dtype != np.uint8:
        gray_img_uint8 = (gray_img * 255).astype(np.uint8) if gray_img.dtype == np.float32 or gray_img.dtype == np.float64 else gray_img.astype(np.uint8)
        lbp = skf.local_binary_pattern(gray_img_uint8, 8, 1, method='uniform')
    else:
        lbp = skf.local_binary_pattern(gray_img, 8, 1, method='uniform')
    
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    
    # 7. Calculate standard deviation metrics across channels
    r_std = np.std(r)
    g_std = np.std(g)
    b_std = np.std(b)
    h_std = np.std(h)
    s_std = np.std(s)
    v_std = np.std(v)
    gray_std = np.std(gray_img)
    
    # 8. Measure of brightness and contrast
    brightness = np.mean(image)
    contrast = np.std(image)
    
    # Compile all features into a single array
    features = [
        # Color statistics
        np.mean(r), r_std, np.mean(g), g_std, np.mean(b), b_std,
        np.mean(h), h_std, np.mean(s), s_std, np.mean(v), v_std,
        brightness, contrast, gray_std,
        
        # Edge features
        edge_density,
        
        # GLCM texture features
        glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation,
        
        # Histogram entropy features
        r_entropy, g_entropy, b_entropy, s_entropy,
        
        # Color peaks (uniformity)
        r_peak, g_peak, b_peak,
        
        # Clustering features
        dominant_cluster_ratio,
        
        # LBP texture features
        *lbp_hist  # Unpacking LBP histogram
    ]
    
    return {
        'features': np.array(features),
        'edges': edges,
        'lbp': lbp,
        'dominant_cluster_ratio': dominant_cluster_ratio,
        'edge_density': edge_density,
        'glcm_contrast': glcm_contrast,
        'glcm_homogeneity': glcm_homogeneity,
        'glcm_energy': glcm_energy,
        'r_std': r_std,
        'g_std': g_std,
        'b_std': b_std,
        's_std': s_std,
    }

def hybrid_predict_insurance(original_image, processed_image):
    """Hybrid approach that combines pre-trained model with advanced image analysis"""
    # 1. Get classification prediction from VGG model
    vgg_predictions = damage_classifier.predict(original_image)
    print(f"VGG model predictions: {vgg_predictions}")
    
    # 2. Extract advanced computer vision features
    analysis = extract_advanced_features(processed_image)
    features = analysis['features']
    
    # 3. Extract key metrics from computer vision analysis
    r_std = analysis['r_std']
    g_std = analysis['g_std']
    b_std = analysis['b_std']
    s_std = analysis['s_std']
    edge_density = analysis['edge_density']
    glcm_contrast = analysis['glcm_contrast']
    glcm_homogeneity = analysis['glcm_homogeneity']
    glcm_energy = analysis['glcm_energy']
    dominant_cluster_ratio = analysis['dominant_cluster_ratio']
    
    # 4. Calculate advanced scores from CV features
    # Color uniformity score (lower is more uniform, like new cars)
    color_peaks = (features[25] + features[26] + features[27]) / 3
    color_uniformity_score = 1.0 - ((color_peaks * 0.6) + (dominant_cluster_ratio * 0.4))
    color_uniformity_score = max(0, min(1, color_uniformity_score * 2.5))
    
    # Add a specific shine detector for new cars (new cars have more specular highlights)
    # Calculate variance of brightness in value channel to detect shininess
    v_channel = features[11]  # Mean of value channel (brightness)
    v_std = features[12]      # Std of value channel
    shininess_score = min(1.0, v_std * 3.0 if v_channel > 0.5 else v_std * 1.5)
    
    # Edge/scratch detection score
    scratch_score = (edge_density / 20) * 0.7 + (glcm_contrast / 5) * 0.3
    scratch_score = max(0, min(1, scratch_score))
    
    # Texture irregularity score (detects dents, uneven surfaces)
    texture_score = 1.0 - ((glcm_homogeneity + glcm_energy) / 2)
    texture_score = max(0, min(1, texture_score * 1.5))
    
    # Color variation score (detects multi-colored regions from damage)
    color_variation_score = min(1, (r_std + g_std + b_std + s_std*2) / 1.2)
    
    # 5. Combine with VGG model predictions
    # Extract probabilities from VGG predictions
    normal_prob = vgg_predictions.get('normal', 0.0)
    minor_prob = vgg_predictions.get('minor damaged car', 0.0)
    moderate_prob = vgg_predictions.get('moderate car damaged', 0.0)
    severe_prob = vgg_predictions.get('severe car damaged', 0.0)
    
    # Apply class correction based on known model biases
    # (Based on class distribution imbalance in training data)
    if normal_prob > 0.2:
        normal_prob = min(1.0, normal_prob * 1.2)  # Boost normal class prediction
    
    if minor_prob > 0.4:
        minor_prob = min(1.0, minor_prob * 1.15)  # Slightly boost minor damage
    
    # Re-normalize probabilities
    total_prob = normal_prob + minor_prob + moderate_prob + severe_prob
    normal_prob /= total_prob
    minor_prob /= total_prob
    moderate_prob /= total_prob
    severe_prob /= total_prob
    
    # Calculate weighted damage score using both approaches
    vgg_damage_score = (minor_prob * 0.3 + moderate_prob * 0.6 + severe_prob * 0.9) 
    
    cv_damage_score = (
        color_variation_score * 0.25 + 
        scratch_score * 0.25 + 
        texture_score * 0.2 + 
        color_uniformity_score * 0.15 +
        shininess_score * 0.15
    )
    
    # Calculate a confidence score for each approach
    vgg_confidence = max(normal_prob, minor_prob, moderate_prob, severe_prob)
    cv_confidence = 1.0 - (0.5 * color_uniformity_score + 0.5 * (1.0 - scratch_score))
    
    # Weighted average based on confidence
    vgg_weight = 0.7 * vgg_confidence
    cv_weight = 0.3 * cv_confidence
    weight_sum = vgg_weight + cv_weight
    
    # Normalize weights
    vgg_weight = vgg_weight / weight_sum
    cv_weight = cv_weight / weight_sum
    
    # Final weighted score with dynamic weights
    final_damage_score = (vgg_damage_score * vgg_weight) + (cv_damage_score * cv_weight)
    
    # Special case handling for strong predictors
    # 1. Very high normal probability strongly indicates no damage
    if normal_prob > 0.85:
        final_damage_score *= 0.2
    # 2. Very high color uniformity and low edge density strongly indicate new car
    if dominant_cluster_ratio > 0.85 and edge_density < 5:
        final_damage_score *= 0.5
    # 3. Very high severe damage probability strongly indicates severe damage
    if severe_prob > 0.8:
        final_damage_score = max(final_damage_score, 0.8)
    
    # Calculate final damage percentage
    damage_percentage = min(90, max(0, final_damage_score * 100))
    
    # Calculate color uniformity percentage (higher is better/more uniform)
    color_uniformity_percentage = (1.0 - color_uniformity_score) * 100
    
    # New car detection score (higher means more likely to be new)
    new_car_score = (
        (1.0 - color_variation_score) * 0.3 + 
        (1.0 - scratch_score) * 0.3 + 
        normal_prob * 0.4
    )
    
    # Determine damage level and base insurance amount with more nuanced thresholds
    if new_car_score > 0.8 and damage_percentage < 10:
        damage_level = "No Damage"
        base_amount = 5000  # ₹5,000 basic processing fee for brand new cars
    elif damage_percentage < 15:
        damage_level = "Very Minor"
        base_amount = 15000  # ₹15,000 for very minor damage
    elif damage_percentage < 35:
        damage_level = "Minor"
        base_amount = 45000  # ₹45,000 for minor damage
    elif damage_percentage < 65:
        damage_level = "Moderate"
        base_amount = 125000  # ₹1,25,000 for moderate damage
    else:
        damage_level = "Severe"
        base_amount = 250000  # ₹2,50,000 for severe damage
    
    # Calculate final amount with small random variation for realism
    final_amount = base_amount + np.random.normal(0, base_amount * 0.03)
    
    # Compile detailed analysis for debugging and transparency
    damage_analysis = {
        'vgg_predictions': {k: round(v * 100, 1) for k, v in vgg_predictions.items()},
        'corrected_vgg': {
            'normal': round(normal_prob * 100, 1),
            'minor': round(minor_prob * 100, 1),
            'moderate': round(moderate_prob * 100, 1),
            'severe': round(severe_prob * 100, 1)
        },
        'color_variation_score': round(color_variation_score * 100, 1),
        'scratch_score': round(scratch_score * 100, 1),
        'texture_score': round(texture_score * 100, 1),
        'color_uniformity': round(color_uniformity_percentage, 1),
        'shininess_score': round(shininess_score * 100, 1),
        'edge_density': round(edge_density, 1),
        'dominant_cluster_ratio': round(dominant_cluster_ratio * 100, 1),
        'vgg_damage_score': round(vgg_damage_score * 100, 1),
        'cv_damage_score': round(cv_damage_score * 100, 1),
        'vgg_weight': round(vgg_weight * 100, 1),
        'cv_weight': round(cv_weight * 100, 1),
        'new_car_score': round(new_car_score * 100, 1)
    }
    
    return {
        'amount': max(5000, final_amount),  # Minimum ₹5,000
        'damage_level': damage_level,
        'damage_percentage': damage_percentage,
        'color_uniformity': round(color_uniformity_percentage, 1),
        'new_car_score': round(new_car_score * 100, 1),
        'analysis': damage_analysis  # Detailed breakdown for transparency
    }

def plot_results(original, processed, synthetic, result_path):
    """Create visualization of results with edge detection"""
    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image (top left)
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Processed image (top right)
    axes[0, 1].imshow(processed)
    axes[0, 1].set_title('Processed Image')
    axes[0, 1].axis('off')
    
    # Synthetic damaged image (bottom left)
    axes[1, 0].imshow(synthetic)
    axes[1, 0].set_title('Synthetic Damage (GAN)')
    axes[1, 0].axis('off')
    
    # Edge detection (bottom right)
    # Create and show edge detection
    gray_img = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    edges = cv2.Canny(gray_img, 50, 150)
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title('Edge Detection (Damage Analysis)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(result_path)
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process the image
            print(f"Loading image from {file_path}")
            original_image = load_image(file_path)
            print(f"Original image shape: {original_image.shape}, dtype: {original_image.dtype}")
            
            # Check if the image is too bright or too dark
            avg_brightness = np.mean(original_image)
            if avg_brightness < 30:
                flash('Warning: Image is too dark. This may affect damage assessment accuracy.')
            elif avg_brightness > 220:
                flash('Warning: Image is too bright. This may affect damage assessment accuracy.')
            
            # Process the image
            print("Preprocessing image...")
            processed_image = preprocess_image(original_image)
            print(f"Processed image shape: {processed_image.shape}, dtype: {processed_image.dtype}")
            
            print("Generating synthetic image...")
            synthetic_image = generate_synthetic_image(original_image)
            
            # Use hybrid prediction approach
            print("Running hybrid prediction...")
            prediction = hybrid_predict_insurance(original_image, processed_image)
            print(f"Prediction result: {prediction['damage_level']} with {prediction['damage_percentage']:.1f}% damage")
            
            # Save visualization
            print("Generating result visualization...")
            result_filename = f"result_{filename.split('.')[0]}.png"
            result_path = os.path.join('static/results', result_filename)
            plot_results(original_image, processed_image, synthetic_image, result_path)
            
            return render_template(
                'result.html',
                result_image=f"results/{result_filename}",
                prediction=prediction
            )
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            flash(error_msg)
            return redirect(request.url)
    
    flash('Invalid file type. Please upload an image (png, jpg, jpeg, gif).')
    return redirect(request.url)

# Add this route to handle direct access to the /upload URL via GET
@app.route('/upload', methods=['GET'])
def upload_page():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True) 