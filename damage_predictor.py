import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import json

# USD to INR conversion rate
USD_TO_INR_RATE = 75.0

class DamagePredictor:
    """
    Car damage assessment model wrapper that provides a simple interface
    for damage prediction and visualization
    """
    
    def __init__(self, model_path="models/fast_model.h5", img_size=160, currency="INR"):
        self.img_size = img_size
        self.class_names = ["minor damaged car", "moderate car damaged", "normal", "severe car damaged"]
        self.damage_multipliers = {
            "minor damaged car": 0.25,     # 25% of car value for minor damage
            "moderate car damaged": 0.45,  # 45% of car value for moderate damage
            "normal": 0.0,                # 0% for undamaged cars
            "severe car damaged": 0.75     # 75% of car value for severe damage
        }
        self.currency = currency
        
        # Load the model if it exists
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} not found")
    
    def preprocess_image(self, image):
        """Preprocess a single image for prediction"""
        # Convert to RGB if grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Resize to match model input size
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert to float and normalize
        image = img_to_array(image) / 255.0
        
        return image
    
    def predict_damage(self, image, return_visualization=False):
        """
        Predict damage category and estimate for a single image
        
        Args:
            image: Input image (numpy array, BGR format from OpenCV)
            return_visualization: Whether to return a visualization of the prediction
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Add batch dimension
        image_batch = np.expand_dims(processed_image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image_batch)[0]
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction)
        confidence = prediction[predicted_class_idx]
        
        # Get class name and damage multiplier
        class_name = self.class_names[predicted_class_idx]
        damage_multiplier = self.damage_multipliers[class_name]
        
        # Create result dictionary
        result = {
            "predicted_class": class_name,
            "confidence": float(confidence),
            "damage_multiplier": damage_multiplier,
            "prediction_vector": prediction.tolist(),
            "class_probabilities": {
                self.class_names[i]: float(prediction[i]) 
                for i in range(len(self.class_names))
            }
        }
        
        # Generate visualization if requested
        if return_visualization:
            result["visualization"] = self.generate_visualization(image, result)
            
        return result
    
    def generate_visualization(self, image, prediction_result):
        """Generate a visualization of the prediction"""
        # Create a copy of the image for drawing
        vis_image = image.copy()
        
        # Resize for display if needed
        if vis_image.shape[0] > 800 or vis_image.shape[1] > 800:
            scale = min(800 / vis_image.shape[0], 800 / vis_image.shape[1])
            vis_image = cv2.resize(vis_image, None, fx=scale, fy=scale)
        
        # Convert BGR to RGB
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(vis_image)
        plt.title(f"Predicted: {prediction_result['predicted_class']}\nConfidence: {prediction_result['confidence']:.2f}")
        plt.axis('off')
        
        # Display probabilities as bar chart
        plt.subplot(1, 2, 2)
        class_probs = prediction_result["class_probabilities"]
        classes = list(class_probs.keys())
        probs = [class_probs[c] for c in classes]
        
        # Choose colors based on damage level
        colors = ['green' if c == 'normal' else 'yellow' if c == 'minor damaged car' 
                 else 'orange' if c == 'moderate car damaged' else 'red' 
                 for c in classes]
        
        bars = plt.barh(classes, probs, color=colors)
        plt.xlim(0, 1)
        plt.title("Damage Category Probabilities")
        plt.xlabel("Probability")
        
        # Add value labels
        for bar, prob in zip(bars, probs):
            plt.text(min(prob + 0.05, 0.95), bar.get_y() + bar.get_height()/2, 
                    f"{prob:.2f}", va='center')
        
        plt.tight_layout()
        
        # Convert plot to image
        fig = plt.gcf()
        plt.close()
        
        # Convert figure to image
        fig.canvas.draw()
        vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return vis
    
    def estimate_repair_cost(self, image, car_value, car_info=None):
        """
        Estimate repair cost based on damage prediction and car value
        
        Args:
            image: Input image (numpy array, BGR format from OpenCV)
            car_value: Estimated value of the car in undamaged condition (in USD)
            car_info: Optional dictionary with additional car information
                      (e.g. make, model, year, etc.)
        
        Returns:
            Dictionary with damage assessment and cost estimate
        """
        # Get basic prediction
        prediction = self.predict_damage(image, return_visualization=True)
        
        # Base repair cost as percentage of car value
        base_repair_cost = car_value * prediction["damage_multiplier"]
        
        # Apply adjustments based on car info if provided
        cost_multiplier = 1.0
        if car_info:
            # Apply age adjustment (newer cars cost more to repair)
            if "year" in car_info:
                current_year = 2025  # Update this as needed
                age = current_year - car_info["year"]
                if age <= 3:
                    cost_multiplier *= 1.2  # Newer cars (0-3 years) cost more to repair
                elif age <= 7:
                    cost_multiplier *= 1.1  # Mid-age cars (4-7 years)
                # Older cars use the base multiplier
            
            # Apply make adjustment (luxury brands cost more to repair)
            if "make" in car_info:
                luxury_brands = ["mercedes", "bmw", "audi", "lexus", "porsche", "tesla"]
                if any(brand in car_info["make"].lower() for brand in luxury_brands):
                    cost_multiplier *= 1.3  # Luxury vehicles cost more to repair
        
        # Calculate final repair cost (in USD)
        repair_cost_usd = base_repair_cost * cost_multiplier
        
        # Convert to INR
        repair_cost_inr = repair_cost_usd * USD_TO_INR_RATE
        car_value_inr = car_value * USD_TO_INR_RATE
        base_repair_cost_inr = base_repair_cost * USD_TO_INR_RATE
        
        # Compile and return results
        estimate = {
            "damage_assessment": prediction,
            "car_value": car_value,
            "car_value_inr": float(car_value_inr),
            "car_info": car_info,
            "base_repair_cost": float(base_repair_cost),
            "base_repair_cost_inr": float(base_repair_cost_inr),
            "cost_multiplier": float(cost_multiplier),
            "estimated_repair_cost": float(repair_cost_usd),
            "estimated_repair_cost_inr": float(repair_cost_inr),
            "confidence": float(prediction["confidence"]),
            "currency": self.currency
        }
        
        return estimate
    
    def process_image(self, image_path, car_value=10000, car_info=None):
        """Process an image file and return damage assessment with visualization"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        
        # Get estimate
        result = self.estimate_repair_cost(image, car_value, car_info)
        
        return result

# Simple example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict car damage from images")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="models/fast_model.h5", help="Path to model file")
    parser.add_argument("--car_value", type=float, default=10000, help="Value of the car in undamaged condition (USD)")
    parser.add_argument("--car_make", default=None, help="Car make (e.g., Toyota, BMW)")
    parser.add_argument("--car_model", default=None, help="Car model (e.g., Camry, 3-Series)")
    parser.add_argument("--car_year", type=int, default=None, help="Car year of manufacture")
    parser.add_argument("--output", default="result.jpg", help="Output path for visualization")
    parser.add_argument("--currency", default="INR", choices=["USD", "INR"], help="Currency for cost estimates")
    
    args = parser.parse_args()
    
    # Create car info dictionary if any info provided
    car_info = None
    if args.car_make or args.car_model or args.car_year:
        car_info = {}
        if args.car_make:
            car_info["make"] = args.car_make
        if args.car_model:
            car_info["model"] = args.car_model
        if args.car_year:
            car_info["year"] = args.car_year
    
    # Initialize predictor
    predictor = DamagePredictor(model_path=args.model, currency=args.currency)
    
    # Process image
    result = predictor.process_image(args.image, args.car_value, car_info)
    
    # Print results
    print("\nDamage Assessment Results:")
    print(f"Predicted class: {result['damage_assessment']['predicted_class']}")
    print(f"Confidence: {result['damage_assessment']['confidence'] * 100:.1f}%")
    print(f"Car value: ${result['car_value']:.2f} (₹{result['car_value_inr']:.2f})")
    
    if args.currency == "USD":
        print(f"Estimated repair cost: ${result['estimated_repair_cost']:.2f}")
        print(f"                       (₹{result['estimated_repair_cost_inr']:.2f})")
    else:
        print(f"Estimated repair cost: ₹{result['estimated_repair_cost_inr']:.2f}")
        print(f"                       (${result['estimated_repair_cost']:.2f})")
    
    # Save visualization
    if "visualization" in result["damage_assessment"]:
        cv2.imwrite(args.output, cv2.cvtColor(result["damage_assessment"]["visualization"], cv2.COLOR_RGB2BGR))
        print(f"Visualization saved to {args.output}")
        
    # Save full result as JSON
    with open("prediction_result.json", "w") as f:
        # Remove visualization from JSON output
        result_copy = result.copy()
        if "visualization" in result_copy["damage_assessment"]:
            del result_copy["damage_assessment"]["visualization"]
        json.dump(result_copy, f, indent=2)
        print("Full results saved to prediction_result.json") 