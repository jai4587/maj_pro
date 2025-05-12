#!/usr/bin/env python
import argparse
import cv2
import json
import os
from damage_predictor import DamagePredictor

def main():
    """Command-line tool for car damage prediction"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict car damage from images")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="models/fast_model.h5", help="Path to model file")
    parser.add_argument("--car_value", type=float, default=1875000, help="Value of the car in INR")
    parser.add_argument("--car_make", default=None, help="Car make (e.g., Toyota, BMW)")
    parser.add_argument("--car_model", default=None, help="Car model (e.g., Camry, 3-Series)")
    parser.add_argument("--car_year", type=int, default=None, help="Car year of manufacture")
    parser.add_argument("--output", default="prediction_result.jpg", help="Output path for visualization")
    parser.add_argument("--currency", default="INR", choices=["USD", "INR"], help="Currency for cost estimates")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return 1
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return 1
    
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
    
    # Convert INR to USD for processing (predictor works in USD internally)
    car_value_usd = args.car_value / 75.0
    
    try:
        # Initialize predictor
        predictor = DamagePredictor(model_path=args.model, currency=args.currency)
        
        # Load image
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image from {args.image}")
            return 1
        
        # Process image
        result = predictor.estimate_repair_cost(image, car_value_usd, car_info)
        
        # Create a copy of result for JSON output
        result_copy = result.copy()
        if "visualization" in result_copy["damage_assessment"]:
            del result_copy["damage_assessment"]["visualization"]
        
        # Save visualization
        if "visualization" in result["damage_assessment"]:
            cv2.imwrite(args.output, cv2.cvtColor(result["damage_assessment"]["visualization"], cv2.COLOR_RGB2BGR))
            print(f"Visualization saved to {args.output}")
        
        # Output result
        if args.json:
            # Print as JSON
            print(json.dumps(result_copy, indent=2))
        else:
            # Print as formatted text
            print("\nDamage Assessment Results:")
            print(f"Predicted class: {result['damage_assessment']['predicted_class']}")
            print(f"Confidence: {result['damage_assessment']['confidence'] * 100:.1f}%")
            
            if args.currency == "USD":
                print(f"Car value: ${result['car_value']:.2f}")
                print(f"         (₹{result['car_value_inr']:.2f})")
                print(f"Estimated repair cost: ${result['estimated_repair_cost']:.2f}")
                print(f"                     (₹{result['estimated_repair_cost_inr']:.2f})")
            else:
                print(f"Car value: ₹{result['car_value_inr']:.2f}")
                print(f"         (${result['car_value']:.2f})")
                print(f"Estimated repair cost: ₹{result['estimated_repair_cost_inr']:.2f}")
                print(f"                     (${result['estimated_repair_cost']:.2f})")
            
            print(f"\nFull results saved to prediction_result.json")
        
        # Save full result as JSON
        with open("prediction_result.json", "w") as f:
            json.dump(result_copy, f, indent=2)
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main()) 