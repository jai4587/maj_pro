# Car Damage Assessment System

An AI-powered application that assesses car damage from images and estimates repair costs.

## Overview

This system uses deep learning to analyze images of damaged cars, classify the damage level, and estimate repair costs based on the car's value and characteristics. It provides a user-friendly web interface for uploading images and viewing results.

## Features

- **Damage Classification**: Accurately categorizes damage as minor, moderate, severe, or normal (undamaged)
- **Cost Estimation**: Calculates repair costs based on damage level and car characteristics
- **Dual Currency Support**: Shows estimates in both INR (primary) and USD
- **Visual Analysis**: Provides visualization of damage assessment with confidence scores
- **PDF Reporting**: Generates downloadable PDF reports for sharing or documentation
- **User-friendly Interface**: Simple web application for easy image uploads and result viewing

## Components

1. **MobileNetV2-based Model**: Lightweight and efficient neural network for damage classification
2. **DamagePredictor Class**: Core class that handles predictions and cost estimation
3. **Flask Web Application**: User interface for interacting with the system
4. **Evaluation Tools**: Scripts for testing model performance

## Technologies Used

- **TensorFlow/Keras**: For deep learning model development
- **Flask**: For web application backend
- **OpenCV**: For image processing
- **ReportLab/PDFKit**: For PDF generation
- **Bootstrap**: For responsive UI design

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd car-damage-assessment
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python damage_app.py
   ```

4. Access the web interface at http://localhost:5000

## Model Information

- Architecture: MobileNetV2 (transfer learning)
- Input Size: 160x160 pixels
- Classes: 4 (minor, moderate, severe, normal)
- Accuracy: ~82% on test set
- Training: Fine-tuned with proper class weighting to handle dataset imbalance

## Usage

1. Upload a car image
2. Enter car details (make, model, year, value in INR)
3. View damage assessment and repair cost estimate
4. Download PDF report if needed

## Directory Structure

- `damage_app.py`: Main Flask application
- `damage_predictor.py`: Core prediction and cost estimation logic
- `predict_damage.py`: Command-line tool for damage prediction
- `models/`: Contains trained model files
- `templates/`: HTML templates for the web interface
- `uploads/`: Temporary storage for uploaded images
- `results/`: Storage for result visualizations and PDFs

## License

This project is licensed under the MIT License - see the LICENSE file for details. 