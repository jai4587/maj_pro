import cv2
import numpy as np
from PIL import Image
import os

class ImageInput:
    def __init__(self, upload_dir='uploads'):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
    
    def load_image(self, image_path):
        """Load image from file path"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            # Read image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image from {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def save_image(self, image, filename):
        """Save image to upload directory"""
        try:
            if not isinstance(image, np.ndarray):
                raise ValueError("Input must be a numpy array")
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(self.upload_dir, filename)
            cv2.imwrite(save_path, image)
            return save_path
        except Exception as e:
            raise Exception(f"Error saving image: {str(e)}")
    
    def capture_from_camera(self):
        """Capture image from camera"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Failed to open camera")
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise Exception("Failed to capture image")
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except Exception as e:
            raise Exception(f"Error capturing image: {str(e)}")
    
    def validate_image(self, image):
        """Validate image format and size"""
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB with 3 channels")
        
        return True 