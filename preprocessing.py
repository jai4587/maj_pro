import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomResizedCrop(height=target_size[0], width=target_size[1], scale=(0.8, 1.0), p=0.5)
        ])
    
    def resize_image(self, image):
        """Resize image to target size"""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image):
        """Normalize image pixel values to [0,1]"""
        return image.astype(np.float32) / 255.0
    
    def denoise_image(self, image):
        """Remove noise from image using Non-local Means Denoising"""
        try:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        except Exception as e:
            print(f"Denoising error: {str(e)}, using original image")
            return image
    
    def augment_image(self, image):
        """Apply data augmentation to image"""
        try:
            augmented = self.augmentation(image=image)
            return augmented['image']
        except Exception as e:
            print(f"Augmentation error: {str(e)}, using original image")
            return image
    
    def preprocess_pipeline(self, image, augment=False):
        """Complete preprocessing pipeline"""
        # Resize
        image = self.resize_image(image)
        
        # Denoise
        image = self.denoise_image(image)
        
        # Augment if requested
        if augment:
            image = self.augment_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        return image
    
    def batch_preprocess(self, images, augment=False):
        """Preprocess a batch of images"""
        return np.array([self.preprocess_pipeline(img, augment) for img in images])
    
    def create_augmentation_generator(self, batch_size=32):
        """Create a data generator for augmentation"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        ) 