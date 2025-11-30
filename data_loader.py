"""
Data loader for ASL alphabet dataset.
Handles loading and preprocessing of training and test images.
"""

import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


class ASLDataLoader:
    """Class to load and preprocess ASL alphabet dataset."""
    
    def __init__(self, train_dir, test_dir, img_size=(64, 64), batch_size=32):
        """
        Initialize the data loader.
        
        Args:
            train_dir: Path to training directory
            test_dir: Path to test directory
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = None
        self.num_classes = None
        
    def get_classes(self):
        """Get sorted list of class names from training directory."""
        if self.classes is None:
            class_dirs = [d for d in os.listdir(self.train_dir) 
                         if os.path.isdir(os.path.join(self.train_dir, d))]
            # Sort classes: A-Z first, then special classes
            letters = sorted([c for c in class_dirs if len(c) == 1 and c.isalpha()])
            special = sorted([c for c in class_dirs if len(c) > 1 or not c.isalpha()])
            self.classes = letters + special
            self.num_classes = len(self.classes)
        return self.classes
    
    def get_class_mapping(self):
        """Get mapping from class names to indices."""
        classes = self.get_classes()
        return {class_name: idx for idx, class_name in enumerate(classes)}
    
    def create_data_generators(self, validation_split=0.2, augment=True):
        """
        Create data generators for training and validation.
        
        Args:
            validation_split: Fraction of data to use for validation
            augment: Whether to use data augmentation for training
            
        Returns:
            train_generator, validation_generator, class_indices
        """
        classes = self.get_classes()
        
        # Data augmentation for training
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split,
                rotation_range=15,  # Reduced rotation
                width_shift_range=0.15,  # Reduced shift
                height_shift_range=0.15,
                shear_range=0.15,  # Reduced shear
                zoom_range=0.15,  # Reduced zoom
                horizontal_flip=False,  # Don't flip ASL signs
                brightness_range=[0.8, 1.2],  # Add brightness variation
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
        
        # Validation generator (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        validation_generator = val_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator, train_generator.class_indices
    
    def load_test_images(self):
        """
        Load test images from test directory.
        
        Returns:
            test_images: List of image arrays
            test_labels: List of true labels
            test_files: List of filenames
        """
        test_images = []
        test_labels = []
        test_files = []
        
        class_mapping = self.get_class_mapping()
        
        for filename in os.listdir(self.test_dir):
            if filename.endswith('.jpg'):
                # Extract class name from filename (e.g., "A_test.jpg" -> "A")
                class_name = filename.replace('_test.jpg', '').lower()
                
                # Map special cases
                if class_name == 'delete' or class_name == 'del':
                    class_name = 'del'
                elif class_name == 'nothing':
                    class_name = 'nothing'
                elif class_name == 'space':
                    class_name = 'space'
                
                # Convert to uppercase for letters
                if len(class_name) == 1 and class_name.isalpha():
                    class_name = class_name.upper()
                
                if class_name in class_mapping:
                    img_path = os.path.join(self.test_dir, filename)
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0
                    
                    test_images.append(img_array)
                    test_labels.append(class_mapping[class_name])
                    test_files.append(filename)
        
        return np.array(test_images), np.array(test_labels), test_files
    
    def get_class_name(self, class_idx):
        """Get class name from index."""
        classes = self.get_classes()
        return classes[class_idx] if 0 <= class_idx < len(classes) else None

