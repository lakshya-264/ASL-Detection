"""
Prediction script for ASL alphabet detection.
Can predict on single images or test dataset.
"""

import os
import argparse
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


def predict_image(model, image_path, class_mapping, img_size=(64, 64)):
    """
    Predict ASL sign from a single image.
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        class_mapping: Dictionary mapping class indices to names
        img_size: Target image size
        
    Returns:
        predicted_class: Predicted class name
        confidence: Confidence score
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    
    # Get class name
    idx_to_class = {v: k for k, v in class_mapping.items()}
    predicted_class = idx_to_class[predicted_idx]
    
    return predicted_class, confidence


def evaluate_test_set(model, test_dir, class_mapping, img_size=(64, 64)):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        test_dir: Path to test directory
        class_mapping: Dictionary mapping class indices to names
        img_size: Target image size
        
    Returns:
        accuracy: Test accuracy
        predictions: List of predictions
        true_labels: List of true labels
    """
    # Load test images manually
    test_images = []
    test_labels = []
    test_files = []
    
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    for filename in os.listdir(test_dir):
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
                img_path = os.path.join(test_dir, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                
                test_images.append(img_array)
                test_labels.append(class_mapping[class_name])
                test_files.append(filename)
    
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    print(f"\nLoaded {len(test_images)} test images")
    
    # Predict
    predictions = model.predict(test_images, verbose=1)
    predicted_indices = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_indices == test_labels)
    
    # Get class names
    idx_to_class = {v: k for k, v in class_mapping.items()}
    predicted_classes = [idx_to_class[idx] for idx in predicted_indices]
    true_classes = [idx_to_class[idx] for idx in test_labels]
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nDetailed Results:")
    print("-" * 60)
    
    correct = 0
    for i, (true, pred, file) in enumerate(zip(true_classes, predicted_classes, test_files)):
        status = "[OK]" if true == pred else "[X]"
        if true == pred:
            correct += 1
        print(f"{status} {file:20s} | True: {true:10s} | Predicted: {pred:10s} | "
              f"Confidence: {predictions[i][predicted_indices[i]]:.4f}")
    
    print("-" * 60)
    print(f"Correct: {correct}/{len(test_images)}")
    
    # Classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    # Get unique classes from test data
    unique_classes = sorted(set(true_classes + predicted_classes))
    print(classification_report(true_classes, predicted_classes, 
                                labels=[class_mapping.get(c) for c in unique_classes if c in class_mapping],
                                target_names=unique_classes, zero_division=0))
    
    return accuracy, predicted_classes, true_classes


def main():
    parser = argparse.ArgumentParser(description='Predict ASL signs from images')
    parser.add_argument('--model_path', type=str, default='models/asl_model_best.h5',
                       help='Path to trained model')
    parser.add_argument('--class_mapping', type=str, default='models/class_mapping.json',
                       help='Path to class mapping JSON file')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to single image for prediction')
    parser.add_argument('--test_dir', type=str, default='asl_alphabet_test/asl_alphabet_test',
                       help='Path to test directory')
    parser.add_argument('--img_size', type=int, nargs=2, default=[64, 64],
                       help='Image size (height width)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path)
    print("Model loaded successfully!")
    
    # Load class mapping
    with open(args.class_mapping, 'r') as f:
        class_mapping = json.load(f)
    
    print(f"Loaded {len(class_mapping)} classes")
    
    # Predict on single image or test set
    if args.image_path:
        if os.path.exists(args.image_path):
            predicted_class, confidence = predict_image(
                model, args.image_path, class_mapping, tuple(args.img_size)
            )
            print(f"\nPredicted class: {predicted_class}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        else:
            print(f"Error: Image file not found: {args.image_path}")
    else:
        evaluate_test_set(
            model, args.test_dir, class_mapping, tuple(args.img_size)
        )


if __name__ == '__main__':
    main()

