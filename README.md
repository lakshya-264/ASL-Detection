# American Sign Language (ASL) Detection System

A deep learning system that detects and classifies American Sign Language signs from images. The system can identify 29 different classes: 26 letters (A-Z) and 3 special signs (SPACE, DELETE, NOTHING).

## Project Status

✅ **Complete and Ready for Deployment**
- All core functionality implemented
- Multiple training options available
- Evaluation and prediction scripts ready
- Comprehensive documentation included

## Dataset Structure

The dataset should be organized as follows:
```
asl_alphabet_train/
  asl_alphabet_train/
    A/          (3000 images)
    B/          (3000 images)
    ...
    Z/          (3000 images)
    space/      (3000 images)
    del/        (3000 images)
    nothing/    (3000 images)

asl_alphabet_test/
  asl_alphabet_test/
    A_test.jpg
    B_test.jpg
    ...
    Z_test.jpg
    space_test.jpg
    del_test.jpg
    nothing_test.jpg
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the model on the training dataset:

```bash
python train.py --train_dir asl_alphabet_train/asl_alphabet_train --epochs 50
```

**Arguments:**
- `--train_dir`: Path to training directory (default: `asl_alphabet_train/asl_alphabet_train`)
- `--model_type`: Type of model - `cnn` or `transfer` (default: `cnn`)
- `--img_size`: Image size as height width (default: `64 64`)
- `--batch_size`: Batch size for training (default: `32`)
- `--epochs`: Number of training epochs (default: `50`)
- `--validation_split`: Fraction of data for validation (default: `0.2`)
- `--learning_rate`: Learning rate (default: `0.001`)
- `--save_dir`: Directory to save model (default: `models`)

**Example with custom parameters:**
```bash
python train.py --train_dir asl_alphabet_train/asl_alphabet_train --epochs 100 --batch_size 64 --img_size 128 128
```

### Making Predictions

#### Predict on a single image:
```bash
python predict.py --image_path path/to/image.jpg
```

#### Evaluate on test set:
```bash
python predict.py --test_dir asl_alphabet_test/asl_alphabet_test
```

**Arguments:**
- `--model_path`: Path to trained model (default: `models/asl_model_best.h5`)
- `--class_mapping`: Path to class mapping JSON (default: `models/class_mapping.json`)
- `--image_path`: Path to single image for prediction
- `--test_dir`: Path to test directory
- `--img_size`: Image size as height width (default: `64 64`)

## Model Architecture

The system uses a Convolutional Neural Network (CNN) with the following architecture:

- **Input**: 64x64x3 RGB images
- **Convolutional Layers**: 4 blocks with increasing filters (32, 64, 128, 256)
- **Regularization**: Batch Normalization, Dropout layers
- **Dense Layers**: 2 fully connected layers (512, 256 neurons)
- **Output**: 29 classes with softmax activation

Alternatively, you can use transfer learning with MobileNetV2 by setting `--model_type transfer`.

## Features

- **Data Augmentation**: Rotation, shifting, shearing, and zooming to improve generalization
- **Early Stopping**: Prevents overfitting by stopping training when validation accuracy stops improving
- **Learning Rate Reduction**: Automatically reduces learning rate when validation loss plateaus
- **Model Checkpointing**: Saves the best model based on validation accuracy

## Output

After training, the following files will be saved in the `models/` directory:
- `asl_model_best.h5`: Best model based on validation accuracy
- `asl_model_final.h5`: Final model after all epochs
- `class_mapping.json`: Mapping of class indices to class names

## Performance

The model is evaluated on a separate test set with 29 test images (one per class). The evaluation includes:
- Overall accuracy
- Per-class accuracy
- Classification report with precision, recall, and F1-score
- Confusion matrix

## Notes

- The model expects RGB images
- Images are automatically resized to the specified size (default 64x64)
- Data augmentation is applied only during training
- The model uses categorical cross-entropy loss for multi-class classification

