"""
Training script for ASL alphabet detection model.
"""

import os
import argparse
from data_loader import ASLDataLoader
from model import create_asl_model, create_transfer_learning_model, compile_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


def train_model(
    train_dir,
    model_type='cnn',
    img_size=(64, 64),
    batch_size=64,  # Increased batch size for faster training
    epochs=50,
    validation_split=0.2,
    learning_rate=0.001,
    save_dir='models'
):
    """
    Train the ASL detection model.
    
    Args:
        train_dir: Path to training directory
        model_type: Type of model ('cnn' or 'transfer')
        img_size: Target image size
        batch_size: Batch size for training
        epochs: Number of training epochs
        validation_split: Fraction of data for validation
        learning_rate: Learning rate
        save_dir: Directory to save model
    """
    print("=" * 60)
    print("ASL Alphabet Detection - Training")
    print("=" * 60)
    
    # Create data loader
    print("\n[1/5] Loading data...")
    data_loader = ASLDataLoader(
        train_dir=train_dir,
        test_dir='',  # Not needed for training
        img_size=img_size,
        batch_size=batch_size
    )
    
    # Get classes
    classes = data_loader.get_classes()
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {', '.join(classes)}")
    
    # Create data generators
    print("\n[2/5] Creating data generators...")
    train_gen, val_gen, class_indices = data_loader.create_data_generators(
        validation_split=validation_split,
        augment=True
    )
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Create model
    print(f"\n[3/5] Creating {model_type.upper()} model...")
    if model_type == 'transfer':
        model = create_transfer_learning_model(
            input_shape=(*img_size, 3),
            num_classes=num_classes
        )
    else:
        model = create_asl_model(
            input_shape=(*img_size, 3),
            num_classes=num_classes
        )
    
    model = compile_model(model, learning_rate=learning_rate)
    model.summary()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Callbacks
    print("\n[4/5] Setting up callbacks...")
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(save_dir, 'asl_model_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n[5/5] Training model...")
    print("=" * 60)
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'asl_model_final.h5')
    model.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    # Save class mapping
    import json
    class_mapping_path = os.path.join(save_dir, 'class_mapping.json')
    with open(class_mapping_path, 'w') as f:
        json.dump(class_indices, f, indent=2)
    print(f"Class mapping saved to: {class_mapping_path}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ASL alphabet detection model')
    parser.add_argument('--train_dir', type=str, 
                       default='asl_alphabet_train/asl_alphabet_train',
                       help='Path to training directory')
    parser.add_argument('--model_type', type=str, default='cnn',
                       choices=['cnn', 'transfer'],
                       help='Type of model to train')
    parser.add_argument('--img_size', type=int, nargs=2, default=[64, 64],
                       help='Image size (height width)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save model')
    
    args = parser.parse_args()
    
    train_model(
        train_dir=args.train_dir,
        model_type=args.model_type,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )

