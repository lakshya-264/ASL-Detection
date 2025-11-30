"""
Quick start script to train and evaluate the ASL detection model.
This is a simplified version for easy execution.
"""

from data_loader import ASLDataLoader
from model import create_asl_model, compile_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import json

def main():
    print("=" * 60)
    print("ASL Alphabet Detection - Quick Start")
    print("=" * 60)
    
    # Configuration
    train_dir = 'asl_alphabet_train/asl_alphabet_train'
    test_dir = 'asl_alphabet_test/asl_alphabet_test'
    img_size = (64, 64)
    batch_size = 32
    epochs = 30
    validation_split = 0.2
    
    # Create data loader
    print("\n[1/4] Loading data...")
    data_loader = ASLDataLoader(
        train_dir=train_dir,
        test_dir=test_dir,
        img_size=img_size,
        batch_size=batch_size
    )
    
    classes = data_loader.get_classes()
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    # Create data generators
    print("\n[2/4] Creating data generators...")
    train_gen, val_gen, class_indices = data_loader.create_data_generators(
        validation_split=validation_split,
        augment=True
    )
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    # Create model
    print("\n[3/4] Creating model...")
    model = create_asl_model(
        input_shape=(*img_size, 3),
        num_classes=len(classes)
    )
    model = compile_model(model, learning_rate=0.001)
    model.summary()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath='models/asl_model_best.h5',
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
    
    # Train
    print("\n[4/4] Training model...")
    print("=" * 60)
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/asl_model_final.h5')
    print("\nModel saved to: models/asl_model_final.h5")
    
    # Save class mapping
    with open('models/class_mapping.json', 'w') as f:
        json.dump(class_indices, f, indent=2)
    print("Class mapping saved to: models/class_mapping.json")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    from predict import evaluate_test_set
    evaluate_test_set(
        model, test_dir, class_indices, img_size
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == '__main__':
    main()

