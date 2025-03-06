#!/usr/bin/env python3
"""
Iris Example - Training a neural network on the Iris flower dataset
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Debug print
print("Script started")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_network import NeuralNetwork

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load Iris dataset
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)
    
    # One-hot encode the target
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_one_hot, test_size=0.2, random_state=42
    )
    
    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(iris.target_names)}")
    print(f"Classes: {iris.target_names}")
    print(f"Feature names: {iris.feature_names}")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create a neural network
    model = NeuralNetwork(input_size=4, loss='categorical_cross_entropy')
    
    # Add layers
    model.add_layer(10, 'tanh')  # Hidden layer with 10 neurons
    model.add_layer(3, 'softmax')  # Output layer with 3 neurons (one for each class)
    
    # Print model structure
    print("\nNeural Network Structure:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}: {layer.input_size} -> {layer.output_size} ({layer.activation_name})")
    
    # Train the model
    print("\nTraining the model...")
    history = model.train(
        X=X_train,
        y=y_train,
        epochs=1000,
        batch_size=16,
        learning_rate=0.1
    )
    
    # Print training results
    print(f"\nTraining completed.")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['accuracy'][-1]:.4f}")
    
    # Evaluate on test set
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('iris_training.png')
    print("\nTraining plot saved as 'iris_training.png'")
    
    # Make predictions on some examples
    print("\nSample predictions:")
    
    # Get 3 samples from each class
    sample_indices = []
    for class_idx in range(3):
        class_indices = np.where(np.argmax(y_one_hot, axis=1) == class_idx)[0]
        sample_indices.extend(class_indices[:3])
    
    sample_X = X_scaled[sample_indices]
    sample_y = y_one_hot[sample_indices]
    sample_predictions = model.predict(sample_X)
    
    for i, idx in enumerate(sample_indices):
        true_class = iris.target_names[np.argmax(sample_y[i])]
        pred_class = iris.target_names[np.argmax(sample_predictions[i])]
        pred_probs = sample_predictions[i]
        
        print(f"Sample {i+1} (Iris {idx}):")
        print(f"  Features: {X[idx]}")
        print(f"  True class: {true_class}")
        print(f"  Predicted class: {pred_class}")
        print(f"  Prediction probabilities: {pred_probs}")
        print(f"  Correct: {'✓' if true_class == pred_class else '✗'}")
        print()

if __name__ == "__main__":
    main() 