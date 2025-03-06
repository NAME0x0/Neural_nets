#!/usr/bin/env python3
"""
XOR Example - A simple example of training a neural network on the XOR problem
"""

import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_network import NeuralNetwork

def main():
    # Set random seed for reproducibility
    np.random.seed(2)  # Same seed as the direct implementation
    
    # Create the XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create a neural network
    model = NeuralNetwork(input_size=2, loss='binary_cross_entropy')
    
    # Add layers - using the same architecture as the direct implementation
    model.add_layer(8, 'tanh')  # Hidden layer with 8 neurons using tanh activation
    model.add_layer(1, 'sigmoid')  # Output layer with 1 neuron
    
    # Manually initialize weights using He initialization
    for i, layer in enumerate(model.layers):
        if i == 0:
            # First layer: input_size = 2, output_size = 8
            layer.weights = np.random.randn(2, 8) * np.sqrt(2/2)
            layer.biases = np.zeros((1, 8))
        else:
            # Second layer: input_size = 8, output_size = 1
            layer.weights = np.random.randn(8, 1) * np.sqrt(2/8)
            layer.biases = np.zeros((1, 1))
    
    # Print model structure
    print("Neural Network Structure:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}: {layer.input_size} -> {layer.output_size} ({layer.activation_name})")
    
    # Train the model
    print("\nTraining the model...")
    history = model.train(
        X=X,
        y=y,
        epochs=20000,  # Same as direct implementation
        batch_size=4,
        learning_rate=3.0  # Same as direct implementation
    )
    
    # Print training results
    print(f"\nTraining completed.")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    print(f"Final accuracy: {history['accuracy'][-1]:.4f}")
    
    # Make predictions
    predictions = model.predict(X)
    
    # Print predictions
    print("\nPredictions:")
    for i in range(len(X)):
        input_values = f"[{X[i,0]}, {X[i,1]}]"
        expected = y[i,0]
        predicted = predictions[i,0]
        correct = "✓" if (predicted > 0.5) == (expected > 0.5) else "✗"
        print(f"Input: {input_values}, Expected: {expected}, Predicted: {predicted:.4f} {correct}")

if __name__ == "__main__":
    main() 