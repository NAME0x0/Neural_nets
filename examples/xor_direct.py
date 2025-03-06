#!/usr/bin/env python3
"""
XOR Direct Implementation - A simple neural network for XOR without using our library
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid activation function"""
    s = sigmoid(x)
    return s * (1 - s)

def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize parameters for a 2-layer neural network
    
    Args:
        n_x: Size of the input layer
        n_h: Size of the hidden layer
        n_y: Size of the output layer
        
    Returns:
        Dictionary containing the initialized parameters
    """
    np.random.seed(2)  # Changed seed
    
    # Using He initialization for better convergence
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2/n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * np.sqrt(2/n_h)
    b2 = np.zeros((n_y, 1))
    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return parameters

def forward_propagation(X, parameters):
    """
    Forward propagation
    
    Args:
        X: Input data, shape (n_x, m)
        parameters: Dictionary containing the parameters
        
    Returns:
        Dictionary containing the forward propagation results
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    
    return A2, cache

def compute_cost(A2, Y):
    """
    Compute the cost
    
    Args:
        A2: Output of the forward propagation, shape (1, m)
        Y: True labels, shape (1, m)
        
    Returns:
        Cost
    """
    m = Y.shape[1]
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    A2 = np.clip(A2, epsilon, 1 - epsilon)
    
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(logprobs) / m
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    Backward propagation
    
    Args:
        parameters: Dictionary containing the parameters
        cache: Dictionary containing the forward propagation results
        X: Input data, shape (n_x, m)
        Y: True labels, shape (1, m)
        
    Returns:
        Dictionary containing the gradients
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters
    
    Args:
        parameters: Dictionary containing the parameters
        grads: Dictionary containing the gradients
        learning_rate: Learning rate
        
    Returns:
        Dictionary containing the updated parameters
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    return parameters

def nn_model(X, Y, n_h, num_iterations, learning_rate, print_cost=False):
    """
    Neural network model
    
    Args:
        X: Input data, shape (n_x, m)
        Y: True labels, shape (1, m)
        n_h: Size of the hidden layer
        num_iterations: Number of iterations
        learning_rate: Learning rate
        print_cost: Whether to print the cost every 1000 iterations
        
    Returns:
        Dictionary containing the parameters
    """
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    costs = []
    
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)
    
    return parameters, costs

def predict(parameters, X):
    """
    Predict using the trained model
    
    Args:
        parameters: Dictionary containing the parameters
        X: Input data, shape (n_x, m)
        
    Returns:
        Predictions, shape (1, m)
    """
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    
    return predictions

def main():
    # Create the XOR dataset
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # 2x4 matrix
    Y = np.array([[0, 1, 1, 0]])  # 1x4 matrix
    
    # Train the model
    print("Training the model...")
    parameters, costs = nn_model(X, Y, n_h=8, num_iterations=20000, learning_rate=3.0, print_cost=True)
    
    # Make predictions
    predictions = predict(parameters, X)
    
    # Print predictions
    print("\nPredictions:")
    for i in range(X.shape[1]):
        input_values = f"[{X[0,i]}, {X[1,i]}]"
        expected = Y[0,i]
        predicted = predictions[0,i]
        correct = "✓" if predicted == expected else "✗"
        print(f"Input: {input_values}, Expected: {expected}, Predicted: {int(predicted)} {correct}")
    
    # Calculate accuracy
    accuracy = np.mean(predictions == Y)
    print(f"\nAccuracy: {accuracy * 100}%")
    
    # Plot the cost
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('Iterations (thousands)')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iterations')
    plt.grid(True)
    plt.savefig('xor_cost.png')
    print("\nCost plot saved as 'xor_cost.png'")

if __name__ == "__main__":
    main() 