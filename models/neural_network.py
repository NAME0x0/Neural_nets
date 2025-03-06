import numpy as np
from typing import List, Dict, Callable, Tuple, Optional, Union
import json
import os

# Activation functions
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid activation function"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU activation function"""
    return np.where(x > 0, 1, 0)

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function"""
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh activation function"""
    return 1 - np.tanh(x) ** 2

def linear(x: np.ndarray) -> np.ndarray:
    """Linear activation function"""
    return x

def linear_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of linear activation function"""
    return np.ones_like(x)

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss functions
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error loss function"""
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Derivative of mean squared error loss function"""
    return -2 * (y_true - y_pred) / y_true.shape[0]

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross entropy loss function"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Derivative of binary cross entropy loss function"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -((y_true / y_pred) - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Categorical cross entropy loss function"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Derivative of categorical cross entropy loss function"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -y_true / y_pred / y_true.shape[0]

# Dictionary of available activation functions
ACTIVATION_FUNCTIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, linear_derivative),
    'softmax': (softmax, None)  # Softmax derivative is handled specially in combination with loss
}

# Dictionary of available loss functions
LOSS_FUNCTIONS = {
    'mean_squared_error': (mean_squared_error, mean_squared_error_derivative),
    'binary_cross_entropy': (binary_cross_entropy, binary_cross_entropy_derivative),
    'categorical_cross_entropy': (categorical_cross_entropy, categorical_cross_entropy_derivative)
}

class Layer:
    """Represents a single layer in a neural network"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'sigmoid'):
        """
        Initialize a layer
        
        Args:
            input_size: Number of input neurons
            output_size: Number of output neurons
            activation: Activation function name
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        
        # Get activation functions
        self.activation_func, self.activation_derivative = ACTIVATION_FUNCTIONS[activation]
        
        # Forward pass storage for backpropagation
        self.input = None
        self.z = None
        self.output = None
        
        # Gradients for visualization
        self.dweights = None
        self.dbiases = None
        
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer
        
        Args:
            input_data: Input data
            
        Returns:
            Output after activation
        """
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        self.output = self.activation_func(self.z)
        return self.output
    
    def backward(self, delta: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through the layer
        
        Args:
            delta: Derivative of loss with respect to layer output
            learning_rate: Learning rate
            
        Returns:
            Derivative of loss with respect to layer input
        """
        if self.activation_name != 'softmax':
            delta = delta * self.activation_derivative(self.z)
        
        # Compute gradients
        self.dweights = np.dot(self.input.T, delta)
        self.dbiases = np.sum(delta, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
        
        # Compute derivative of loss with respect to input
        return np.dot(delta, self.weights.T)
    
    def get_params(self) -> Dict:
        """Get parameters for saving/visualization"""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation_name,
            'weights': self.weights.tolist(),
            'biases': self.biases.tolist()
        }
    
    def set_params(self, params: Dict) -> None:
        """Set parameters from saved data"""
        self.weights = np.array(params['weights'])
        self.biases = np.array(params['biases'])


class NeuralNetwork:
    """Feedforward neural network implementation"""
    
    def __init__(self, input_size: int = 2, loss: str = 'mean_squared_error'):
        """
        Initialize a neural network
        
        Args:
            input_size: Number of input features
            loss: Loss function name
        """
        self.layers: List[Layer] = []
        self.input_size = input_size
        self.loss_name = loss
        self.loss_func, self.loss_derivative = LOSS_FUNCTIONS[loss]
        
        # Training history for visualization
        self.history = {
            'loss': [],
            'accuracy': []
        }
        
        # Current batch for visualization
        self.current_batch_X = None
        self.current_batch_y = None
        self.current_predictions = None
        
        # Training parameters
        self.learning_rate = 0.01
        self.batch_size = 32
        self.epochs = 100
        
    def add_layer(self, output_size: int, activation: str = 'sigmoid') -> None:
        """
        Add a layer to the network
        
        Args:
            output_size: Number of neurons in the layer
            activation: Activation function name
        """
        input_size = self.input_size if not self.layers else self.layers[-1].output_size
        self.layers.append(Layer(input_size, output_size, activation))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire network
        
        Args:
            X: Input data
            
        Returns:
            Network output
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        """
        Backward pass through the entire network
        
        Args:
            X: Input data
            y: Target data
            learning_rate: Learning rate
        """
        # Forward pass
        y_pred = self.forward(X)
        
        # Compute initial delta for output layer
        delta = self.loss_derivative(y, y_pred)
        
        # Backward pass through all layers
        for layer in reversed(self.layers):
            delta = layer.backward(delta, learning_rate)
    
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[float, float]:
        """
        Perform a single training step on a batch
        
        Args:
            X_batch: Batch of input data
            y_batch: Batch of target data
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Store current batch for visualization
        self.current_batch_X = X_batch
        self.current_batch_y = y_batch
        
        # Forward and backward pass
        y_pred = self.forward(X_batch)
        self.current_predictions = y_pred
        self.backward(X_batch, y_batch, self.learning_rate)
        
        # Calculate loss and accuracy
        loss = self.loss_func(y_batch, y_pred)
        
        # Calculate accuracy based on problem type
        if y_batch.shape[1] == 1:  # Binary classification
            accuracy = np.mean((y_pred > 0.5).astype(int) == y_batch)
        else:  # Multi-class classification
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            
        return loss, accuracy
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = None, batch_size: int = None, 
              learning_rate: float = None, callback=None) -> Dict:
        """
        Train the neural network
        
        Args:
            X: Input data
            y: Target data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            callback: Function to call after each batch
            
        Returns:
            Training history
        """
        # Update parameters if provided
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if learning_rate is not None:
            self.learning_rate = learning_rate
        
        n_samples = X.shape[0]
        
        # Reset history
        self.history = {
            'loss': [],
            'accuracy': []
        }
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Train in batches
            batch_losses = []
            batch_accuracies = []
            
            for i in range(0, n_samples, self.batch_size):
                end = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end]
                y_batch = y_shuffled[i:end]
                
                loss, accuracy = self.train_step(X_batch, y_batch)
                batch_losses.append(loss)
                batch_accuracies.append(accuracy)
                
                # Call callback if provided
                if callback:
                    callback(self, epoch, i // self.batch_size, loss, accuracy)
            
            # Compute epoch metrics
            epoch_loss = np.mean(batch_losses)
            epoch_accuracy = np.mean(batch_accuracies)
            
            # Update history
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_accuracy)
            
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        return self.forward(X)
    
    def save(self, filepath: str) -> None:
        """
        Save the neural network to a file
        
        Args:
            filepath: Path to save file
        """
        data = {
            'input_size': self.input_size,
            'loss': self.loss_name,
            'layers': [layer.get_params() for layer in self.layers],
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    def load(self, filepath: str) -> None:
        """
        Load a neural network from a file
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.input_size = data['input_size']
        self.loss_name = data['loss']
        self.loss_func, self.loss_derivative = LOSS_FUNCTIONS[self.loss_name]
        self.learning_rate = data['learning_rate']
        self.batch_size = data['batch_size']
        self.epochs = data['epochs']
        
        # Recreate layers
        self.layers = []
        for layer_data in data['layers']:
            layer = Layer(
                layer_data['input_size'],
                layer_data['output_size'],
                layer_data['activation']
            )
            layer.set_params(layer_data)
            self.layers.append(layer)
    
    def get_layer_activations(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Get activations for all layers given input X
        
        Args:
            X: Input data
            
        Returns:
            List of activations for each layer
        """
        activations = []
        output = X
        
        for layer in self.layers:
            output = layer.forward(output)
            activations.append(output)
            
        return activations

    def get_network_structure(self) -> List[Dict]:
        """
        Get network structure for visualization
        
        Returns:
            List of dictionaries with layer information
        """
        structure = []
        prev_size = self.input_size
        
        # Add input layer
        structure.append({
            'type': 'input',
            'size': self.input_size
        })
        
        # Add hidden and output layers
        for i, layer in enumerate(self.layers):
            structure.append({
                'type': 'output' if i == len(self.layers) - 1 else 'hidden',
                'size': layer.output_size,
                'activation': layer.activation_name,
                'weights_shape': layer.weights.shape,
                'biases_shape': layer.biases.shape
            })
            
        return structure 