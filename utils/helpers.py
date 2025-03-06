import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import json
import os
import datetime

def format_array_for_display(arr: np.ndarray, precision: int = 4) -> str:
    """
    Format a numpy array for display
    
    Args:
        arr: Numpy array to format
        precision: Number of decimal places to show
        
    Returns:
        Formatted string representation
    """
    if arr.ndim == 1:
        return "[" + ", ".join([f"{x:.{precision}f}" for x in arr]) + "]"
    else:
        return "[" + ", ".join([format_array_for_display(x, precision) for x in arr]) + "]"

def get_activation_description(name: str) -> str:
    """
    Get description of an activation function
    
    Args:
        name: Name of the activation function
        
    Returns:
        Description of the activation function
    """
    descriptions = {
        'sigmoid': "Sigmoid: f(x) = 1 / (1 + e^(-x)). Range [0, 1]. Useful for binary classification.",
        'relu': "ReLU: f(x) = max(0, x). Range [0, inf). Fast to compute and helps solve vanishing gradient problem.",
        'tanh': "Tanh: f(x) = tanh(x). Range [-1, 1]. Similar to sigmoid but zero-centered.",
        'linear': "Linear: f(x) = x. Range (-inf, inf). No transformation. Useful for regression.",
        'softmax': "Softmax: f(x_i) = e^(x_i) / sum(e^(x_j)). Maps real values to probabilities. Useful for multi-class classification."
    }
    
    return descriptions.get(name, f"Unknown activation function: {name}")

def get_loss_description(name: str) -> str:
    """
    Get description of a loss function
    
    Args:
        name: Name of the loss function
        
    Returns:
        Description of the loss function
    """
    descriptions = {
        'mean_squared_error': "Mean Squared Error: Measures the average squared difference between predictions and actual values. Useful for regression.",
        'binary_cross_entropy': "Binary Cross Entropy: Measures the performance of a classification model whose output is a probability value between 0 and 1. Useful for binary classification.",
        'categorical_cross_entropy': "Categorical Cross Entropy: Measures the performance of a classification model whose output is a probability distribution over multiple classes. Useful for multi-class classification."
    }
    
    return descriptions.get(name, f"Unknown loss function: {name}")

def save_figure(fig: plt.Figure, filename: str, directory: str = "results") -> str:
    """
    Save a figure to a file
    
    Args:
        fig: Matplotlib figure to save
        filename: Name of the file
        directory: Directory to save the file in
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_with_timestamp = f"{filename}_{timestamp}.png"
    
    # Create full path
    path = os.path.join(directory, filename_with_timestamp)
    
    # Save figure
    fig.savefig(path, dpi=300, bbox_inches='tight')
    
    return path

def export_training_results(history: Dict, filepath: str) -> None:
    """
    Export training results to a JSON file
    
    Args:
        history: Training history
        filepath: Path to save the file
    """
    # Convert history to serializable format
    serializable_history = {}
    for key, values in history.items():
        serializable_history[key] = [float(x) for x in values]
    
    # Add timestamp
    serializable_history['timestamp'] = datetime.datetime.now().isoformat()
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=4)

def generate_experiment_name(model_description: str) -> str:
    """
    Generate a name for an experiment
    
    Args:
        model_description: Description of the model
        
    Returns:
        Experiment name
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_description}_{timestamp}" 