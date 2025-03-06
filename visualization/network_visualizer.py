import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt6.QtCore import Qt
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import colorsys

# Set plot style
plt.style.use('ggplot')


class NetworkVisualizer(FigureCanvasQTAgg):
    """Visualization component for neural networks using NetworkX and Matplotlib"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """
        Initialize the network visualizer
        
        Args:
            parent: Parent widget
            width: Width of the figure in inches
            height: Height of the figure in inches
            dpi: Dots per inch
        """
        # Create figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.tight_layout()
        
        # Initialize canvas
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initialize graph
        self.graph = nx.DiGraph()
        self.pos = {}
        
        # Create axes
        self.axes = self.fig.add_subplot(111)
        self.axes.set_axis_off()
        
        # Set up animation properties
        self.anim = None
        self.is_playing = False
        self.current_frame = 0
        
        # Store network data
        self.network = None
        self.data = None
        self.node_colors = {}
        self.edge_colors = {}
        self.node_sizes = {}
        self.label_offset = 0.1
        
        # Set up color maps
        self.neuron_cmap = plt.cm.coolwarm
        self.weight_cmap = plt.cm.coolwarm
        
        # Initialize animation frames
        self.frames = []
        
        # Configure interactions
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, 
                          QtWidgets.QSizePolicy.Policy.Expanding)
        
        # Use a dark background
        self.fig.patch.set_facecolor('#2E2E2E')
        self.axes.set_facecolor('#2E2E2E')
    
    def create_network_graph(self, network_structure: List[Dict], 
                           weights: Optional[List[np.ndarray]] = None) -> None:
        """
        Create a network graph from a network structure
        
        Args:
            network_structure: List of dictionaries with layer information
            weights: List of weight matrices (optional)
        """
        # Clear previous graph
        self.graph.clear()
        self.pos.clear()
        self.node_colors.clear()
        self.edge_colors.clear()
        self.node_sizes.clear()
        
        # Create nodes
        node_id = 0
        layer_nodes = []
        
        # Parse network structure
        for layer_idx, layer in enumerate(network_structure):
            layer_type = layer['type']
            layer_size = layer['size']
            layer_nodes_ids = []
            
            # Add nodes for this layer
            for i in range(layer_size):
                node_name = f"{layer_type[0]}{layer_idx}_{i}"
                self.graph.add_node(node_name, layer=layer_idx, neuron=i, type=layer_type)
                layer_nodes_ids.append(node_name)
                node_id += 1
            
            layer_nodes.append(layer_nodes_ids)
        
        # Create edges between nodes
        if weights is not None:
            # Use actual weights if provided
            for layer_idx in range(len(layer_nodes) - 1):
                source_nodes = layer_nodes[layer_idx]
                target_nodes = layer_nodes[layer_idx + 1]
                weight_matrix = weights[layer_idx]
                
                for i, source in enumerate(source_nodes):
                    for j, target in enumerate(target_nodes):
                        weight = weight_matrix[i, j]
                        # Only add edges for non-zero weights or if explicitly showing all connections
                        if weight != 0:
                            self.graph.add_edge(source, target, weight=weight)
        else:
            # Create default edges if no weights provided
            for layer_idx in range(len(layer_nodes) - 1):
                source_nodes = layer_nodes[layer_idx]
                target_nodes = layer_nodes[layer_idx + 1]
                
                for source in source_nodes:
                    for target in target_nodes:
                        # Add edge with default weight
                        self.graph.add_edge(source, target, weight=0.1)
        
        # Position nodes in layers
        self.pos = {}
        layers = max(data['layer'] for _, data in self.graph.nodes(data=True)) + 1
        
        for layer in range(layers):
            layer_nodes = [node for node, data in self.graph.nodes(data=True) if data['layer'] == layer]
            nodes_in_layer = len(layer_nodes)
            
            for i, node in enumerate(layer_nodes):
                # Position nodes in a grid layout
                x = layer / max(1, layers - 1)
                if nodes_in_layer > 1:
                    y = i / (nodes_in_layer - 1)
                else:
                    y = 0.5
                self.pos[node] = (x, y)
        
        # Set default node colors and sizes
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_type = node_data['type']
            
            # Set color based on node type
            if node_type == 'input':
                self.node_colors[node] = '#4CAF50'  # Green
            elif node_type == 'hidden':
                self.node_colors[node] = '#2196F3'  # Blue
            else:  # Output
                self.node_colors[node] = '#F44336'  # Red
            
            # Set size
            self.node_sizes[node] = 300
        
        # Set default edge colors
        for u, v in self.graph.edges():
            self.edge_colors[(u, v)] = '#CCCCCC'  # Light gray
        
        # Store the network structure
        self.network_structure = network_structure
    
    def update_node_values(self, activations: List[np.ndarray]) -> None:
        """
        Update node values based on activations
        
        Args:
            activations: List of activation arrays for each layer
        """
        # Update node colors based on activations
        max_activation = max([np.max(act) for act in activations]) if activations else 1
        min_activation = min([np.min(act) for act in activations]) if activations else 0
        
        # Normalize to range [0, 1]
        norm = plt.Normalize(min_activation, max_activation)
        
        # Update colors layer by layer
        for layer_idx, layer_activations in enumerate(activations):
            # Skip input layer
            if layer_idx + 1 >= len(self.network_structure):
                break
            
            layer_type = self.network_structure[layer_idx + 1]['type']
            layer_nodes = [node for node, data in self.graph.nodes(data=True) 
                          if data['layer'] == layer_idx + 1]
            
            for i, node in enumerate(layer_nodes):
                if i < layer_activations.shape[1]:
                    activation = layer_activations[0, i]
                    color = self.neuron_cmap(norm(activation))
                    self.node_colors[node] = color
        
        # Redraw
        self.draw_network()
    
    def update_edge_weights(self, weights: List[np.ndarray]) -> None:
        """
        Update edge weights and colors
        
        Args:
            weights: List of weight matrices
        """
        # Flatten all weights to find global min/max
        all_weights = np.concatenate([w.flatten() for w in weights])
        max_weight = np.max(np.abs(all_weights))
        
        # Normalize to range [-1, 1]
        norm = plt.Normalize(-max_weight, max_weight)
        
        # Update edge colors based on weights
        for layer_idx in range(len(weights)):
            source_layer = self.network_structure[layer_idx]['type']
            target_layer = self.network_structure[layer_idx + 1]['type']
            
            source_nodes = [node for node, data in self.graph.nodes(data=True) 
                           if data['layer'] == layer_idx]
            target_nodes = [node for node, data in self.graph.nodes(data=True) 
                           if data['layer'] == layer_idx + 1]
            
            for i, source in enumerate(source_nodes):
                for j, target in enumerate(target_nodes):
                    if (source, target) in self.graph.edges():
                        if i < weights[layer_idx].shape[0] and j < weights[layer_idx].shape[1]:
                            weight = weights[layer_idx][i, j]
                            # Update edge weight
                            self.graph[source][target]['weight'] = weight
                            # Update edge color
                            edge_color = self.weight_cmap(norm(weight))
                            self.edge_colors[(source, target)] = edge_color
        
        # Redraw
        self.draw_network()
    
    def draw_network(self) -> None:
        """Draw the network graph"""
        # Clear the figure
        self.axes.clear()
        
        # Extract node colors and sizes
        node_colors = [self.node_colors.get(node, '#CCCCCC') for node in self.graph.nodes()]
        node_sizes = [self.node_sizes.get(node, 300) for node in self.graph.nodes()]
        
        # Extract edge colors and widths
        edge_colors = []
        edge_widths = []
        
        for u, v in self.graph.edges():
            edge_colors.append(self.edge_colors.get((u, v), '#CCCCCC'))
            
            # Scale edge width based on absolute weight
            weight = abs(self.graph[u][v].get('weight', 0.1))
            edge_widths.append(max(0.5, min(3, weight * 3)))
        
        # Draw the graph
        nx.draw_networkx_nodes(
            self.graph, 
            self.pos, 
            ax=self.axes,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(
            self.graph, 
            self.pos, 
            ax=self.axes,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            arrowsize=10,
            arrowstyle='->'
        )
        
        # Add node labels for input and output layers
        labels = {}
        for node, data in self.graph.nodes(data=True):
            if data['type'] in ['input', 'output']:
                labels[node] = f"{data['neuron'] + 1}"
        
        nx.draw_networkx_labels(
            self.graph, 
            {node: (pos[0], pos[1] + self.label_offset) for node, pos in self.pos.items()}, 
            labels=labels,
            font_size=8,
            font_color='white'
        )
        
        # Set axis limits
        self.axes.set_xlim(-0.1, 1.1)
        self.axes.set_ylim(-0.1, 1.1)
        
        # Turn off axis
        self.axes.set_axis_off()
        
        # Update canvas
        self.fig.tight_layout()
        self.draw()
    
    def add_animation_frame(self) -> None:
        """Add current network state as an animation frame"""
        # Store current network state
        frame = {
            'node_colors': self.node_colors.copy(),
            'edge_colors': self.edge_colors.copy(),
            'node_sizes': self.node_sizes.copy(),
        }
        
        self.frames.append(frame)
    
    def clear_animation(self) -> None:
        """Clear animation frames"""
        if self.anim is not None:
            self.anim.event_source.stop()
            self.anim = None
        
        self.frames.clear()
        self.is_playing = False
        self.current_frame = 0
    
    def play_animation(self, interval: int = 100) -> None:
        """
        Play animation of network changes
        
        Args:
            interval: Time between frames in milliseconds
        """
        # Stop any existing animation
        if self.anim is not None:
            self.anim.event_source.stop()
        
        if not self.frames:
            return
        
        self.is_playing = True
        self.current_frame = 0
        
        def update(frame_idx):
            if frame_idx >= len(self.frames):
                return
            
            frame = self.frames[frame_idx]
            self.node_colors = frame['node_colors']
            self.edge_colors = frame['edge_colors']
            self.node_sizes = frame['node_sizes']
            self.draw_network()
            self.current_frame = frame_idx
            
            return []
        
        self.anim = animation.FuncAnimation(
            self.fig, 
            update,
            frames=len(self.frames),
            interval=interval,
            blit=True,
            repeat=True
        )
        
        self.draw()
    
    def stop_animation(self) -> None:
        """Stop animation playback"""
        if self.anim is not None:
            self.anim.event_source.stop()
            self.is_playing = False
    
    def step_animation_forward(self) -> None:
        """Step animation forward by one frame"""
        if not self.frames:
            return
        
        next_frame = (self.current_frame + 1) % len(self.frames)
        frame = self.frames[next_frame]
        self.node_colors = frame['node_colors']
        self.edge_colors = frame['edge_colors']
        self.node_sizes = frame['node_sizes']
        self.draw_network()
        self.current_frame = next_frame
    
    def step_animation_backward(self) -> None:
        """Step animation backward by one frame"""
        if not self.frames:
            return
        
        prev_frame = (self.current_frame - 1) % len(self.frames)
        frame = self.frames[prev_frame]
        self.node_colors = frame['node_colors']
        self.edge_colors = frame['edge_colors']
        self.node_sizes = frame['node_sizes']
        self.draw_network()
        self.current_frame = prev_frame
    
    def reset_animation(self) -> None:
        """Reset animation to first frame"""
        if not self.frames:
            return
        
        frame = self.frames[0]
        self.node_colors = frame['node_colors']
        self.edge_colors = frame['edge_colors']
        self.node_sizes = frame['node_sizes']
        self.draw_network()
        self.current_frame = 0
    
    def highlight_node(self, node_id: str, highlight_color: str = '#FFA000') -> None:
        """
        Highlight a specific node
        
        Args:
            node_id: ID of the node to highlight
            highlight_color: Color to use for highlighting
        """
        if node_id in self.graph.nodes():
            original_color = self.node_colors.get(node_id)
            self.node_colors[node_id] = highlight_color
            self.node_sizes[node_id] = self.node_sizes.get(node_id, 300) * 1.5
            self.draw_network()
            
            # Return original values for restoring
            return original_color, self.node_sizes.get(node_id, 300) / 1.5
        
        return None, None
    
    def highlight_data_flow(self, data_point: np.ndarray, network, sample_idx: int = 0) -> None:
        """
        Visualize data flow through the network for a specific data point
        
        Args:
            data_point: Input data point
            network: Neural network object
            sample_idx: Index of the sample in the batch
        """
        # Clear animation
        self.clear_animation()
        
        # Create initial frame
        self.add_animation_frame()
        
        # Get layer activations
        activations = []
        output = data_point
        
        for layer in network.layers:
            output = layer.forward(output)
            activations.append(output)
        
        # Update node values layer by layer
        for layer_idx in range(len(activations)):
            # Skip input layer (already set)
            if layer_idx + 1 >= len(self.network_structure):
                break
            
            layer_data = self.network_structure[layer_idx + 1]
            layer_type = layer_data['type']
            layer_nodes = [node for node, data in self.graph.nodes(data=True) 
                          if data['layer'] == layer_idx + 1]
            
            for i, node in enumerate(layer_nodes):
                if i < activations[layer_idx].shape[1]:
                    act_value = activations[layer_idx][sample_idx, i]
                    
                    # Calculate color based on activation value
                    norm = plt.Normalize(0, 1)
                    color = self.neuron_cmap(norm(act_value))
                    
                    # Update node color
                    self.node_colors[node] = color
                    
                    # Update node size based on activation
                    self.node_sizes[node] = 300 * (0.5 + act_value)
            
            # Add frame after updating each layer
            self.add_animation_frame()
        
        # Draw the initial state
        self.draw_network()
    
    def plot_training_metrics(self, history: Dict, ax=None) -> None:
        """
        Plot training metrics
        
        Args:
            history: Dictionary of training metrics
            ax: Matplotlib axis to plot on (optional)
        """
        if ax is None:
            # Clear the figure
            self.axes.clear()
            ax = self.axes
        
        # Plot loss
        if 'loss' in history:
            epochs = range(1, len(history['loss']) + 1)
            ax.plot(epochs, history['loss'], 'b-', label='Loss')
        
        # Plot accuracy if available
        if 'accuracy' in history:
            epochs = range(1, len(history['accuracy']) + 1)
            ax.plot(epochs, history['accuracy'], 'r-', label='Accuracy')
        
        ax.set_title('Training Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Update canvas
        self.fig.tight_layout()
        self.draw()
    
    def visualize_weights_distribution(self, weights: List[np.ndarray], ax=None) -> None:
        """
        Visualize weight distribution
        
        Args:
            weights: List of weight matrices
            ax: Matplotlib axis to plot on (optional)
        """
        if ax is None:
            # Clear the figure
            self.axes.clear()
            ax = self.axes
        
        # Flatten all weights
        all_weights = np.concatenate([w.flatten() for w in weights])
        
        # Plot histogram
        ax.hist(all_weights, bins=50, color='#2196F3', alpha=0.7)
        ax.set_title('Weight Distribution')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        
        # Add vertical line for mean
        mean_weight = np.mean(all_weights)
        ax.axvline(mean_weight, color='r', linestyle='--', label=f'Mean: {mean_weight:.4f}')
        
        # Add vertical line for median
        median_weight = np.median(all_weights)
        ax.axvline(median_weight, color='g', linestyle='-.', label=f'Median: {median_weight:.4f}')
        
        ax.legend()
        
        # Update canvas
        self.fig.tight_layout()
        self.draw()
    
    def visualize_decision_boundary(self, network, X: np.ndarray, y: np.ndarray, 
                                  feature_idx: Tuple[int, int] = (0, 1), ax=None) -> None:
        """
        Visualize decision boundary (for 2D data)
        
        Args:
            network: Neural network object
            X: Input data
            y: Target data
            feature_idx: Indices of features to plot
            ax: Matplotlib axis to plot on (optional)
        """
        if ax is None:
            # Clear the figure
            self.axes.clear()
            ax = self.axes
        
        if X.shape[1] < 2:
            ax.text(0.5, 0.5, "Cannot plot decision boundary for data with less than 2 features",
                   ha='center', va='center')
            return
        
        # Extract the two features
        X_plot = X[:, feature_idx]
        
        # Create a mesh grid
        h = 0.01  # Step size
        x_min, x_max = X_plot[:, 0].min() - 0.1, X_plot[:, 0].max() + 0.1
        y_min, y_max = X_plot[:, 1].min() - 0.1, X_plot[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # Create input data for prediction
        X_mesh = np.c_[xx.ravel(), yy.ravel()]
        
        # Add zeros for other features
        if X.shape[1] > 2:
            X_mesh_full = np.zeros((X_mesh.shape[0], X.shape[1]))
            X_mesh_full[:, feature_idx[0]] = X_mesh[:, 0]
            X_mesh_full[:, feature_idx[1]] = X_mesh[:, 1]
            X_mesh = X_mesh_full
        
        # Make predictions
        Z = network.predict(X_mesh)
        
        # Convert to class predictions
        if Z.shape[1] == 1:  # Binary classification
            Z = (Z > 0.5).astype(int)
        else:  # Multi-class classification
            Z = np.argmax(Z, axis=1)
        
        # Reshape result to match the grid
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        
        # Plot data points
        if y.shape[1] == 1:  # Binary classification
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y.flatten(), cmap=plt.cm.coolwarm, 
                              edgecolors='k', s=40)
        else:  # Multi-class classification
            scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=np.argmax(y, axis=1), 
                              cmap=plt.cm.coolwarm, edgecolors='k', s=40)
        
        ax.set_title('Decision Boundary')
        ax.set_xlabel(f'Feature {feature_idx[0]}')
        ax.set_ylabel(f'Feature {feature_idx[1]}')
        
        # Add legend
        if y.shape[1] <= 10:  # Only show legend for few classes
            if y.shape[1] == 1:  # Binary classification
                classes = [0, 1]
            else:  # Multi-class classification
                classes = list(range(y.shape[1]))
            
            legend = ax.legend(*scatter.legend_elements(), title="Classes")
            ax.add_artist(legend)
        
        # Update canvas
        self.fig.tight_layout()
        self.draw()
    
    def visualize_feature_importance(self, network, input_size: int, ax=None) -> None:
        """
        Visualize feature importance based on weights
        
        Args:
            network: Neural network object
            input_size: Number of input features
            ax: Matplotlib axis to plot on (optional)
        """
        if ax is None:
            # Clear the figure
            self.axes.clear()
            ax = self.axes
        
        if not network.layers:
            ax.text(0.5, 0.5, "No layers in the network", ha='center', va='center')
            return
        
        # Get weights of the first layer
        first_layer_weights = network.layers[0].weights
        
        # Calculate importance as the sum of absolute weights for each feature
        importance = np.sum(np.abs(first_layer_weights), axis=1)
        
        # Normalize importance
        importance = importance / np.sum(importance)
        
        # Plot feature importance
        feature_names = [f"Feature {i+1}" for i in range(input_size)]
        indices = np.argsort(importance)
        
        # Plot horizontal bar chart
        bars = ax.barh(range(len(indices)), importance[indices], align='center', color='#2196F3')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title('Feature Importance')
        ax.set_xlabel('Relative Importance')
        
        # Update canvas
        self.fig.tight_layout()
        self.draw()
    
    def visualize_activation_functions(self, ax=None) -> None:
        """
        Visualize common activation functions
        
        Args:
            ax: Matplotlib axis to plot on (optional)
        """
        if ax is None:
            # Clear the figure
            self.axes.clear()
            ax = self.axes
        
        # Generate x values
        x = np.linspace(-5, 5, 1000)
        
        # Plot sigmoid
        from models.neural_network import sigmoid
        ax.plot(x, sigmoid(x), 'b-', label='Sigmoid')
        
        # Plot ReLU
        from models.neural_network import relu
        ax.plot(x, relu(x), 'r-', label='ReLU')
        
        # Plot tanh
        from models.neural_network import tanh
        ax.plot(x, tanh(x), 'g-', label='Tanh')
        
        # Plot linear
        from models.neural_network import linear
        ax.plot(x, linear(x), 'k-', label='Linear')
        
        ax.set_title('Activation Functions')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True)
        
        # Update canvas
        self.fig.tight_layout()
        self.draw()
    
    def clear(self) -> None:
        """Clear the visualization"""
        self.axes.clear()
        self.axes.set_axis_off()
        self.draw()


# Import at the end to avoid circular imports
from PyQt6 import QtWidgets 