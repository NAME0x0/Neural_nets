import sys
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QComboBox, 
                            QSpinBox, QDoubleSpinBox, QTabWidget, QFileDialog,
                            QSlider, QGroupBox, QSplitter, QLineEdit, QMessageBox,
                            QGridLayout, QFormLayout, QTextEdit, QCheckBox, QTreeView,
                            QListWidget, QScrollArea, QStackedWidget, QTableWidget,
                            QTableWidgetItem, QHeaderView, QInputDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QFont

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import os

# Import custom modules
from models.neural_network import NeuralNetwork, ACTIVATION_FUNCTIONS, LOSS_FUNCTIONS
from datasets.dataset_handler import DatasetHandler
from visualization.network_visualizer import NetworkVisualizer


class MainWindow(QMainWindow):
    """Main window for neural network visualization application"""
    
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        
        # Set up basic window properties
        self.setWindowTitle("Neural Network Visualization Tool")
        self.setMinimumSize(1200, 800)
        
        # Initialize components
        self.init_components()
        self.setup_layout()
        self.setup_menu()
        self.setup_connections()
        
        # Initialize neural network and dataset handler
        self.neural_network = NeuralNetwork()
        self.dataset_handler = DatasetHandler()
        
        # Initialize state variables
        self.is_training = False
        self.training_timer = QTimer()
        self.training_timer.timeout.connect(self.training_step)
        
        # Show the window
        self.show()
    
    def init_components(self):
        """Initialize GUI components"""
        # Main structure
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create tab widgets
        self.network_tab = QWidget()
        self.visualization_tab = QWidget()
        self.training_tab = QWidget()
        self.datasets_tab = QWidget()
        self.analysis_tab = QWidget()
        
        # Add tabs to tab widget
        self.tabs.addTab(self.network_tab, "Network Architecture")
        self.tabs.addTab(self.datasets_tab, "Datasets")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.visualization_tab, "Visualization")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        
        # Placeholder for network visualizer
        self.network_visualizer = NetworkVisualizer()
        
        # Set central widget
        self.setCentralWidget(self.central_widget)
    
    def setup_layout(self):
        """Set up the layout structure"""
        # Add tabs to main layout
        self.main_layout.addWidget(self.tabs)
        
        # Setup individual tab layouts
        self.setup_network_tab()
        self.setup_datasets_tab()
        self.setup_training_tab()
        self.setup_visualization_tab()
        self.setup_analysis_tab()
    
    def setup_menu(self):
        """Set up the menu bar"""
        # Create menu bar
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        # New network action
        new_action = QAction("New Network", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_network)
        file_menu.addAction(new_action)
        
        # Load network action
        load_action = QAction("Load Network", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_network)
        file_menu.addAction(load_action)
        
        # Save network action
        save_action = QAction("Save Network", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_network)
        file_menu.addAction(save_action)
        
        # Separator
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_connections(self):
        """Set up signal-slot connections"""
        # Network tab connections
        self.add_layer_button.clicked.connect(self.add_layer)
        self.remove_layer_button.clicked.connect(self.remove_layer)
        self.update_network_button.clicked.connect(self.update_network)
        self.input_size_spinner.valueChanged.connect(self.update_input_size)
        self.loss_function_combo.currentTextChanged.connect(self.update_loss_function)
        
        # Layer table connections
        self.layers_table.itemChanged.connect(self.layer_property_changed)
    
    def setup_network_tab(self):
        """Set up the network architecture tab"""
        # Main layout for network tab
        network_layout = QHBoxLayout(self.network_tab)
        
        # Left side - Network configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Input size group
        input_group = QGroupBox("Input Configuration")
        input_layout = QFormLayout(input_group)
        
        self.input_size_spinner = QSpinBox()
        self.input_size_spinner.setRange(1, 1000)
        self.input_size_spinner.setValue(2)
        input_layout.addRow("Input Size:", self.input_size_spinner)
        
        # Loss function group
        loss_group = QGroupBox("Loss Function")
        loss_layout = QFormLayout(loss_group)
        
        self.loss_function_combo = QComboBox()
        self.loss_function_combo.addItems(list(LOSS_FUNCTIONS.keys()))
        loss_layout.addRow("Loss Function:", self.loss_function_combo)
        
        # Layer management group
        layers_group = QGroupBox("Layers")
        layers_layout = QVBoxLayout(layers_group)
        
        # Table for layers
        self.layers_table = QTableWidget(0, 3)
        self.layers_table.setHorizontalHeaderLabels(["Size", "Activation", "Description"])
        self.layers_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layers_layout.addWidget(self.layers_table)
        
        # Buttons for layer management
        layer_buttons_layout = QHBoxLayout()
        
        self.add_layer_button = QPushButton("Add Layer")
        self.remove_layer_button = QPushButton("Remove Layer")
        self.update_network_button = QPushButton("Update Network")
        
        layer_buttons_layout.addWidget(self.add_layer_button)
        layer_buttons_layout.addWidget(self.remove_layer_button)
        layers_layout.addLayout(layer_buttons_layout)
        
        # Add groups to left layout
        left_layout.addWidget(input_group)
        left_layout.addWidget(loss_group)
        left_layout.addWidget(layers_group)
        left_layout.addWidget(self.update_network_button)
        left_layout.addStretch()
        
        # Right side - Network visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Network visualization
        right_layout.addWidget(self.network_visualizer)
        
        # Add left and right panels to main layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        network_layout.addWidget(splitter)
    
    def setup_datasets_tab(self):
        """Set up the datasets tab"""
        # Main layout for datasets tab
        datasets_layout = QVBoxLayout(self.datasets_tab)
        
        # Top section - Dataset selection
        selection_group = QGroupBox("Dataset Selection")
        selection_layout = QHBoxLayout(selection_group)
        
        # Dataset combo box
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("Select a dataset...")
        selection_layout.addWidget(QLabel("Dataset:"))
        selection_layout.addWidget(self.dataset_combo, 1)
        
        # Load dataset button
        self.load_dataset_button = QPushButton("Load Dataset")
        selection_layout.addWidget(self.load_dataset_button)
        
        # Upload dataset button
        self.upload_dataset_button = QPushButton("Upload CSV Dataset")
        selection_layout.addWidget(self.upload_dataset_button)
        
        # Add selection group to main layout
        datasets_layout.addWidget(selection_group)
        
        # Middle section - Dataset info
        info_group = QGroupBox("Dataset Information")
        info_layout = QGridLayout(info_group)
        
        # Dataset info labels
        self.dataset_name_label = QLabel("Name: -")
        self.dataset_description_label = QLabel("Description: -")
        self.dataset_shape_label = QLabel("Shape: -")
        self.dataset_type_label = QLabel("Type: -")
        
        info_layout.addWidget(self.dataset_name_label, 0, 0)
        info_layout.addWidget(self.dataset_description_label, 1, 0)
        info_layout.addWidget(self.dataset_shape_label, 0, 1)
        info_layout.addWidget(self.dataset_type_label, 1, 1)
        
        # Add info group to main layout
        datasets_layout.addWidget(info_group)
        
        # Bottom section - Dataset preview
        preview_group = QGroupBox("Dataset Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        # Dataset preview table
        self.dataset_preview_table = QTableWidget(0, 0)
        preview_layout.addWidget(self.dataset_preview_table)
        
        # Add preview group to main layout
        datasets_layout.addWidget(preview_group)
        
        # Connect signals
        self.load_dataset_button.clicked.connect(self.load_dataset)
        self.upload_dataset_button.clicked.connect(self.upload_dataset)
        
        # Populate dataset combo box
        self.populate_dataset_combo()
    
    def setup_training_tab(self):
        """Set up the training tab"""
        # Main layout for training tab
        training_layout = QVBoxLayout(self.training_tab)
        
        # Top section - Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout(params_group)
        
        # Learning rate
        self.learning_rate_spinner = QDoubleSpinBox()
        self.learning_rate_spinner.setRange(0.0001, 1.0)
        self.learning_rate_spinner.setSingleStep(0.001)
        self.learning_rate_spinner.setValue(0.01)
        self.learning_rate_spinner.setDecimals(4)
        params_layout.addRow("Learning Rate:", self.learning_rate_spinner)
        
        # Batch size
        self.batch_size_spinner = QSpinBox()
        self.batch_size_spinner.setRange(1, 1000)
        self.batch_size_spinner.setValue(32)
        params_layout.addRow("Batch Size:", self.batch_size_spinner)
        
        # Epochs
        self.epochs_spinner = QSpinBox()
        self.epochs_spinner.setRange(1, 10000)
        self.epochs_spinner.setValue(100)
        params_layout.addRow("Epochs:", self.epochs_spinner)
        
        # Validation split
        self.validation_split_spinner = QDoubleSpinBox()
        self.validation_split_spinner.setRange(0.0, 0.5)
        self.validation_split_spinner.setSingleStep(0.05)
        self.validation_split_spinner.setValue(0.2)
        self.validation_split_spinner.setDecimals(2)
        params_layout.addRow("Validation Split:", self.validation_split_spinner)
        
        # Add params group to main layout
        training_layout.addWidget(params_group)
        
        # Middle section - Training controls
        controls_group = QGroupBox("Training Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Start/stop training button
        self.train_button = QPushButton("Start Training")
        controls_layout.addWidget(self.train_button)
        
        # Step training button
        self.step_button = QPushButton("Step")
        controls_layout.addWidget(self.step_button)
        
        # Reset training button
        self.reset_button = QPushButton("Reset")
        controls_layout.addWidget(self.reset_button)
        
        # Training progress
        self.progress_label = QLabel("Progress: 0%")
        controls_layout.addWidget(self.progress_label)
        
        # Training status
        self.status_label = QLabel("Status: Ready")
        controls_layout.addWidget(self.status_label)
        
        # Add controls group to main layout
        training_layout.addWidget(controls_group)
        
        # Bottom section - Training visualization
        viz_group = QGroupBox("Training Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Tab widget for different visualizations
        viz_tabs = QTabWidget()
        
        # Metrics tab
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        # Metrics plot placeholder
        self.metrics_visualizer = NetworkVisualizer()
        metrics_layout.addWidget(self.metrics_visualizer)
        
        # Add metrics tab to viz tabs
        viz_tabs.addTab(metrics_tab, "Metrics")
        
        # Weights tab
        weights_tab = QWidget()
        weights_layout = QVBoxLayout(weights_tab)
        
        # Weights plot placeholder
        self.weights_visualizer = NetworkVisualizer()
        weights_layout.addWidget(self.weights_visualizer)
        
        # Add weights tab to viz tabs
        viz_tabs.addTab(weights_tab, "Weights")
        
        # Add viz tabs to viz layout
        viz_layout.addWidget(viz_tabs)
        
        # Add viz group to main layout
        training_layout.addWidget(viz_group)
        
        # Connect signals
        self.train_button.clicked.connect(self.toggle_training)
        self.step_button.clicked.connect(self.training_step)
        self.reset_button.clicked.connect(self.reset_training)
    
    def setup_visualization_tab(self):
        """Set up the visualization tab"""
        # Main layout for visualization tab
        visualization_layout = QVBoxLayout(self.visualization_tab)
        
        # Top section - Visualization controls
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Visualization type combo box
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Network Architecture",
            "Data Flow",
            "Decision Boundary",
            "Feature Importance",
            "Activation Functions"
        ])
        controls_layout.addWidget(QLabel("Visualization Type:"))
        controls_layout.addWidget(self.viz_type_combo, 1)
        
        # Sample selection (for data flow)
        self.sample_combo = QComboBox()
        self.sample_combo.addItem("Random Sample")
        for i in range(1, 11):
            self.sample_combo.addItem(f"Sample {i}")
        self.sample_combo.setEnabled(False)  # Disabled until data flow viz is selected
        controls_layout.addWidget(QLabel("Sample:"))
        controls_layout.addWidget(self.sample_combo)
        
        # Visualization button
        self.visualize_button = QPushButton("Visualize")
        controls_layout.addWidget(self.visualize_button)
        
        # Add controls group to main layout
        visualization_layout.addWidget(controls_group)
        
        # Bottom section - Visualization display
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        # Visualization display
        self.viz_visualizer = NetworkVisualizer()
        viz_layout.addWidget(self.viz_visualizer)
        
        # Animation controls (for data flow)
        anim_controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.step_forward_button = QPushButton("Step Forward")
        self.step_backward_button = QPushButton("Step Backward")
        self.reset_anim_button = QPushButton("Reset")
        
        anim_controls_layout.addWidget(self.play_button)
        anim_controls_layout.addWidget(self.stop_button)
        anim_controls_layout.addWidget(self.step_forward_button)
        anim_controls_layout.addWidget(self.step_backward_button)
        anim_controls_layout.addWidget(self.reset_anim_button)
        
        # Initially disable animation controls
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.step_forward_button.setEnabled(False)
        self.step_backward_button.setEnabled(False)
        self.reset_anim_button.setEnabled(False)
        
        viz_layout.addLayout(anim_controls_layout)
        
        # Add viz group to main layout
        visualization_layout.addWidget(viz_group)
        
        # Connect signals
        self.viz_type_combo.currentTextChanged.connect(self.on_viz_type_changed)
        self.visualize_button.clicked.connect(self.update_visualization)
        self.play_button.clicked.connect(self.play_visualization)
        self.stop_button.clicked.connect(self.stop_visualization)
        self.step_forward_button.clicked.connect(self.step_forward_visualization)
        self.step_backward_button.clicked.connect(self.step_backward_visualization)
        self.reset_anim_button.clicked.connect(self.reset_visualization)
    
    def setup_analysis_tab(self):
        """Set up the analysis tab"""
        # Main layout for analysis tab
        analysis_layout = QVBoxLayout(self.analysis_tab)
        
        # Top section - Analysis controls
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Analysis type combo box
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Performance Metrics",
            "Weight Distribution",
            "Confusion Matrix",
            "Learning Curves",
            "Prediction Error"
        ])
        controls_layout.addWidget(QLabel("Analysis Type:"))
        controls_layout.addWidget(self.analysis_type_combo, 1)
        
        # Dataset selection
        self.analysis_dataset_combo = QComboBox()
        self.analysis_dataset_combo.addItems(["Training Set", "Test Set", "Validation Set"])
        controls_layout.addWidget(QLabel("Dataset:"))
        controls_layout.addWidget(self.analysis_dataset_combo)
        
        # Analysis button
        self.analyze_button = QPushButton("Analyze")
        controls_layout.addWidget(self.analyze_button)
        
        # Add controls group to main layout
        analysis_layout.addWidget(controls_group)
        
        # Bottom section - Analysis display
        analysis_group = QGroupBox("Analysis Results")
        analysis_display_layout = QVBoxLayout(analysis_group)
        
        # Analysis display
        self.analysis_visualizer = NetworkVisualizer()
        analysis_display_layout.addWidget(self.analysis_visualizer)
        
        # Text output for analysis results
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMaximumHeight(150)
        analysis_display_layout.addWidget(self.analysis_text)
        
        # Add analysis group to main layout
        analysis_layout.addWidget(analysis_group)
        
        # Connect signals
        self.analyze_button.clicked.connect(self.perform_analysis)
    
    def new_network(self):
        """Create a new neural network"""
        # Clear existing network
        self.neural_network = NeuralNetwork(input_size=self.input_size_spinner.value(),
                                           loss=self.loss_function_combo.currentText())
        
        # Clear layers table
        self.layers_table.setRowCount(0)
        
        # Update visualization
        self.update_network_visualization()
        
        # Show confirmation message
        QMessageBox.information(self, "New Network", "Created a new neural network.")
    
    def load_network(self):
        """Load a neural network from file"""
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Network", "", "JSON Files (*.json)")
        
        if filepath:
            try:
                # Load network
                self.neural_network.load(filepath)
                
                # Update UI
                self.update_ui_from_network()
                
                # Update visualization
                self.update_network_visualization()
                
                # Show confirmation message
                QMessageBox.information(self, "Load Network", f"Loaded network from {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load network: {str(e)}")
    
    def save_network(self):
        """Save the neural network to file"""
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Network", "", "JSON Files (*.json)")
        
        if filepath:
            try:
                # Add .json extension if not present
                if not filepath.endswith('.json'):
                    filepath += '.json'
                
                # Save network
                self.neural_network.save(filepath)
                
                # Show confirmation message
                QMessageBox.information(self, "Save Network", f"Saved network to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save network: {str(e)}")
    
    def add_layer(self):
        """Add a new layer to the network"""
        # Get current number of rows
        current_rows = self.layers_table.rowCount()
        
        # Add a new row
        self.layers_table.setRowCount(current_rows + 1)
        
        # Create a size spinbox
        size_spinner = QSpinBox()
        size_spinner.setRange(1, 1000)
        size_spinner.setValue(10)
        self.layers_table.setCellWidget(current_rows, 0, size_spinner)
        
        # Create an activation combo box
        activation_combo = QComboBox()
        activation_combo.addItems(list(ACTIVATION_FUNCTIONS.keys()))
        self.layers_table.setCellWidget(current_rows, 1, activation_combo)
        
        # Add a description
        description_item = QTableWidgetItem(f"Hidden Layer {current_rows + 1}")
        self.layers_table.setItem(current_rows, 2, description_item)
    
    def remove_layer(self):
        """Remove the last layer from the network"""
        # Get current number of rows
        current_rows = self.layers_table.rowCount()
        
        if current_rows > 0:
            # Remove the last row
            self.layers_table.setRowCount(current_rows - 1)
    
    def update_network(self):
        """Update the neural network model based on UI settings"""
        # Get input size and loss function
        input_size = self.input_size_spinner.value()
        loss_function = self.loss_function_combo.currentText()
        
        # Create a new neural network
        self.neural_network = NeuralNetwork(input_size=input_size, loss=loss_function)
        
        # Add layers
        for row in range(self.layers_table.rowCount()):
            # Get layer properties
            size_spinner = self.layers_table.cellWidget(row, 0)
            activation_combo = self.layers_table.cellWidget(row, 1)
            
            if size_spinner and activation_combo:
                size = size_spinner.value()
                activation = activation_combo.currentText()
                
                # Add layer to network
                self.neural_network.add_layer(size, activation)
        
        # Update visualization
        self.update_network_visualization()
        
        # Show confirmation message
        QMessageBox.information(self, "Update Network", "Neural network updated successfully.")
    
    def update_input_size(self, value):
        """Update input size of the network"""
        pass  # Will be updated when update_network is called
    
    def update_loss_function(self, value):
        """Update loss function of the network"""
        pass  # Will be updated when update_network is called
    
    def layer_property_changed(self, item):
        """Handle changes in layer properties"""
        pass  # Will be updated when update_network is called
    
    def update_network_visualization(self):
        """Update the network visualization"""
        if not self.neural_network.layers:
            return
        
        # Get network structure
        network_structure = self.neural_network.get_network_structure()
        
        # Get weights
        weights = [layer.weights for layer in self.neural_network.layers]
        
        # Update visualization
        self.network_visualizer.create_network_graph(network_structure, weights)
        self.network_visualizer.draw_network()
    
    def update_ui_from_network(self):
        """Update UI components based on the loaded network"""
        # Update input size
        self.input_size_spinner.setValue(self.neural_network.input_size)
        
        # Update loss function
        index = self.loss_function_combo.findText(self.neural_network.loss_name)
        if index >= 0:
            self.loss_function_combo.setCurrentIndex(index)
        
        # Clear and update layers table
        self.layers_table.setRowCount(0)
        
        for i, layer in enumerate(self.neural_network.layers):
            # Add a new row
            current_rows = self.layers_table.rowCount()
            self.layers_table.setRowCount(current_rows + 1)
            
            # Create a size spinbox
            size_spinner = QSpinBox()
            size_spinner.setRange(1, 1000)
            size_spinner.setValue(layer.output_size)
            self.layers_table.setCellWidget(current_rows, 0, size_spinner)
            
            # Create an activation combo box
            activation_combo = QComboBox()
            activation_combo.addItems(list(ACTIVATION_FUNCTIONS.keys()))
            index = activation_combo.findText(layer.activation_name)
            if index >= 0:
                activation_combo.setCurrentIndex(index)
            self.layers_table.setCellWidget(current_rows, 1, activation_combo)
            
            # Add a description
            description = "Output Layer" if i == len(self.neural_network.layers) - 1 else f"Hidden Layer {i + 1}"
            description_item = QTableWidgetItem(description)
            self.layers_table.setItem(current_rows, 2, description_item)
    
    def toggle_training(self):
        """Start or stop training"""
        if self.is_training:
            # Stop training
            self.is_training = False
            self.training_timer.stop()
            self.train_button.setText("Start Training")
            self.status_label.setText("Status: Paused")
        else:
            # Check if dataset is loaded
            if self.dataset_handler.X_train is None or self.dataset_handler.y_train is None:
                QMessageBox.warning(self, "Warning", "Please load a dataset first.")
                return
            
            # Check if network has layers
            if not self.neural_network.layers:
                QMessageBox.warning(self, "Warning", "Please create a network with at least one layer.")
                return
            
            # Start training
            self.is_training = True
            
            # Get training parameters
            learning_rate = self.learning_rate_spinner.value()
            batch_size = self.batch_size_spinner.value()
            epochs = self.epochs_spinner.value()
            
            # Set training parameters on neural network
            self.neural_network.learning_rate = learning_rate
            self.neural_network.batch_size = batch_size
            self.neural_network.epochs = epochs
            
            # Start timer for training steps
            self.training_timer.start(50)  # 50ms between steps
            
            # Update UI
            self.train_button.setText("Stop Training")
            self.status_label.setText("Status: Training")
            
            # Reset training history
            self.neural_network.history = {'loss': [], 'accuracy': []}
            
            # Update progress
            self.progress_label.setText("Progress: 0%")
    
    def training_step(self):
        """Perform a single training step"""
        if not self.is_training and self.sender() != self.step_button:
            return
        
        # Check if dataset is loaded
        if self.dataset_handler.X_train is None or self.dataset_handler.y_train is None:
            if self.is_training:
                self.toggle_training()  # Stop training
            QMessageBox.warning(self, "Warning", "No dataset loaded.")
            return
        
        try:
            # Get a batch of data
            X_batch, y_batch = self.dataset_handler.get_batch(self.neural_network.batch_size)
            
            # Perform training step
            loss, accuracy = self.neural_network.train_step(X_batch, y_batch)
            
            # Update history
            self.neural_network.history['loss'].append(loss)
            self.neural_network.history['accuracy'].append(accuracy)
            
            # Update metrics visualization
            self.metrics_visualizer.plot_training_metrics(self.neural_network.history)
            
            # Update weights visualization if appropriate
            if len(self.neural_network.history['loss']) % 10 == 0:
                weights = [layer.weights for layer in self.neural_network.layers]
                self.weights_visualizer.visualize_weights_distribution(weights)
            
            # Update progress
            epoch_count = len(self.neural_network.history['loss'])
            progress_pct = min(100, int(100 * epoch_count / self.neural_network.epochs))
            self.progress_label.setText(f"Progress: {progress_pct}%")
            
            # Update status
            self.status_label.setText(f"Status: Training (Loss: {loss:.4f}, Accuracy: {accuracy:.4f})")
            
            # Check if training is complete
            if epoch_count >= self.neural_network.epochs:
                self.toggle_training()  # Stop training
                QMessageBox.information(self, "Training Complete", 
                                      "Training has completed successfully.")
        except Exception as e:
            # Stop training on error
            if self.is_training:
                self.toggle_training()
            QMessageBox.critical(self, "Error", f"Training error: {str(e)}")
    
    def reset_training(self):
        """Reset training state"""
        # Stop training if in progress
        if self.is_training:
            self.toggle_training()
        
        # Reset neural network (keep structure but re-initialize weights)
        structure = [(layer.output_size, layer.activation_name) for layer in self.neural_network.layers]
        input_size = self.neural_network.input_size
        loss = self.neural_network.loss_name
        
        self.neural_network = NeuralNetwork(input_size=input_size, loss=loss)
        
        for size, activation in structure:
            self.neural_network.add_layer(size, activation)
        
        # Reset visualizations
        self.metrics_visualizer.clear()
        self.weights_visualizer.clear()
        
        # Update network visualization
        self.update_network_visualization()
        
        # Reset progress and status
        self.progress_label.setText("Progress: 0%")
        self.status_label.setText("Status: Ready")
        
        # Show confirmation
        QMessageBox.information(self, "Reset Training", "Training state has been reset.")
    
    def show_about(self):
        """Show the about dialog"""
        QMessageBox.about(
            self,
            "About Neural Network Visualization Tool",
            "Neural Network Visualization Tool\n\n"
            "An interactive educational application for visualizing and understanding neural networks.\n\n"
            "Version 1.0.0"
        )
    
    def populate_dataset_combo(self):
        """Populate dataset combobox with available datasets"""
        # Clear the combo box
        self.dataset_combo.clear()
        self.dataset_combo.addItem("Select a dataset...")
        
        # Get available datasets
        try:
            datasets = self.dataset_handler.get_available_datasets()
            self.dataset_combo.addItems(datasets)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to get available datasets: {str(e)}")
    
    def load_dataset(self):
        """Load the selected dataset"""
        selected_dataset = self.dataset_combo.currentText()
        
        if selected_dataset == "Select a dataset...":
            QMessageBox.warning(self, "Warning", "Please select a dataset.")
            return
        
        try:
            # Load dataset
            info = self.dataset_handler.load_dataset(selected_dataset)
            
            # Update dataset info
            self.update_dataset_info(info)
            
            # Update dataset preview
            self.update_dataset_preview()
            
            # Show confirmation message
            QMessageBox.information(self, "Load Dataset", f"Loaded dataset: {selected_dataset}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")
    
    def upload_dataset(self):
        """Upload a CSV dataset"""
        # Open file dialog
        filepath, _ = QFileDialog.getOpenFileName(self, "Upload CSV Dataset", "", "CSV Files (*.csv)")
        
        if not filepath:
            return
        
        # Get target column
        target_column, ok = QInputDialog.getText(
            self, "Target Column", "Enter the name of the target column:")
        
        if not ok or not target_column:
            return
        
        # Scaling and encoding options
        scale = QMessageBox.question(
            self, "Scale Features", 
            "Do you want to scale the features?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes
        
        one_hot_encode = QMessageBox.question(
            self, "One-Hot Encode Target", 
            "Do you want to one-hot encode the target?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes
        
        try:
            # Load dataset
            info = self.dataset_handler.load_from_csv(
                filepath, target_column, one_hot_encode, scale)
            
            # Update dataset info
            self.update_dataset_info(info)
            
            # Update dataset preview
            self.update_dataset_preview()
            
            # Refresh dataset combo
            self.populate_dataset_combo()
            
            # Show confirmation message
            QMessageBox.information(self, "Upload Dataset", f"Uploaded dataset from {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to upload dataset: {str(e)}")
    
    def update_dataset_info(self, info):
        """Update dataset information display"""
        self.dataset_name_label.setText(f"Name: {info.get('name', '-')}")
        self.dataset_description_label.setText(f"Description: {info.get('description', '-')}")
        
        # Create shape text with X_train, y_train shapes
        shape_text = f"Shape: X={info.get('X_train_shape', '-')}, y={info.get('y_train_shape', '-')}"
        self.dataset_shape_label.setText(shape_text)
        
        problem_type = info.get('problem_type', '-')
        self.dataset_type_label.setText(f"Type: {problem_type}")
    
    def update_dataset_preview(self):
        """Update dataset preview table"""
        # Get data from dataset handler
        if self.dataset_handler.X_train is None or self.dataset_handler.y_train is None:
            return
        
        # Get a small sample of the data
        X = self.dataset_handler.X_train[:10]
        y = self.dataset_handler.y_train[:10]
        
        # Setup table dimensions
        n_features = X.shape[1]
        n_targets = y.shape[1] if len(y.shape) > 1 else 1
        
        self.dataset_preview_table.setRowCount(X.shape[0])
        self.dataset_preview_table.setColumnCount(n_features + n_targets)
        
        # Set header labels
        feature_headers = [f"Feature {i+1}" for i in range(n_features)]
        
        if n_targets == 1:
            target_headers = ["Target"]
        else:
            target_headers = [f"Target {i+1}" for i in range(n_targets)]
        
        self.dataset_preview_table.setHorizontalHeaderLabels(feature_headers + target_headers)
        
        # Fill the table with data
        for row in range(X.shape[0]):
            # Add feature values
            for col in range(n_features):
                item = QTableWidgetItem(f"{X[row, col]:.4f}")
                self.dataset_preview_table.setItem(row, col, item)
            
            # Add target values
            if n_targets == 1:
                item = QTableWidgetItem(f"{y[row][0] if len(y.shape) > 1 else y[row]:.4f}")
                self.dataset_preview_table.setItem(row, n_features, item)
            else:
                for col in range(n_targets):
                    item = QTableWidgetItem(f"{y[row, col]:.4f}")
                    self.dataset_preview_table.setItem(row, n_features + col, item)
        
        # Resize columns to contents
        self.dataset_preview_table.resizeColumnsToContents()
    
    def on_viz_type_changed(self, viz_type):
        """Handle visualization type change"""
        # Enable/disable sample selection based on viz type
        if viz_type == "Data Flow":
            self.sample_combo.setEnabled(True)
        else:
            self.sample_combo.setEnabled(False)
        
        # Clear visualization
        self.viz_visualizer.clear()
        
        # Disable animation controls until data flow visualization is created
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.step_forward_button.setEnabled(False)
        self.step_backward_button.setEnabled(False)
        self.reset_anim_button.setEnabled(False)
    
    def update_visualization(self):
        """Update the visualization based on selected type"""
        # Get selected visualization type
        viz_type = self.viz_type_combo.currentText()
        
        # Clear existing visualization
        self.viz_visualizer.clear()
        
        try:
            if viz_type == "Network Architecture":
                self.visualize_network_architecture()
            elif viz_type == "Data Flow":
                self.visualize_data_flow()
            elif viz_type == "Decision Boundary":
                self.visualize_decision_boundary()
            elif viz_type == "Feature Importance":
                self.visualize_feature_importance()
            elif viz_type == "Activation Functions":
                self.visualize_activation_functions()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Visualization error: {str(e)}")
    
    def visualize_network_architecture(self):
        """Visualize the network architecture"""
        if not self.neural_network.layers:
            QMessageBox.warning(self, "Warning", "Please create a network with at least one layer.")
            return
        
        # Get network structure
        network_structure = self.neural_network.get_network_structure()
        
        # Get weights
        weights = [layer.weights for layer in self.neural_network.layers]
        
        # Update visualization
        self.viz_visualizer.create_network_graph(network_structure, weights)
        self.viz_visualizer.draw_network()
    
    def visualize_data_flow(self):
        """Visualize data flow through the network"""
        if not self.neural_network.layers:
            QMessageBox.warning(self, "Warning", "Please create a network with at least one layer.")
            return
        
        if self.dataset_handler.X_train is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return
        
        # Get sample index
        sample_idx = self.sample_combo.currentIndex() - 1  # -1 for "Random Sample"
        
        # Get data point
        if sample_idx == -1:
            # Random sample
            rand_idx = np.random.randint(0, len(self.dataset_handler.X_train))
            data_point = self.dataset_handler.X_train[rand_idx:rand_idx+1]
        else:
            # Specific sample
            data_point = self.dataset_handler.X_train[sample_idx:sample_idx+1]
        
        # Get network structure
        network_structure = self.neural_network.get_network_structure()
        
        # Create network graph
        self.viz_visualizer.create_network_graph(network_structure)
        
        # Visualize data flow
        self.viz_visualizer.highlight_data_flow(data_point, self.neural_network)
        
        # Enable animation controls
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.step_forward_button.setEnabled(True)
        self.step_backward_button.setEnabled(True)
        self.reset_anim_button.setEnabled(True)
    
    def visualize_decision_boundary(self):
        """Visualize decision boundary"""
        if not self.neural_network.layers:
            QMessageBox.warning(self, "Warning", "Please create a network with at least one layer.")
            return
        
        if self.dataset_handler.X_train is None or self.dataset_handler.y_train is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return
        
        # Check if data is 2D or can be reduced to 2D
        X_train = self.dataset_handler.X_train
        y_train = self.dataset_handler.y_train
        
        if X_train.shape[1] < 2:
            QMessageBox.warning(self, "Warning", "Dataset must have at least 2 features.")
            return
        
        # Use only the first two features if more than 2D
        feature_idx = (0, 1)
        
        # Visualize decision boundary
        self.viz_visualizer.visualize_decision_boundary(
            self.neural_network, X_train, y_train, feature_idx)
    
    def visualize_feature_importance(self):
        """Visualize feature importance"""
        if not self.neural_network.layers:
            QMessageBox.warning(self, "Warning", "Please create a network with at least one layer.")
            return
        
        # Get input size
        input_size = self.neural_network.input_size
        
        # Visualize feature importance
        self.viz_visualizer.visualize_feature_importance(
            self.neural_network, input_size)
    
    def visualize_activation_functions(self):
        """Visualize activation functions"""
        self.viz_visualizer.visualize_activation_functions()
    
    def play_visualization(self):
        """Play the visualization animation"""
        self.viz_visualizer.play_animation()
    
    def stop_visualization(self):
        """Stop the visualization animation"""
        self.viz_visualizer.stop_animation()
    
    def step_forward_visualization(self):
        """Step forward in the visualization animation"""
        self.viz_visualizer.step_animation_forward()
    
    def step_backward_visualization(self):
        """Step backward in the visualization animation"""
        self.viz_visualizer.step_animation_backward()
    
    def reset_visualization(self):
        """Reset the visualization animation"""
        self.viz_visualizer.reset_animation()
    
    def perform_analysis(self):
        """Perform analysis based on selected type"""
        # Get selected analysis type
        analysis_type = self.analysis_type_combo.currentText()
        
        # Get selected dataset
        dataset_type = self.analysis_dataset_combo.currentText()
        
        # Check if dataset is loaded
        if self.dataset_handler.X_train is None or self.dataset_handler.y_train is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return
        
        # Check if network has layers
        if not self.neural_network.layers:
            QMessageBox.warning(self, "Warning", "Please create a network with at least one layer.")
            return
        
        # Get dataset
        if dataset_type == "Training Set":
            X = self.dataset_handler.X_train
            y = self.dataset_handler.y_train
            dataset_name = "training"
        elif dataset_type == "Test Set":
            X = self.dataset_handler.X_test
            y = self.dataset_handler.y_test
            dataset_name = "test"
        else:  # Validation Set
            if self.dataset_handler.X_val is None or self.dataset_handler.y_val is None:
                QMessageBox.warning(self, "Warning", "No validation set available.")
                return
            X = self.dataset_handler.X_val
            y = self.dataset_handler.y_val
            dataset_name = "validation"
        
        # Clear existing analysis
        self.analysis_visualizer.clear()
        self.analysis_text.clear()
        
        try:
            if analysis_type == "Performance Metrics":
                self.analyze_performance_metrics(X, y, dataset_name)
            elif analysis_type == "Weight Distribution":
                self.analyze_weight_distribution()
            elif analysis_type == "Confusion Matrix":
                self.analyze_confusion_matrix(X, y, dataset_name)
            elif analysis_type == "Learning Curves":
                self.analyze_learning_curves()
            elif analysis_type == "Prediction Error":
                self.analyze_prediction_error(X, y, dataset_name)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis error: {str(e)}")
    
    def analyze_performance_metrics(self, X, y, dataset_name):
        """Analyze performance metrics"""
        # Make predictions
        y_pred = self.neural_network.predict(X)
        
        # Calculate metrics
        if y.shape[1] == 1:  # Binary classification
            y_true = y.flatten()
            y_pred_class = (y_pred > 0.5).astype(int).flatten()
            
            # Calculate accuracy
            accuracy = np.mean(y_true == y_pred_class)
            
            # Calculate precision and recall
            true_positives = np.sum((y_true == 1) & (y_pred_class == 1))
            false_positives = np.sum((y_true == 0) & (y_pred_class == 1))
            false_negatives = np.sum((y_true == 1) & (y_pred_class == 0))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add text results
            self.analysis_text.append(f"Performance Metrics on {dataset_name} set:")
            self.analysis_text.append(f"Accuracy: {accuracy:.4f}")
            self.analysis_text.append(f"Precision: {precision:.4f}")
            self.analysis_text.append(f"Recall: {recall:.4f}")
            self.analysis_text.append(f"F1 Score: {f1:.4f}")
            
            # Plot metrics
            metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
            
            # Create bar chart
            ax = self.analysis_visualizer.axes
            ax.bar(metrics.keys(), metrics.values(), color='#2196F3')
            ax.set_ylim(0, 1)
            ax.set_title(f"Performance Metrics ({dataset_name} set)")
            self.analysis_visualizer.draw()
        else:  # Multi-class classification
            y_true = np.argmax(y, axis=1)
            y_pred_class = np.argmax(y_pred, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(y_true == y_pred_class)
            
            # Calculate per-class precision and recall
            n_classes = y.shape[1]
            precision = np.zeros(n_classes)
            recall = np.zeros(n_classes)
            
            for i in range(n_classes):
                true_positives = np.sum((y_true == i) & (y_pred_class == i))
                false_positives = np.sum((y_true != i) & (y_pred_class == i))
                false_negatives = np.sum((y_true == i) & (y_pred_class != i))
                
                precision[i] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall[i] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Calculate macro-averaged F1 score
            f1 = np.zeros(n_classes)
            for i in range(n_classes):
                f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
            
            macro_f1 = np.mean(f1)
            
            # Add text results
            self.analysis_text.append(f"Performance Metrics on {dataset_name} set:")
            self.analysis_text.append(f"Accuracy: {accuracy:.4f}")
            self.analysis_text.append(f"Macro-averaged F1 Score: {macro_f1:.4f}")
            
            # Plot metrics
            metrics = {'Accuracy': accuracy, 'Macro F1': macro_f1}
            
            # Create bar chart
            ax = self.analysis_visualizer.axes
            ax.bar(metrics.keys(), metrics.values(), color='#2196F3')
            ax.set_ylim(0, 1)
            ax.set_title(f"Performance Metrics ({dataset_name} set)")
            self.analysis_visualizer.draw()
    
    def analyze_weight_distribution(self):
        """Analyze weight distribution"""
        # Get weights from all layers
        weights = [layer.weights for layer in self.neural_network.layers]
        
        # Visualize weight distribution
        self.analysis_visualizer.visualize_weights_distribution(weights)
        
        # Calculate weight statistics
        all_weights = np.concatenate([w.flatten() for w in weights])
        
        mean_weight = np.mean(all_weights)
        median_weight = np.median(all_weights)
        std_weight = np.std(all_weights)
        min_weight = np.min(all_weights)
        max_weight = np.max(all_weights)
        
        # Add text results
        self.analysis_text.append("Weight Distribution Statistics:")
        self.analysis_text.append(f"Mean: {mean_weight:.4f}")
        self.analysis_text.append(f"Median: {median_weight:.4f}")
        self.analysis_text.append(f"Standard Deviation: {std_weight:.4f}")
        self.analysis_text.append(f"Min: {min_weight:.4f}")
        self.analysis_text.append(f"Max: {max_weight:.4f}")
    
    def analyze_confusion_matrix(self, X, y, dataset_name):
        """Analyze confusion matrix"""
        # Make predictions
        y_pred = self.neural_network.predict(X)
        
        # Calculate confusion matrix
        if y.shape[1] == 1:  # Binary classification
            y_true = y.flatten()
            y_pred_class = (y_pred > 0.5).astype(int).flatten()
            
            # Calculate confusion matrix
            tn = np.sum((y_true == 0) & (y_pred_class == 0))
            fp = np.sum((y_true == 0) & (y_pred_class == 1))
            fn = np.sum((y_true == 1) & (y_pred_class == 0))
            tp = np.sum((y_true == 1) & (y_pred_class == 1))
            
            confusion_matrix = np.array([[tn, fp], [fn, tp]])
            
            # Add text results
            self.analysis_text.append(f"Confusion Matrix on {dataset_name} set:")
            self.analysis_text.append(f"True Negatives: {tn}")
            self.analysis_text.append(f"False Positives: {fp}")
            self.analysis_text.append(f"False Negatives: {fn}")
            self.analysis_text.append(f"True Positives: {tp}")
            
            # Plot confusion matrix
            ax = self.analysis_visualizer.axes
            im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f"Confusion Matrix ({dataset_name} set)")
            
            # Add labels and values
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
            
            # Add text to each cell
            thresh = confusion_matrix.max() / 2.
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if confusion_matrix[i, j] > thresh else "black")
            
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
            self.analysis_visualizer.fig.colorbar(im)
            self.analysis_visualizer.draw()
        else:  # Multi-class classification
            n_classes = y.shape[1]
            y_true = np.argmax(y, axis=1)
            y_pred_class = np.argmax(y_pred, axis=1)
            
            # Calculate confusion matrix
            confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
            for i in range(len(y_true)):
                confusion_matrix[y_true[i], y_pred_class[i]] += 1
            
            # Add text results
            self.analysis_text.append(f"Confusion Matrix on {dataset_name} set:")
            
            # Plot confusion matrix
            ax = self.analysis_visualizer.axes
            im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f"Confusion Matrix ({dataset_name} set)")
            
            # Add labels and values
            tick_marks = np.arange(n_classes)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels([f'Class {i}' for i in range(n_classes)])
            ax.set_yticklabels([f'Class {i}' for i in range(n_classes)])
            
            # Add text to each cell
            thresh = confusion_matrix.max() / 2.
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if confusion_matrix[i, j] > thresh else "black")
            
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
            self.analysis_visualizer.fig.colorbar(im)
            self.analysis_visualizer.draw()
    
    def analyze_learning_curves(self):
        """Analyze learning curves"""
        # Check if training history exists
        if not self.neural_network.history.get('loss'):
            QMessageBox.warning(self, "Warning", "No training history available. Please train the network first.")
            return
        
        # Plot learning curves
        self.analysis_visualizer.plot_training_metrics(self.neural_network.history)
        
        # Add text results
        history = self.neural_network.history
        epochs = len(history['loss'])
        
        self.analysis_text.append("Learning Curves Analysis:")
        self.analysis_text.append(f"Epochs: {epochs}")
        self.analysis_text.append(f"Initial Loss: {history['loss'][0]:.4f}")
        self.analysis_text.append(f"Final Loss: {history['loss'][-1]:.4f}")
        
        if 'accuracy' in history:
            self.analysis_text.append(f"Initial Accuracy: {history['accuracy'][0]:.4f}")
            self.analysis_text.append(f"Final Accuracy: {history['accuracy'][-1]:.4f}")
    
    def analyze_prediction_error(self, X, y, dataset_name):
        """Analyze prediction error"""
        # Make predictions
        y_pred = self.neural_network.predict(X)
        
        # Calculate errors
        if y.shape[1] == 1:  # Binary classification
            y_true = y.flatten()
            
            # Calculate errors
            errors = np.abs(y_true - y_pred.flatten())
            
            # Plot histogram of errors
            ax = self.analysis_visualizer.axes
            ax.hist(errors, bins=20, alpha=0.7, color='#2196F3')
            ax.set_title(f"Prediction Error Distribution ({dataset_name} set)")
            ax.set_xlabel('Absolute Error')
            ax.set_ylabel('Frequency')
            self.analysis_visualizer.draw()
            
            # Calculate error statistics
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)
            
            # Add text results
            self.analysis_text.append(f"Prediction Error Analysis on {dataset_name} set:")
            self.analysis_text.append(f"Mean Absolute Error: {mean_error:.4f}")
            self.analysis_text.append(f"Median Absolute Error: {median_error:.4f}")
            self.analysis_text.append(f"Standard Deviation: {std_error:.4f}")
            self.analysis_text.append(f"Maximum Error: {max_error:.4f}")
        else:  # Multi-class classification
            y_true = np.argmax(y, axis=1)
            y_pred_class = np.argmax(y_pred, axis=1)
            
            # Calculate errors (0 for correct, 1 for incorrect)
            errors = (y_true != y_pred_class).astype(int)
            
            # Count errors per class
            n_classes = y.shape[1]
            class_errors = np.zeros(n_classes)
            class_counts = np.zeros(n_classes)
            
            for i in range(len(y_true)):
                true_class = y_true[i]
                class_counts[true_class] += 1
                if errors[i] == 1:
                    class_errors[true_class] += 1
            
            # Calculate error rate per class
            error_rates = np.zeros(n_classes)
            for i in range(n_classes):
                if class_counts[i] > 0:
                    error_rates[i] = class_errors[i] / class_counts[i]
            
            # Plot bar chart of error rates per class
            ax = self.analysis_visualizer.axes
            x = np.arange(n_classes)
            ax.bar(x, error_rates, alpha=0.7, color='#2196F3')
            ax.set_title(f"Error Rate per Class ({dataset_name} set)")
            ax.set_xlabel('Class')
            ax.set_ylabel('Error Rate')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Class {i}' for i in range(n_classes)])
            self.analysis_visualizer.draw()
            
            # Add text results
            overall_error_rate = np.mean(errors)
            self.analysis_text.append(f"Prediction Error Analysis on {dataset_name} set:")
            self.analysis_text.append(f"Overall Error Rate: {overall_error_rate:.4f}")
            for i in range(n_classes):
                self.analysis_text.append(f"Class {i} Error Rate: {error_rates[i]:.4f} ({int(class_errors[i])}/{int(class_counts[i])})")
