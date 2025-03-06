#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow
import os
import numpy as np
import random

def main():
    """Main application entry point"""
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for consistent look on all platforms
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 