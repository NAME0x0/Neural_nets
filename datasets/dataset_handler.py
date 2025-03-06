import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import sqlite3
import tempfile


class DatasetHandler:
    """Handles dataset loading, preprocessing, and storage"""
    
    def __init__(self):
        """Initialize dataset handler"""
        self.db_path = os.path.join(tempfile.gettempdir(), "neural_net_datasets.db")
        self._init_database()
        
        # Available datasets
        self.available_datasets = {
            'xor': self.load_xor,
            'iris': self.load_iris,
            'mnist_small': self.load_mnist_small
        }
        
        # Current dataset
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.dataset_info = {}
        
    def _init_database(self):
        """Initialize the database for storing datasets"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create datasets table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            description TEXT,
            input_shape TEXT,
            output_shape TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create dataset_data table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_data (
            id INTEGER PRIMARY KEY,
            dataset_id INTEGER,
            data_type TEXT,  -- 'X_train', 'y_train', etc.
            data BLOB,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_dataset(self, dataset_name: str, test_size: float = 0.2, 
                    val_size: float = 0.1, random_state: int = 42) -> Dict:
        """
        Load a dataset by name
        
        Args:
            dataset_name: Name of the dataset
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with dataset information
        """
        if dataset_name in self.available_datasets:
            X, y, info = self.available_datasets[dataset_name]()
            return self.process_data(X, y, info, test_size, val_size, random_state)
        else:
            # Try to load from database
            return self.load_from_database(dataset_name, test_size, val_size, random_state)
    
    def process_data(self, X: np.ndarray, y: np.ndarray, info: Dict,
                    test_size: float = 0.2, val_size: float = 0.1, 
                    random_state: int = 42) -> Dict:
        """
        Process data by splitting and scaling
        
        Args:
            X: Input data
            y: Target data
            info: Dataset information
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with dataset information
        """
        # Split data into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        if val_size > 0:
            val_proportion = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_proportion, random_state=random_state
            )
        else:
            X_val, y_val = None, None
        
        # Scale data if needed
        if info.get('scale', False):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            if X_val is not None:
                X_val = scaler.transform(X_val)
        
        # Store data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        
        # Update dataset info with shapes
        info.update({
            'X_train_shape': X_train.shape,
            'y_train_shape': y_train.shape,
            'X_test_shape': X_test.shape,
            'y_test_shape': y_test.shape,
        })
        
        if X_val is not None:
            info.update({
                'X_val_shape': X_val.shape,
                'y_val_shape': y_val.shape,
            })
        
        self.dataset_info = info
        return info
    
    def load_xor(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load XOR dataset
        
        Returns:
            Tuple of (X, y, info)
        """
        # XOR dataset
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Dataset info
        info = {
            'name': 'xor',
            'description': 'XOR logical operation dataset',
            'input_shape': (2,),
            'output_shape': (1,),
            'problem_type': 'binary_classification',
            'scale': False
        }
        
        return X, y, info
    
    def load_iris(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load Iris dataset
        
        Returns:
            Tuple of (X, y, info)
        """
        from sklearn.datasets import load_iris
        
        # Load dataset
        iris = load_iris()
        X = iris.data
        y = iris.target.reshape(-1, 1)
        
        # One-hot encode targets
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y)
        
        # Dataset info
        info = {
            'name': 'iris',
            'description': 'Iris flower dataset',
            'input_shape': (4,),
            'output_shape': (3,),
            'feature_names': iris.feature_names,
            'target_names': iris.target_names,
            'problem_type': 'multi_classification',
            'scale': True
        }
        
        return X, y, info
    
    def load_mnist_small(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load a small subset of MNIST dataset
        
        Returns:
            Tuple of (X, y, info)
        """
        from sklearn.datasets import fetch_openml
        
        # Load a subset of MNIST
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.astype('float32').values[:1000] / 255.0
        y = mnist.target.astype('int').values[:1000].reshape(-1, 1)
        
        # One-hot encode targets
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y)
        
        # Dataset info
        info = {
            'name': 'mnist_small',
            'description': 'Small subset of MNIST handwritten digits',
            'input_shape': (784,),
            'output_shape': (10,),
            'problem_type': 'multi_classification',
            'scale': False
        }
        
        return X, y, info
    
    def load_from_csv(self, filepath: str, target_column: str, 
                    one_hot_encode: bool = True, scale: bool = True) -> Dict:
        """
        Load dataset from CSV file
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target column
            one_hot_encode: Whether to one-hot encode target
            scale: Whether to scale features
            
        Returns:
            Dictionary with dataset information
        """
        # Load data
        data = pd.read_csv(filepath)
        
        # Extract features and target
        y = data[target_column].values.reshape(-1, 1)
        X = data.drop(columns=[target_column]).values
        
        # One-hot encode target if needed
        if one_hot_encode and len(np.unique(y)) > 2:
            encoder = OneHotEncoder(sparse_output=False)
            y = encoder.fit_transform(y)
            problem_type = 'multi_classification'
        elif one_hot_encode and len(np.unique(y)) == 2:
            problem_type = 'binary_classification'
        else:
            problem_type = 'regression'
        
        # Dataset info
        info = {
            'name': os.path.basename(filepath),
            'description': f'Dataset loaded from {filepath}',
            'input_shape': (X.shape[1],),
            'output_shape': (y.shape[1] if y.ndim > 1 else 1,),
            'feature_names': data.drop(columns=[target_column]).columns.tolist(),
            'target_name': target_column,
            'problem_type': problem_type,
            'scale': scale
        }
        
        # Add to database
        self.save_to_database(X, y, info)
        
        return self.process_data(X, y, info)
    
    def save_to_database(self, X: np.ndarray, y: np.ndarray, info: Dict) -> None:
        """
        Save dataset to database
        
        Args:
            X: Input data
            y: Target data
            info: Dataset information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if dataset already exists
        cursor.execute("SELECT id FROM datasets WHERE name = ?", (info['name'],))
        result = cursor.fetchone()
        
        if result:
            # Update existing dataset
            dataset_id = result[0]
            cursor.execute("""
            UPDATE datasets 
            SET description = ?, input_shape = ?, output_shape = ?
            WHERE id = ?
            """, (
                info['description'], 
                str(info['input_shape']), 
                str(info['output_shape']), 
                dataset_id
            ))
            
            # Delete old data
            cursor.execute("DELETE FROM dataset_data WHERE dataset_id = ?", (dataset_id,))
        else:
            # Insert new dataset
            cursor.execute("""
            INSERT INTO datasets (name, description, input_shape, output_shape)
            VALUES (?, ?, ?, ?)
            """, (
                info['name'], 
                info['description'], 
                str(info['input_shape']), 
                str(info['output_shape'])
            ))
            dataset_id = cursor.lastrowid
        
        # Save data
        cursor.execute("""
        INSERT INTO dataset_data (dataset_id, data_type, data)
        VALUES (?, ?, ?)
        """, (dataset_id, 'X', X.tobytes()))
        
        cursor.execute("""
        INSERT INTO dataset_data (dataset_id, data_type, data)
        VALUES (?, ?, ?)
        """, (dataset_id, 'y', y.tobytes()))
        
        # Save info as JSON
        import json
        cursor.execute("""
        INSERT INTO dataset_data (dataset_id, data_type, data)
        VALUES (?, ?, ?)
        """, (dataset_id, 'info', json.dumps(info).encode()))
        
        conn.commit()
        conn.close()
    
    def load_from_database(self, dataset_name: str, test_size: float = 0.2, 
                         val_size: float = 0.1, random_state: int = 42) -> Dict:
        """
        Load dataset from database
        
        Args:
            dataset_name: Name of the dataset
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with dataset information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if dataset exists
        cursor.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise ValueError(f"Dataset '{dataset_name}' not found in database")
        
        dataset_id = result[0]
        
        # Load X
        cursor.execute("""
        SELECT data FROM dataset_data 
        WHERE dataset_id = ? AND data_type = ?
        """, (dataset_id, 'X'))
        X_bytes = cursor.fetchone()[0]
        
        # Load y
        cursor.execute("""
        SELECT data FROM dataset_data 
        WHERE dataset_id = ? AND data_type = ?
        """, (dataset_id, 'y'))
        y_bytes = cursor.fetchone()[0]
        
        # Load info
        cursor.execute("""
        SELECT data FROM dataset_data 
        WHERE dataset_id = ? AND data_type = ?
        """, (dataset_id, 'info'))
        info_bytes = cursor.fetchone()[0]
        
        conn.close()
        
        # Convert bytes to numpy arrays
        import json
        info = json.loads(info_bytes.decode())
        
        X_shape = eval(info['input_shape'])
        if isinstance(X_shape, tuple) and len(X_shape) == 1:
            X_shape = (-1, X_shape[0])
        
        y_shape = eval(info['output_shape'])
        if isinstance(y_shape, tuple) and len(y_shape) == 1:
            y_shape = (-1, y_shape[0])
        
        X = np.frombuffer(X_bytes).reshape(X_shape)
        y = np.frombuffer(y_bytes).reshape(y_shape)
        
        return self.process_data(X, y, info, test_size, val_size, random_state)
    
    def get_batch(self, batch_size: int = 32, dataset: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a random batch of data
        
        Args:
            batch_size: Batch size
            dataset: Dataset to get batch from ('train', 'test', or 'val')
            
        Returns:
            Tuple of (X_batch, y_batch)
        """
        if dataset == 'train':
            X, y = self.X_train, self.y_train
        elif dataset == 'test':
            X, y = self.X_test, self.y_test
        elif dataset == 'val':
            X, y = self.X_val, self.y_val
        else:
            raise ValueError(f"Dataset '{dataset}' not recognized")
        
        if X is None or y is None:
            raise ValueError(f"Dataset '{dataset}' not loaded")
        
        # Get random indices
        indices = np.random.choice(X.shape[0], size=min(batch_size, X.shape[0]), replace=False)
        
        return X[indices], y[indices]
    
    def get_all_data(self) -> Dict[str, np.ndarray]:
        """
        Get all loaded data
        
        Returns:
            Dictionary with all data
        """
        data = {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
        
        if self.X_val is not None and self.y_val is not None:
            data.update({
                'X_val': self.X_val,
                'y_val': self.y_val
            })
            
        return data
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the current dataset
        
        Returns:
            Dictionary with dataset information
        """
        return self.dataset_info
    
    def get_available_datasets(self) -> List[str]:
        """
        Get list of available datasets
        
        Returns:
            List of dataset names
        """
        # Get built-in datasets
        builtin_datasets = list(self.available_datasets.keys())
        
        # Get datasets from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM datasets")
        db_datasets = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Combine and remove duplicates
        all_datasets = builtin_datasets + [d for d in db_datasets if d not in builtin_datasets]
        
        return all_datasets 