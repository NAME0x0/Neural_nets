# 🧠 Neural Network Visualization Tool

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.13+-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> An interactive educational application that brings neural networks to life through visualization, making AI concepts accessible to everyone.

## 📋 Table of Contents

- [What is a Neural Network?](#what-is-a-neural-network)
- [Why This Tool?](#why-this-tool)
- [✨ Features](#-features)
- [🚀 Getting Started](#-getting-started)
- [📱 Using the Application](#-using-the-application)
- [🔍 Detailed GUI Navigation Guide](#-detailed-gui-navigation-guide)
- [📚 Practical Use Cases](#-practical-use-cases)
- [🏃‍♂️ Running Example Scripts](#-running-example-scripts)
- [❓ Common Issues and Solutions](#-common-issues-and-solutions)
- [📘 Learning More](#-learning-more)
- [📁 Project Structure](#-project-structure)
- [📄 License](#-license)

---

## What is a Neural Network?

Neural networks are a type of computer program inspired by how the human brain works. They're used in many technologies you use every day, like:

- 🔊 Voice assistants that understand your speech
- 📸 Photo apps that recognize faces
- 🛒 Shopping sites that recommend products you might like
- 🚗 Cars that can drive themselves

Think of a neural network as a series of connected "neurons" organized in layers. Each neuron takes information, processes it, and passes it to the next layer. The network "learns" by adjusting connections between neurons based on examples, similar to how you might learn to recognize cats after seeing many pictures of cats.

**In simple terms:** A neural network is like a student learning to identify animals. At first, they might confuse a dog with a cat, but after seeing many examples and being corrected, they get better and better at telling them apart.

---

## Why This Tool?

Understanding how neural networks work can be challenging because they often seem like mysterious "black boxes." This tool helps by:

1. **Making the invisible visible**: You can see how information flows through the network and how it changes at each step
2. **Learning by doing**: Instead of just reading about neural networks, you can build and train your own
3. **Experimenting safely**: You can change settings and immediately see the results without needing to write code

This tool is perfect for:

- 🎓 Students learning about AI and machine learning
- 👨‍🏫 Teachers demonstrating neural network concepts
- 🔍 Hobbyists curious about how AI works
- 👩‍💼 Professionals wanting to understand the technology changing their industry

---

## ✨ Features

- **🏗️ Customizable Neural Network Architecture**: Build your own neural network by adding layers, choosing the number of neurons in each layer, and selecting different activation functions - no coding required!

- **👁️ Real-time Visualization**: Watch your neural network learn in real-time with dynamic visualizations that show how data flows through the network and how weights change during training

- **🎮 Interactive Training Controls**: Control the training process with intuitive pause/play buttons and step through the learning process one iteration at a time to understand how neural networks learn

- **📊 Built-in and Custom Datasets**: Start with beginner-friendly datasets like XOR (simple logic problems), Iris (flower classification), and MNIST (handwritten digit recognition), or upload your own data to experiment with

- **🎛️ Adjustable Learning Parameters**: Experiment with different settings like learning rate (how quickly the network learns), batch size (how many examples to learn from at once), and epochs (how many times to review the data) using simple sliders

- **💾 Save and Load Your Work**: Save your neural network designs and training progress to continue later or share with others, and load existing configurations to build upon previous work

---

## 🚀 Getting Started

### Requirements

You'll need:

- A computer running Windows, Mac, or Linux
- Python 3.13 or newer installed on your computer
  - If you don't have Python installed, visit [python.org](https://www.python.org/downloads/) and download the latest version for your operating system

### Installation: Step by Step

1. **Download the code**:
   - Click the green "Code" button on the repository page
   - Select "Download ZIP"
   - Extract the ZIP file to a folder on your computer

2. **Open a terminal or command prompt**:
   - On Windows: Search for "Command Prompt" or "PowerShell" in the Start menu
   - On Mac: Open the "Terminal" application from Applications > Utilities
   - On Linux: Open your terminal application

3. **Navigate to the folder**:

   ```bash
   # Replace "path/to/folder" with the actual location where you extracted the files
   cd path/to/Neural_nets
   ```

4. **Set up a virtual environment** (this keeps the project's packages separate from your system):

   ```bash
   # Create the virtual environment
   python -m venv venv
   
   # Activate it (use the appropriate command for your system)
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

   You'll know it's activated when you see `(venv)` at the beginning of your command line.

5. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   This might take a few minutes as it downloads and installs all necessary components.

---

## 📱 Using the Application

### Starting the Main Application

To launch the application with the full graphical interface:

```bash
python main.py
```

When the application opens, you'll see several tabs:

- **Network Architecture**: Design your neural network here by adding layers
- **Datasets**: Load example datasets or your own data
- **Training**: Configure and run the training process
- **Visualization**: See your network in action with different visualization types
- **Analysis**: Analyze your network's performance after training

### Quick Start Guide: Your First Neural Network

Here's how to create and train your first neural network in 5 minutes:

1. **Build a simple network**:
   - In the Network Architecture tab, add 1-2 hidden layers
   - Set the first layer to 4 neurons with "ReLU" activation
   - Set the output layer based on your task (1 neuron for yes/no problems, multiple for classification)

2. **Train on the XOR problem**:
   - XOR is a classic problem where the network learns a simple logic rule:
     - If both inputs are the same (0,0 or 1,1), output 0
     - If inputs are different (0,1 or 1,0), output 1
   - Load the XOR dataset and start training to see how the network learns this pattern

3. **Visualize the learning process**:
   - Switch to the Visualization tab during training
   - Watch how the network's internal connections (weights) change as it learns
   - See how data flows through your network

4. **Analyze the Results**:
   - Go to the Analysis tab
   - Select "Performance Metrics"
   - See if your network correctly learned the XOR pattern

5. **Experiment with Complexity**:
   - Go back to Network Architecture
   - Add more neurons or layers
   - Retrain and observe how the decision boundaries become more complex

---

## 🏃‍♂️ Running Example Scripts

The project includes pre-made examples that demonstrate neural networks in action:

### XOR Example

This example shows a neural network learning the XOR logic operation (explained above):

```bash
python examples/xor_example.py
```

**What to expect:** The program will train a neural network and show you the final predictions. You should see that it correctly predicts the XOR pattern (0,0→0, 0,1→1, 1,0→1, 1,1→0).

**Real-world analogy:** This is like teaching a computer to understand that:

- If both light switches are in the same position (both up or both down), the light is off
- If they're in different positions, the light is on

### Iris Flower Classification Example

This example shows how neural networks can classify different types of flowers:

```bash
python examples/iris_example.py
```

**What to expect:** The program will train a neural network to distinguish between three types of iris flowers based on measurements of their petals and sepals. It will show you:

- Information about the dataset
- The network's structure
- Training progress
- Final accuracy (how many flowers it correctly identifies)
- Example predictions for different flowers

**Real-world analogy:** This is similar to how AI might help a botanist identify plant species, a doctor diagnose diseases based on symptoms, or a bank determine credit risk based on financial data.

### Direct XOR Implementation

This shows an alternative way to implement the XOR problem:

```bash
python examples/xor_direct.py
```

**What to expect:** Similar to the first XOR example, but with more detailed output showing how the error decreases during training. It also creates a graph showing how the network improves over time.

---

## 🔍 Detailed GUI Navigation Guide

### 1. Network Architecture Tab

This is where you'll design your neural network:

- **Input Configuration**: Located at the top-left, this is where you set:
  - **Input Size**: The number of input features your data has (e.g., 2 for XOR, 4 for Iris)

- **Loss Function**: Below the input configuration, select the type of loss function:
  - **Mean Squared Error**: Good for regression problems (predicting numbers)
  - **Binary Cross Entropy**: Good for yes/no problems
  - **Categorical Cross Entropy**: Good for multi-class problems (like identifying flower types)

- **Layers Table**: The central component of this tab:
  - **Add Layer Button**: Click to add a new layer to your network
  - **Remove Layer Button**: Click to remove the selected layer
  - **Size Column**: Set how many neurons are in each layer
  - **Activation Column**: Select the activation function (ReLU, Sigmoid, Tanh, etc.)
  - **Description Column**: Automatically shows information about the layer

- **Update Network Button**: After making changes to your network design, click this to update the visualization

- **Visualization Area**: The right side of the tab shows a visual representation of your network's structure

**Example:** Building a network for an image recognition task

- Set input size to 784 (for 28x28 pixel images)
- Add a hidden layer with 128 neurons using ReLU activation
- Add another hidden layer with 64 neurons using ReLU activation
- Add an output layer with 10 neurons (for 10 digit classes) using Softmax activation

### 2. Datasets Tab

Here you'll load and explore data to train your network:

- **Dataset Selection**: At the top, you'll find:
  - **Dataset Dropdown**: Select from built-in datasets like XOR, Iris, or MNIST
  - **Load Dataset Button**: Click to load the selected dataset
  - **Upload CSV Dataset Button**: Click to upload your own data from a CSV file

- **Dataset Information**: The middle section shows details about the loaded dataset:
  - Name, description, input and output shapes, etc.

- **Data Preview**: The bottom section displays:
  - A preview of the actual data points
  - Visual representations like scatter plots or images
  - Statistics about the data distribution

**Example:** Loading the IRIS dataset

- Select "IRIS" from the dataset dropdown
- Click "Load Dataset"
- The information panel shows you have 150 samples with 4 features each
- The preview shows measurements of sepal length, sepal width, petal length, and petal width
- The visualization shows how the three species cluster based on these measurements

### 3. Training Tab

This is where the learning happens:

- **Training Parameters**: On the left side, configure:
  - **Learning Rate**: Controls how quickly the network learns (lower is more stable but slower)
  - **Batch Size**: How many examples to learn from at once
  - **Epochs**: How many times to go through the entire dataset
  - **Other Parameters**: Depending on the problem

- **Training Controls**: Below the parameters:
  - **Start/Pause Button**: Begin or pause the training process
  - **Step Button**: Move forward one training step at a time
  - **Reset Button**: Clear the current training progress

- **Training Metrics**: On the right side:
  - **Loss Graph**: Shows how the error decreases over time
  - **Accuracy Graph**: Shows how the network's accuracy improves
  - **Current Values**: Displays the latest loss and accuracy values

**Example:** Training a network for XOR

- Set learning rate to 0.1 (moderate speed)
- Set batch size to 4 (all our examples)
- Set epochs to 1000 (enough to learn this simple problem)
- Click Start and watch the loss graph steadily decrease from around 0.25 to near 0
- The accuracy graph should increase from around 50% to nearly 100%

### 4. Visualization Tab

This tab lets you see inside the "black box":

- **Visualization Type**: At the top, select what to visualize:
  - **Network Architecture**: The structure of your network
  - **Weight Changes**: How connection strengths change during training
  - **Activation Values**: How data transforms as it moves through layers
  - **Decision Boundaries**: For classification problems, how the network divides the input space

- **Visualization Controls**: Below the type selection:
  - **Play/Pause Button**: Animate the visualization
  - **Step Forward/Backward**: Move through training steps manually
  - **Speed Control**: Adjust animation speed

- **Visualization Display**: The main area shows the selected visualization

**Example:** Visualizing decision boundaries

- Create a simple network for binary classification
- Train it using a 2D dataset (like a simplified version of XOR)
- Select "Decision Boundaries" visualization
- The display shows the input space with two colors representing the two classes
- A curved line shows where the network "decides" to separate the classes
- As training progresses, this line adjusts to better separate the data points

### 5. Analysis Tab

After training, analyze your network's performance:

- **Analysis Options**: On the left, select what to analyze:
  - **Performance Metrics**: Accuracy, precision, recall, etc.
  - **Weight Distribution**: Histogram of weight values
  - **Confusion Matrix**: For classification tasks, shows prediction patterns
  - **Learning Curves**: How metrics changed during training

- **Analysis Display**: The right side shows the selected analysis
  - Visual representations like charts and matrices
  - Textual explanations of what the metrics mean

**Example:** Analyzing MNIST digit recognition performance

- After training a network on MNIST, go to the Analysis tab
- Select "Confusion Matrix"
- The display shows a 10x10 grid
- Bright spots along the diagonal indicate correct predictions
- Off-diagonal spots show where the network confused one digit for another
- You might notice it frequently confuses 4s and 9s, which look similar

---

## 📚 Practical Use Cases

Here are detailed examples of how you might use this tool in different scenarios:

### Use Case 1: Learning the Basics of Neural Networks

**Ideal for:** Students, beginners, or anyone curious about neural networks

**Scenario:** Imagine you're taking an online course on AI and want to understand how neural networks really work.

**Steps:**

1. **Start Simple**: Begin with the XOR problem
   - Go to the Network Architecture tab
   - Set input size to 2
   - Add a hidden layer with 4 neurons using Sigmoid activation
   - Add an output layer with 1 neuron using Sigmoid activation
   - Click "Update Network" to see the visualization

2. **Load the Data**:
   - Switch to the Datasets tab
   - Select "XOR" from the dropdown
   - Click "Load Dataset"
   - Examine the dataset information - you'll see it's a simple pattern:

     ```text
     Input: [0,0] → Output: 0
     Input: [0,1] → Output: 1
     Input: [1,0] → Output: 1
     Input: [1,1] → Output: 0
     ```

3. **Train Step by Step**:
   - Go to the Training tab
   - Set a learning rate of 0.1
   - Set batch size to 4 and epochs to 1000
   - Instead of clicking "Start Training," use the "Step" button repeatedly
   - Watch how the loss and accuracy change with each step
   - Notice how the network initially guesses randomly, then gradually improves

4. **Visualize the Learning Process**:
   - Switch to the Visualization tab
   - Select "Weight Changes" visualization type
   - Use step controls to move back and forth through training
   - Observe how the weights change as the network learns
   - At first, the changes are large; later, they become more subtle refinements

5. **Analyze the Results**:
   - Go to the Analysis tab
   - Select "Performance Metrics"
   - See if your network correctly learned the XOR pattern
   - Try entering test inputs like [0,0] and [1,0] to see the network's predictions

**What you'll learn:** How a neural network starts with random guesses and gradually improves its predictions by adjusting its internal connections, much like how a child learns a new skill through practice and feedback.

### Use Case 2: Teaching Neural Network Concepts

**Ideal for:** Teachers, presenters, workshop leaders

**Scenario:** You're leading a workshop for high school students on AI basics and want to give them an intuitive understanding of neural networks.

**Steps:**

1. **Prepare Multiple Networks**:
   - Create a simple network with just one hidden layer
   - Save it using File > Save Network (call it "simple.nn")
   - Create a complex network with multiple hidden layers
   - Save it as well (call it "complex.nn")

2. **Demonstrate Layer Effects**:
   - Load the simple network
   - Train it while showing the Visualization tab
   - Load the complex network
   - Train it the same way
   - Compare the learning speed and final accuracy
   - Point out how the simple network learns faster but may have lower accuracy

3. **Show Different Activation Functions**:
   - Create another network but change the activation to ReLU
   - Train it and compare performance to Sigmoid
   - Use the visualization tab to show how ReLU and Sigmoid activate differently
   - Explain that ReLU is like a light switch (on/off) while Sigmoid is like a dimmer switch

4. **Illustrate Overfitting**:
   - Use the Iris dataset (has more features)
   - Train a very large network (many neurons) for many epochs
   - Use the Analysis tab to show how it performs perfectly on training data but worse on test data
   - Compare to a simpler network that generalizes better
   - Relate this to memorization vs. understanding in human learning

5. **Interactive Exercise**: Have students predict:
   - What happens if you increase the learning rate?
   - What happens if you remove all hidden layers?
   - What happens if you add too many neurons?
   - Let them test their predictions with the tool

**What they'll learn:** How neural network design choices affect learning, performance, and generalization ability, with visual evidence that makes abstract concepts concrete.

### Use Case 3: Experimenting with Your Own Data

**Ideal for:** Researchers, data enthusiasts, project creators

**Scenario:** You're curious whether a neural network could predict house prices based on features like square footage, number of bedrooms, etc.

**Steps:**

1. **Prepare Your CSV Data**:
   - Create a CSV file with your real estate data
   - Include columns for square footage, bedrooms, bathrooms, neighborhood, etc.
   - Include a column for house prices
   - Make sure to clean the data (remove missing values, etc.)

2. **Import the Data**:
   - Go to the Datasets tab
   - Click "Upload CSV Dataset"
   - Browse to your file and select it
   - In the dialog that appears, select "house_price" as the target column
   - Choose to normalize the data (scales all values to similar ranges)
   - Review the dataset information

3. **Design an Appropriate Network**:
   - Go to the Network Architecture tab
   - Set the input size to match your data's features (e.g., 5 for sq_ft, bedrooms, bathrooms, age, neighborhood)
   - Add a hidden layer with 10 neurons using ReLU activation
   - Add another hidden layer with 5 neurons using ReLU
   - Set the output layer to 1 neuron (price prediction) with Linear activation
   - Choose Mean Squared Error as your loss function (good for regression)

4. **Experiment with Parameters**:
   - In the Training tab, try different learning rates (start with 0.01)
   - Adjust batch size (try 32) and epochs (try 100 to start)
   - Train multiple times with different settings
   - Use the Analysis tab to compare results
   - Look especially at the Mean Absolute Error metric to see how close your predictions are

5. **Refine and Export**:
   - Once you have a good model, save it (File > Save Network)
   - Use the model to make predictions on new data
   - Try removing less important features to simplify the model
   - Try adding more neurons or layers to see if accuracy improves
   - Export the results for use in other applications

**What you'll discover:** Whether neural networks can effectively predict house prices based on your data, which features are most important, and how complex a network needs to be for this type of prediction task.

### Use Case 4: Understanding Classification Boundaries

**Ideal for:** Students learning about machine learning decision making

**Scenario:** You want to understand how neural networks make decisions to classify data points.

**Steps:**

1. **Create a Simple Classifier**:
   - Use the Network Architecture tab to create a network with:
     - 2 input features (for easy visualization)
     - 1 hidden layer with 4-8 neurons
     - Output layer appropriate for your classes (e.g., 3 neurons for Iris species)

2. **Load a Classification Dataset**:
   - Go to the Datasets tab
   - Load the Iris dataset but select only 2 features (e.g., petal length and width)
   - The scatter plot will show points clustered by species

3. **Train the Classifier**:
   - In the Training tab, set:
     - Learning rate: 0.01
     - Batch size: 16
     - Epochs: 200
   - Start the training process
   - Watch the accuracy improve over time

4. **Visualize Decision Boundaries**:
   - Switch to the Visualization tab
   - Select "Decision Boundaries" visualization
   - Watch how the colored regions (representing different classes) develop
   - Pause at different points to see how the network's decisions evolve
   - Notice how the boundaries start random, then gradually form to separate the classes

5. **Experiment with Complexity**:
   - Go back to Network Architecture
   - Add more neurons (e.g., increase from 4 to 16)
   - Retrain and observe how the decision boundaries become more complex
   - Try adding another hidden layer
   - Notice how the boundaries can now form more complex shapes to separate the data

**What you'll understand:** How neural networks create decision boundaries that separate different classes, how these boundaries become more complex with larger networks, and why more complex isn't always better (sometimes simple boundaries generalize better).

---

## ❓ Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **"Command not found" errors** | Make sure you've activated the virtual environment as described in step 4 of installation. You should see `(venv)` at the beginning of your command line. |
| **Import errors** | Ensure you've installed all requirements correctly with `pip install -r requirements.txt`. Try reinstalling if problems persist. |
| **Display issues** | This application requires a graphical interface. If you're using a remote server, you may need to set up X forwarding or use a desktop environment. |
| **Performance problems** | Training large networks can be slow on some computers. Try reducing the network size or using smaller datasets. You can also lower the number of epochs. |
| **"No module named 'PyQt5'"** | Run `pip install PyQt5` in your activated virtual environment. |
| **Application freezes during training** | Use a smaller batch size or reduce the network complexity. You can also try the "Step" button instead of continuous training. |
| **Dataset won't load** | Make sure your CSV file is properly formatted with consistent delimiters and no missing values in important columns. |
| **Out of memory errors** | Reduce batch size, network size, or dataset size. Close other applications to free up memory. |

---

## 📘 Learning More

If you're interested in learning more about neural networks:

- **In-App Resources**:
  - The "Network Architecture" tab includes descriptions of different activation functions
  - The "Analysis" tab provides insights into how well your network is performing
  - Try modifying the example scripts to see how different parameters affect learning

- **Recommended Next Steps**:
  - After mastering the XOR problem, try MNIST digit recognition
  - Experiment with different network architectures for the same problem
  - Import your own datasets to solve real-world problems

- **Community and Support**:
  - Join AI and machine learning forums to share your experiments
  - Contribute to this project by submitting bug reports or feature requests
  - Share your created networks with colleagues or classmates

---

## 📁 Project Structure

- `main.py`: Application entry point
- `models/`: Neural network implementation
- `visualization/`: Network visualization components
- `gui/`: PyQt GUI components
- `datasets/`: Dataset handlers and sample data
- `utils/`: Utility functions
- `examples/`: Example scripts demonstrating neural network functionality

---

## 📄 License

[MIT License](LICENSE)
