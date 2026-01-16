# Multi-Layer Perceptron (MLP) for Digit Recognition in C

A raw implementation of a Feedforward Neural Network in C, built from scratch to recognize handwritten digits using the MNIST database. This project includes a companion Python tool for drawing custom digits to test the network's real-time prediction capabilities.

Unlike modern AI implementations that rely on high-level libraries (TensorFlow, PyTorch), this project implements the mathematical engine (matrix operations, backpropagation, and gradient descent) purely in C to demonstrate the low-level mechanics of machine learning.

## Implementation Overview

- **Architecture**: A fully connected Feedforward Neural Network (Multi-Layer Perceptron).
- **Learning Algorithm**: The network minimizes error using **Backpropagation** and **Stochastic Gradient Descent (SGD)**.
- **Activation & Loss**: Uses the **Sigmoid** activation function to normalize neuron output and **Mean Squared Error (MSE)** to calculate loss.
- **Data Handling**: Features a custom CSV parser to load and process the massive MNIST dataset (60,000+ images) efficiently in C.

## Prerequisites

- **C Compiler**: GCC (Recommended).
- **Python**: Required for the drawing tool (Dependencies: `pygame`, `numpy`, `opencv-python`).
- **Data**: MNIST CSV Dataset (Train and Test).
  - [Download MNIST in CSV format here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

## File Overview

- `main.c`: The core C source code containing the neural network engine and training logic.
- `Draw.py`: The Python script for drawing digits with your mouse.
- `network.txt`: Stores the learned weights and biases. **(Generated automatically after training)**.
- `mnist_train.csv`: The training dataset (You must download this).
- `my_drawing.csv`: Temporary file used to pass your drawing from Python to C.

## Compilation

1. Ensure `main.c`, `Draw.py`, and `mnist_train.csv` are in the **same directory**.
2. Open your terminal in that directory.
3. Compile using GCC:

```bash
gcc main.c -o network
```

(On Windows, this will create `network.exe`)

## Controls & Usage

### Step 1: Train the Network

1. Run the executable (`./network` or `network.exe`).
2. Select Option 1 (Train Network).
3. The program will train on 60,000 images. This takes ~250 seconds depending on your CPU.
4. **IMPORTANT**: When prompted "Save Network?", type `0` (Yes).
   - This generates the `network.txt` file. You only need to do this once! Future runs can load this file instantly.

### Step 2: Draw a Digit

1. Run the Python tool:

```bash
python Draw.py
```

2. Draw a digit (0-9) using the Left Mouse Button.
3. Press **S** to save (creates `my_drawing.csv`).
4. Press **C** to clear if you make a mistake.

### Step 3: Predict

1. Run the C executable again.
2. Select Option 2 (Load Network). It will read weights from `network.txt`.
3. Select Option 2 (Load 'my_drawing.csv').
4. The network will output its prediction.

## Customization

You can tweak the network topology (number of layers and neurons) directly in `main.c`.

*Note: Deeper networks may require longer training times.*

```c
// Example: 784 Inputs -> 128 Hidden -> 64 Hidden -> 10 Outputs
int number_neurons[] = {784, 128, 64, 10};        
net = create_Network(4, number_neurons);
```

## Results & Performance

- **Accuracy**: Achieved ~92.5% accuracy on the MNIST test set using a topology of `{784, 128, 64, 10}`.
- **Training Time**: Approximately 250 seconds for one epoch.
- **Real-World Testing**:
  - The network performs best on clearly drawn digits, specifically 1, 3, 4, 8, and 9.
  - **Limitation**: There is a slight drop in accuracy with custom hand-drawn inputs due to differences in preprocessing (downsampling/grayscale conversion) between the Python tool and the original MNIST dataset processing.

---

**Enjoy !**
