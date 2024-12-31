# Neural Network Class

A customizable implementation of a neural network from scratch without using machine learning libraries.  
This project showcases the design of a Neural Network capable of forward propagation, backpropagation, training, and testing.

## Features
- Fully implemented in Python using only NumPy and Matplotlib.
- Supports multi-layer architectures.
- Sigmoid activation function.
- Training uses Mean Squared Error (MSE): 1/2(y_pred - y_ref)^2.
- Visualizes error evolution through epochs.
- XOR problem demo included.

## How to Use

### Prerequisites
Ensure you have Python installed with the following libraries:
- **NumPy**: For numerical computations.
- **Matplotlib**: For plotting training error.

### Initialize a Neural Network
Create a neural network instance using:
`nn = NeuralNetwork(layer_sizes=[input_size, hidden_size, output_size], learning_rate=lr)`
- `input_size`: Number of input neurons (e.g., 2 for XOR).
- `hidden_size`: Number of neurons in the hidden layer(s). You can have multiple layers.
- `output_size`: Number of output neurons (e.g., 1 for XOR).
- `lr`: Learning rate for weight and bias updates.

### Train the Neural Network
Train your model with:
`nn.train(input_data, output_data, epochs=15000)`

### Test the Neural Network
Test your trained model using:
`nn.test(input_set, output_set)`

## Example: XOR Problem
This project includes a demonstration of the XOR problem:
`input_set = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])`
`output_set = np.array([[1], [0], [0], [1]])`
`nn = NeuralNetwork(layer_sizes=[2, 2, 1], learning_rate=0.2)`

`nn.train(input_set, output_set, epochs=15000)`
`nn.test(input_set, output_set)`

## Training Visualization
During training, the Mean Squared Error (MSE) evolution is plotted and saved as `error_plot.png`.

## Limitations
- Currently supports only the Sigmoid activation function.
- Training is performed on a sample-by-sample basis (no batch training).
- Error calculation uses MSE.

## Future Improvements
- Add support for other activation functions (ReLU, Tanh, etc.).
- Implement batch training for larger datasets.
- Extend error functions beyond MSE (e.g., Cross-Entropy).
