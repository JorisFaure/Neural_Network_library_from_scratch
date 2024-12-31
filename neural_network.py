import numpy as np
import time
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.uniform(0.5, 1.0, size=(input_size,))
        self.bias = 0
        self.value = 0
        self.activated_value = 0

    def sigmoid(self):
        return 1 / (1 + np.exp(-self.value))

    def sigmoid_derivative(self):
        return self.activated_value * (1 - self.activated_value)

    def activate(self):
        self.activated_value = self.sigmoid()

    def forward_pass(self, input_vec):
        self.value = np.dot(self.weights, input_vec) + self.bias
        self.activate()

    def cost_derivative_by_layer(self, output_ref) : # (1/2) * 2(y - y_ref)
        return (self.activated_value - output_ref)

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        self.total_layers = []
        for i in range(len(layer_sizes) - 1):
            self.total_layers.append([Neuron(layer_sizes[i]) for _ in range(layer_sizes[i + 1])])
        self.learning_rate = learning_rate

    def forward_pass(self, input_vec):
        current_input = input_vec
        for layer in self.total_layers:
            for neuron in layer:
                neuron.forward_pass(current_input)
            current_input = [neuron.activated_value for neuron in layer]

    def backpropagation(self, output_vec, input_vec):
        layer_outputs = [input_vec]
        for layer in self.total_layers: # we store all the a(L=n) values
            layer_outputs.append([neuron.activated_value for neuron in layer])

        errors = [] # store the error of each layer
        weight_gradients = [] # store all the weights gradient of each layers (a list of np arrays)
        bias_gradients = [] #store all the bias_gradient of each layers (a list of values)

        
        output_layer = self.total_layers[-1]
        output_errors = [] # store the propagate errors from each output neuron to the previous layer 
        for i, neuron in enumerate(output_layer): # Output backprop
            bias_gradient = neuron.cost_derivative_by_layer(output_vec[i]) * neuron.sigmoid_derivative() # da(L)/dz(L) * dC0/da(L) ***the last term is a(L-1)***
            output_errors.append(bias_gradient) # we use this list for propagating the error
            bias_gradients.append(bias_gradient) # we use this list for the bias update
            weight_gradients.append(np.array(layer_outputs[-2]) * bias_gradient)

        errors.append(output_errors)

        # hidden layers backprop
        for l in range(len(self.total_layers) - 2, -1, -1): #we start from the last hidden layer, to the first hidden layer
            layer = self.total_layers[l]
            next_layer = self.total_layers[l + 1]
            hidden_errors = [] # store the propagate errors from each a(L) neuron to the a(L-1) layer
            for j, neuron in enumerate(layer): # we calculate the propagate sum error from layer L to layer L-1 : dC/dz(L-1) = (Σ(dC0/dz(L)) * w(L)) * (da(L-1)/dz(L-1))
                s = sum(next_layer[k].weights[j] * errors[-1][k] for k in range(len(next_layer))) # error[-1][k] represent the errors of each neuron of the last layer (-1 is the last layer append to the errors list)
                bias_gradient = s * neuron.sigmoid_derivative()
                hidden_errors.append(bias_gradient)
                bias_gradients.append(bias_gradient)
                weight_gradients.append(np.array(layer_outputs[l]) * bias_gradient) # bias_gradient * dC0/da(L-1)
            errors.append(hidden_errors) # we will use the error of the a(l-1) for the next iteration with error[-1]
        
        #weights and biases update
        i = 0
        for l in range(len(self.total_layers) - 1, -1, -1):
            for neuron in self.total_layers[l]:
                neuron.weights -= self.learning_rate * weight_gradients[i] #we update each weight list of each neurons
                neuron.bias -= self.learning_rate * bias_gradients[i] #we update each bias of each neurons
                i+=1

    def train(self, input_data, output_data, epochs):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for i, sample in enumerate(input_data):
                self.forward_pass(sample)
                self.backpropagation(output_data[i], sample)
                #calculate total error
                total_error += sum((neuron.activated_value - output_data[i][j])**2 for j, neuron in enumerate(self.total_layers[-1]))
            errors.append(total_error / len(input_data))
            if epoch % 1000 == 0:
                # show the MSE every x epoch
                print(f"Epoch {epoch}, MSE: {errors[-1]}")
        plt.plot(range(epochs), errors, label='Total error')
        plt.xlabel('Epoch')
        plt.ylabel('Mean error')
        plt.title('Error evolution through epoch')
        plt.legend()
        plt.savefig("error_plot.png")
        print("Plot saved as 'error_plot.png'")

    def test(self, input_set, output_set):
        for i, sample in enumerate(input_set):
            self.forward_pass(sample)
            prediction = [round(neuron.activated_value, 2) for neuron in self.total_layers[-1]]
            print(f"Predicted: {prediction} | Reference: {output_set[i]}")



# -----------------------Test of the entire NN-------------------------------
# XOR example
input_set = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_set = np.array([[1], [0], [0], [1]])
nn = NeuralNetwork(layer_sizes=[2, 2, 1], learning_rate=0.2)

# Mesure training time
print("Starting training...")
start_time = time.time()
nn.train(input_set, output_set, epochs=15000)
end_time = time.time()
print("Training over")

training_time = end_time - start_time
print(f"Training time : {training_time:.2f} seconds.")

# Network test
print("\nTest result :")
nn.test(input_set, output_set)