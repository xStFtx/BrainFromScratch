import numpy as np

class MeanSquaredErrorLoss:
    def calculate(self, target, prediction):
        return np.mean(np.square(target - prediction))

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update_weights(self, layers):
        if self.m is None:
            self.m = [np.zeros_like(neuron.weights) for neuron in layers[0].neurons]
            self.v = [np.zeros_like(neuron.weights) for neuron in layers[0].neurons]

        self.t += 1
        for layer in layers[:-1]:
            for i, neuron in enumerate(layer.neurons):
                grad = neuron.delta * neuron.activation
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                neuron.weights += -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class LearningRateScheduler:
    def __init__(self, initial_lr=0.1, decay_rate=0.9):
        self.lr = initial_lr
        self.decay_rate = decay_rate
        self.epoch = 0

    def get_learning_rate(self, epoch):
        self.epoch = epoch
        return self.lr * self.decay_rate ** epoch

class Neuron:
    def __init__(self, weights, bias, dropout_rate=0.0):
        self.weights = weights
        self.bias = bias
        self.activation = 0
        self.delta = 0
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def activate(self, input_data):
        weighted_sum = np.dot(self.weights, input_data) + self.bias
        self.activation = np.maximum(0, weighted_sum)  
        if self.dropout_rate > 0:
            self.mask = (np.random.rand(*self.activation.shape) > self.dropout_rate).astype(float)
            self.activation *= self.mask
        return self.activation

class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

class Brain:
    def __init__(self):
        self.layers = []
    
    def construct(self, layers, neurons_per_layer, dropout_rate=0.0):
        self.layers = []
        for _ in range(layers):
            layer_neurons = []
            for _ in range(neurons_per_layer):
                weights = np.random.randn(neurons_per_layer)  
                bias = np.random.randn()
                neuron = Neuron(weights, bias, dropout_rate)
                layer_neurons.append(neuron)
            layer = Layer(layer_neurons)
            self.layers.append(layer)
    
    def think(self, input_data):
        output_data = input_data
        for layer in self.layers:
            new_output_data = []
            for neuron in layer.neurons:
                activation = neuron.activate(output_data)
                new_output_data.append(activation)
            output_data = new_output_data
        return output_data
    
    def backpropagate(self, target):
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            error = target[i] - neuron.activation
            neuron.delta = error * neuron.activation * (1 - neuron.activation)
        
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            
            for i, neuron in enumerate(current_layer.neurons):
                error = sum(neuron.weights[j] * next_neuron.delta for j, next_neuron in enumerate(next_layer.neurons))
                neuron.delta = error * neuron.activation * (1 - neuron.activation)
    
    def update_weights(self, learning_rate):
        for layer in self.layers[:-1]:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    neuron.weights[i] += learning_rate * neuron.delta * neuron.activation

    def sensor(self, input_data):
        processed_input = input_data  
        output = self.think(processed_input)
        return output

def visualize_training_progress(epoch, loss):
    print(f"Epoch {epoch}, Loss: {loss}")

def main():
    na, nl = map(int, input("Enter the number of neurons and layers (space-separated): ").split())
    
    brain = Brain()
    brain.construct(nl, na)
    
    input_data = np.random.randn(na)
    target_data = np.random.rand(na)
    
    activation_fn = 'relu'  
    loss_function = MeanSquaredErrorLoss()
    
    num_epochs = 1000
    learning_rate_scheduler = LearningRateScheduler()
    optimizer = AdamOptimizer()
    
    for epoch in range(num_epochs):
        output = brain.sensor(input_data)
        loss = loss_function.calculate(target_data, output)
        brain.backpropagate(target_data)
        optimizer.update_weights(brain.layers)
        
        if epoch % 100 == 0:
            visualize_training_progress(epoch, loss)
        
        learning_rate = learning_rate_scheduler.get_learning_rate(epoch)
        optimizer.learning_rate = learning_rate 

if __name__ == "__main__":
    main()
