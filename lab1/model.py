import numpy as np

class Layer:
    def __init__(self, in_channel, out_channel, activation = 'sigmoid', lr = '0.1', optimizer='adam'):
        self.weight = np.random.normal(0, 1, (in_channel + 1, out_channel)) # in_channel + 1 -> weight + bias
        self.activation = activation
        self.lr = lr 
        self.gradient=0

        # For Adam Optimizer
        self.optimizer = optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = np.zeros_like(self.weight)  # First moment vector
        self.v = np.zeros_like(self.weight)  # Second moment vector
        self.t = 0  # Timestep

    def forward(self, inputs):

        self.hidden_input = np.append(inputs, np.ones((inputs.shape[0], 1)), axis=1)
        
        if self.activation == 'sigmoid':
            self.hidden_output = self.sigmoid(np.matmul(self.hidden_input, self.weight))
        
        elif self.activation == 'tanh':
            self.hidden_output = self.tanh(np.matmul(self.hidden_input, self.weight))

        elif self.activation == 'relu':
            self.hidden_output = self.relu(np.matmul(self.hidden_input, self.weight))
        else:
            self.hidden_output = np.matmul(self.hidden_input, self.weight)

        return self.hidden_output
    def backward(self, derivative_loss):
        if self.activation == 'sigmoid':
            self.backward_grad = np.multiply(self.derivative_sigmoid(self.hidden_output), derivative_loss)
        
        elif self.activation == 'tanh':
            self.backward_grad = np.multiply(self.derivative_tanh(self.hidden_output), derivative_loss)

        elif self.activation == 'relu':
            self.backward_grad = np.multiply(self.derivative_relu(self.hidden_output), derivative_loss)
        else:
            self.backward_grad = derivative_loss

        self.gradient = np.matmul(self.hidden_input.T, self.backward_grad)

        return np.matmul(self.backward_grad, self.weight[:-1].T) # weight[:-1] denotes excluding bias
    
    def optimize(self):
        if self.optimizer == 'adam':
            self.t += 1  # Increment the timestep
            self.m = self.beta1 * self.m + (1 - self.beta1) * self.gradient  
            self.v = self.beta2 * self.v + (1 - self.beta2) * (self.gradient ** 2)  
            m_hat = self.m / (1 - self.beta1 ** self.t)  
            v_hat = self.v / (1 - self.beta2 ** self.t) 
            self.weight -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)  # Update parameters
        else:
            self.weight -= self.lr * self.gradient

    def relu(self, x):
        return np.maximum(0, x)

    def derivative_relu(self, x):
        return np.where(x > 0, 1.0, 0.0)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return np.multiply(x, 1.0 - x)

    def tanh(self, x):
        return np.tanh(x)
    
    def derivative_tanh(self, y):
        return 1.0 - y ** 2

class Model:
    def __init__(self, in_unit = 2, hidden_unit = 4, activation = 'sigmoid', lr = 0.1, optimizer = 'adam'):
        self.lr = lr
        self.activation = activation

        self.layers = [
            Layer(in_unit, hidden_unit, activation, lr, optimizer),    # input layer
            Layer(hidden_unit, hidden_unit, activation, lr, optimizer),# hidden layer
            Layer(hidden_unit, hidden_unit, activation, lr, optimizer),# hidden layer
            Layer(hidden_unit, 1, 'sigmoid', lr, optimizer)            # output layer
            ]
       
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, derivative_loss):
        for layer in self.layers[::-1]: # reverse the layers list
            derivative_loss = layer.backward(derivative_loss)
    
    def optimize(self):
        for layer in self.layers:
            layer.optimize()

