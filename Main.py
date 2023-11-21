# This program  demonstrates a simple feed-forward neural network with two input neurons,
# two hidden neurons, and one output neuron. The network is trained using backpropagation
# and the weights and biases are updated according to the computed gradients.
# Finally, we make predictions for each input and display the results.
from numpy.random import randn
from numpy import array, exp

class NeuralNetwork:
    def __init__(self, layers):
        self.layers, self.num_layers, self.weights, self.biases =\
            layers, len(layers), (*(randn(y, x) for x, y in zip(layers[:-1], layers[1:])),),\
            (*(randn(y, 1) for y in layers[1:]),)

    def sigmoid(self, act):
        return 1 / (1 + exp(-act)) #e^-act

    def sigmoid_prime(self, act):
        sig_Act = self.sigmoid(act)

        return sig_Act * (1 - sig_Act) #sig_Act-sig_Act**2

    def f_act(self, acts, pesos, biases):
        return acts @ pesos + biases

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(self.f_act(w, a, b))
            #prod punto consigo mismo
            #no es lo mismo a@w q w@a NO sirve a.T trasponer col->fila
            
        return a

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for x, target in zip(X, y):
                x,target,zs = array(x).reshape(-1, 1),\
                           array(target).reshape(-1, 1), []              
                activations = [x] # Forward pass

                for w, b in zip(self.weights, self.biases):
                    z = self.f_act(w,activations[-1],b)

                    zs.append(z)
                    activations.append(self.sigmoid(z))

                # Backward pass
                delta = self.sigmoid_prime(zs[-1]) *\
                        (activations[-1] - target) 
                        

                nabla_b, nabla_w = [delta],\
                                   [delta @ activations[-2].T]

                for l in range(2, self.num_layers):
                    delta = self.weights[-l + 1].T @ delta *\
                            self.sigmoid_prime(zs[-l])

                    nabla_b.append(delta)
                    nabla_w.append(delta @ activations[-l - 1].T)

                nabla_b, nabla_w = nabla_b[::-1], nabla_w[::-1]

                # Update weights and biases
                self.weights, self.biases = (*(w - learning_rate * dw for w, dw in zip(self.weights, nabla_w)),),\
                    (*(b - learning_rate * db for b, db in zip(self.biases, nabla_b)),)

# Example usage
X, y, nn = ((0, 0), (0, 1), (1, 0), (1, 1)), (0, 1, 1, 0),\
           NeuralNetwork((2, 2, 1))  # 2 input neurons, 2 hidden neurons
           #1 output neuron

nn.train(X, y, epochs=int(1e4), learning_rate=1e-1)

'''
print((*(zip(X, y)),))
(((0, 0), 0),
((0, 1), 1),
((1, 0), 1),
((1, 1), 0))
rango 4
'''
acts1=[]
for x, target in zip(X, y):    
    prediction = nn.feedforward(array(x).reshape(-1, 1))
    acts1.append(round(float(prediction), 2))

    '''
    print(x,target)
    print('acts:',x,'acts1:',round(float(prediction), 2))
    f"Input: {x} -> Prediction: {prediction}")
    '''
    
print('acts:',X,'\nacts1:',tuple(acts1))
    
