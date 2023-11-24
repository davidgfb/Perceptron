from math import exp
from random import seed, random
from numpy import linspace, round
from matplotlib.pyplot import figure, show, plot

def o(x, y): # the objective function
    return 1 if x**2 + y**2 < 1 else 0

def sigmoid(x): # activation function
    return 1 / (1 + exp(-x))

def sigmoid_derivative(x): 
    output = sigmoid(x)

    return output * (1 - output)

sample_density = 10 # samples
xs = (*((*(-2 + 4 * x/sample_density, -2 + 4 * y/sample_density),)
     for x in range(sample_density+1) for y in range(sample_density+1)),)
dataset = (*((x, y, o(x, y)) for x, y in xs),)

seed(0)

class Neuron: # neural network
    def __init__(self, num_inputs):        
        self.weights, self.bias = [random()-0.5 for i in range(num_inputs)],\
                                  0.0 #no puede ser tupla
        self.z_cache, self.inputs_cache = None, None # caches

    def forward(self, inputs):
        # z = wx + b
        self.inputs_cache, self.z_cache = inputs,\
                                        sum((*(i * w for i, w in\
                       zip(inputs, self.weights)),)) + self.bias
   
        return sigmoid(self.z_cache) # sigmoid(wx + b)

    def zero_grad(self):
        self.d_weights, self.d_bias = [0.0 for w in self.weights],\
                                      0.0 #no puede ser tupla

    def backward(self, d_a):
        d_loss_z = d_a * sigmoid_derivative(self.z_cache)
        self.d_bias += d_loss_z

        for i in range(len(self.inputs_cache)):
            self.d_weights[i] += d_loss_z * self.inputs_cache[i]

        return (*(d_loss_z * w for w in self.weights),)

    def update_params(self, learning_rate):
        self.bias -= learning_rate * self.d_bias

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.d_weights[i]

    '''def params(self):
        #print(self.weights + [self.bias])
        return self.weights + [self.bias]'''

class MyNet:
    def __init__(self, num_inputs, hidden_shapes):
        layer_shapes, input_shapes = (hidden_shapes, 1),\
                                     (num_inputs,hidden_shapes)
        
        self.layers = (*((*(Neuron(pre_layer_size)
                                    for _ in range(layer_size)),)
                            for layer_size, pre_layer_size in\
                             zip(layer_shapes, input_shapes)),)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = (*(neuron.forward(inputs) for neuron in layer),)
      
        return inputs[0] # return the output of the last neuron

    def zero_grad(self):
        (*((*(neuron.zero_grad() for neuron in layer),) for layer in self.layers),)
            
    def backward(self, d_loss):
        d_as = (d_loss,)

        for layer in reversed(self.layers):
            da_list = (*(neuron.backward(d_a)
                      for neuron, d_a in zip(layer, d_as)),) #OJO si d_as = (d_loss,)
            d_as = (*(sum(da) for da in zip(*da_list)),)

    def update_params(self, learning_rate):
        (*((*(neuron.update_params(learning_rate) for neuron in layer),) for layer in self.layers),)
            
    def params(self):
        return (*((*(neuron.params() for neuron in layer),)
                for layer in self.layers),)

def square_loss(predict, target): # loss function
    return (predict-target)**2

def square_loss_derivative(predict, target):
    return 2 * (predict-target)

# build neural network
net = MyNet(2, 4)

#print(round(net.forward((0, 0)),2)) #valor inicial

targets = (*(z for x, y, z in dataset),)

# train
def one_step(learning_rate):
    net.zero_grad()

    loss = 0.0
    num_samples = len(dataset)
    for x, y, z in dataset:
        predict = net.forward((x, y))
        loss += square_loss(predict, z)

        net.backward(square_loss_derivative(predict, z) / num_samples)

    net.update_params(learning_rate)

    return loss / num_samples

ptos = [[], []]
def train(epoch, learning_rate):
    for i in range(epoch):
        loss = one_step(learning_rate)

        (*(ptos[j].append(k) for j,k in ((0,i + 1),(1,round(loss, 2)))),) if i == 0 or (i + 1) % 100 == 0 else None
                       
def inference(x, y):
    return net.forward((x, y))

data, r, poss = [], round(linspace(-2,2,20),2), [[],[],[]] #sample density

train(2000, learning_rate=10)
plot(*ptos, 'o-') #f error/perdida

(*((*(data.append((x,y,round(inference(x,y),2))) for y in r),) for x in r),)    
(*((*(poss[j].append(data[i][j]) for j in range(3)),) for i in range(len(data))),)
    
figure(figsize=(8, 8)).add_subplot(111, projection='3d').\
                   scatter(*poss) 
show()
