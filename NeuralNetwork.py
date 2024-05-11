import numpy as np

def Sigmoid(x):
    #Returns a value forced between 1 and 0
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = Sigmoid(x)
  return fx * (1 - fx)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def feedforward(self, inputs):
        self.total = np.dot(self.weights, inputs) + self.bias
        
        self.value = Sigmoid(self.total)
        return self.value

class Layer:
    def __init__(self, size, numInputs):
        self.size = size
        self.neurons = []
        self.values = []
        for i in range(0, size):
            self.neurons.append(Neuron(self.GenerateWeights(numInputs), 0))
    
    def GenerateWeights(self, numInputs):
        weights = []
        for i in range(0, numInputs):
            weights.append(1)

        return np.array(weights)


        

class Network:
    def __init__(self):
        self.layers = [Layer(2,2), Layer(1,2)]

    def feedforward(self, inputs):
        self.feedforwardNeurons(0, inputs)
        for i in range(1, len(self.layers)):
            self.feedforwardNeurons(i, np.array(self.layers[i - 1].values))
        
        return self.layers[1].neurons[0].value


    def feedforwardNeurons(self, layerIndex, inputs):
        for neuron in self.layers[layerIndex].neurons:
            neuron.feedforward(inputs)
            self.layers[layerIndex].values.append(neuron.value)

def MSE_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def deriv_MSE_loss(y_true, y_pred):
    return -2*(y_true- y_pred)

def TrainNetwork(network):
    index = 0
    previousLayer = 0
    for layer in reversed(network.layers):

        if(index == 0):
            layer.neurons[0].derivLoss = deriv_MSE_loss(1, layer.neurons[0].value)
            print("DerivRLT: " + str(layer.neurons[0].derivLoss))
        else:
            for neuronIndex in range(0, len(layer.neurons)):
                layer.neurons[neuronIndex].derivLoss = FindAverageDerivative(previousLayer, neuronIndex)
                derivLoss = layer.neurons[neuronIndex].derivLoss
                print("Layer: {}; Neuron: {}; DerivLoss: {}".format(index, neuronIndex, derivLoss))
                trueIndex = len(network.layers) - (index + 1)
                for weight in range(0, len(layer.neurons[neuronIndex].weights)):
                    print("Weight: {}, DerivLoss: {}".format(weight, derivLoss * FindDerivativeRTW(network, trueIndex, layer.neurons[neuronIndex], inputs, weight)))

        index = index + 1
        previousLayer = layer
        
            


#Average derivative of a layer with respect to one neuron (layer is the layer you're differentiating, and neuronIndex is the thing you're differentiating with respect to)
def FindAverageDerivative(layer, neuronIndex):
    
    derivativesLoss = []
    for neuron in layer.neurons:
        derivativeRTN = FindDerivativeRTN(neuron, neuronIndex)
        derivativesLoss.append(derivativeRTN * neuron.derivLoss)
    return np.array(derivativesLoss).mean()

#Respect to weight
def FindDerivativeRTW(network, layerIndex, neuron, inputs, weightIndex):
    
    if layerIndex > 0:
        
        return deriv_sigmoid(neuron.total) * network.layers[layerIndex - 1].neurons[weightIndex].value
    else:
        return deriv_sigmoid(neuron.total) * inputs[weightIndex]
    
#Respect to bias
def FindDerivativeRTB(neuron):
    return deriv_sigmoid(neuron.total)


    


#Respect to another neuron (neuron is the neuron you're differentiating, neuronIndex is the neuron you're  differentiating with respect to)
def FindDerivativeRTN(neuron, neuronIndex):
    return deriv_sigmoid(neuron.total) * neuron.weights[neuronIndex]
    



    




network = Network()
inputs = np.array([0,1])
output = network.feedforward(inputs)
print("Output: " + str(output))
TrainNetwork(network)
#TrainNetwork(network)



