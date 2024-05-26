import numpy as np
import csv 
import random

def Sigmoid(x):
    #Returns a value forced between 1 and 0
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = Sigmoid(x)
  return fx * (1 - fx)

class Neuron:
    def __init__(self, weights, startingWeights, bias):
        self.weights = weights
        self.startingWeights = startingWeights
        self.bias = bias
        
    def feedforward(self, inputs):
        #print("Weights: {}, Inputs:{}".format(self.weights, inputs))
        self.total = np.dot(self.weights, inputs) + self.bias
        
        self.value = Sigmoid(self.total)
        return self.value

class Layer:
    def __init__(self, size, numInputs):
        self.size = size
        self.neurons = []
        self.values = []
        for i in range(0, size):
            weights = self.GenerateWeights(numInputs)
            self.neurons.append(Neuron(weights, weights, random.random()))
    
    def GenerateWeights(self, numInputs):
        weights = []
        for i in range(0, numInputs):
            weights.append(random.random())
        print("Starting weights: {}".format(weights))
        return np.array(weights)


        

class Network:
    def __init__(self, learnRate):
        self.layers = [Layer(2,2), Layer(1,2)]
        self.learnRate = learnRate

    def feedforward(self, inputs):
        self.feedforwardNeurons(0, inputs)
        for i in range(1, len(self.layers)):
            #print("Layer: {}, Input length: {}".format(i, len(self.layers[i - 1].values)))
            self.feedforwardNeurons(i, np.array(self.layers[i - 1].values))
        
        return self.layers[1].neurons[0].value


    def feedforwardNeurons(self, layerIndex, inputs):
        for neuron in self.layers[layerIndex].neurons:
            
            neuron.feedforward(inputs)
            
            self.layers[layerIndex].values.append(neuron.value)

    def clearLayers(self):
        for layer in self.layers:
            layer.values.clear()

    def printWeights(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                print(neuron.weights)

    def printStartingWeights(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                print(neuron.startingWeights)
            

def MSE_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def deriv_MSE_loss(y_true, y_pred):
    return -2*(y_true- y_pred)

def TrainNetwork(network, truevalue, predicted):
    index = 0
    previousLayer = 0
    for layer in reversed(network.layers):
        for neuronIndex in range(0, len(layer.neurons)):
                if(index == 0):
                    layer.neurons[0].derivLoss = deriv_MSE_loss(truevalue, predicted)
                else:
                    layer.neurons[neuronIndex].derivLoss = FindAverageDerivative(previousLayer, neuronIndex)

                derivLoss = layer.neurons[neuronIndex].derivLoss
                #print("Layer: {}; Neuron: {}; DerivLoss: {}".format(index, neuronIndex, derivLoss))
                trueIndex = len(network.layers) - (index + 1)
                for weight in range(0, len(layer.neurons[neuronIndex].weights)):
                    subtractFromWeight = derivLoss * FindDerivativeRTW(network, trueIndex, layer.neurons[neuronIndex], inputs, weight)
                    #print("Weight before: {} descent value: {}".format(layer.neurons[neuronIndex].weights[weight], subtractFromWeight))
                    layer.neurons[neuronIndex].weights[weight] -= subtractFromWeight * network.learnRate
                    #print("Weight After: {}".format(layer.neurons[neuronIndex].weights[weight]))

                bias = layer.neurons[neuronIndex].bias
                #print("Bias: {}, DerivLoss: {}".format(bias, derivLoss * FindDerivativeRTB(layer.neurons[neuronIndex])))
                layer.neurons[neuronIndex].bias -= network.learnRate * derivLoss * FindDerivativeRTB(layer.neurons[neuronIndex])
        
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
    

names = ["Cat", "Dog"]

fileData = []
network = Network(10)
with open("centered_cats_and_dogs.csv", mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        print(row)
        inputs = np.array([float(row[1]), float(row[2])])
        
        output = network.feedforward(inputs)
        TrainNetwork(network, float(row[0]), output)
        print("Network answer: {}, true answer: {}".format(names[round(output)], names[int(row[0])]))
        network.clearLayers()
        

print("Finishing Weights")
network.printWeights()








#TrainNetwork(network)



