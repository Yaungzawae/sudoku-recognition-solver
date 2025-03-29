import numpy as np
import pandas as pd
import os

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weights = None, biases = None):
        if weights is None or biases is None:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
            self.biases = np.zeros((1, n_neurons))
        else:
            self.weights = weights
            self.biases = biases

    def forward(self, inputs):
        self.inputs = inputs
        output = np.dot(inputs, self.weights) + self.biases
        self.output = output
        return output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLu: # return O if less than 0, else return the input
    def forward(self, inputs):
        output = np.maximum(0, inputs)
        self.inputs = inputs
        self.output = output
        return output

    def backward(self, dvalue):
        self.dinputs = dvalue.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, keepdims=True, axis=1))
        probabilities = exp_values / np.sum(exp_values, keepdims=True, axis=1)
        output = probabilities
        self.output = output
        return output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalue)

class Loss:
    def caculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidiences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidiences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidiences)
        return  negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(y_true)
        labels = len(y_true[0])
        #one hot encode
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true=None):
        self.activation.forward(inputs)
        self.output = self.activation.output
        if y_true is not None:
            return self.loss.caculate(self.activation.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optmizer_SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

class Model:
    def __init__(self, use_existing, folder_path=None):
        self.weights1 = self.weights2 = self.biases1 = self.biases2 = None
        self.folder_path = "parameters"
        if folder_path:
            self.folder_path = folder_path

        if use_existing:
            self.weights1 = np.array(pd.read_csv(f"{folder_path}/Dense1W.csv", header=None))
            self.biases1 = np.array(pd.read_csv(f"{folder_path}/Dense1B.csv", header=None))
            self.weights2 = np.array(pd.read_csv(f"{folder_path}/Dense2W.csv", header=None))
            self.biases2 = np.array(pd.read_csv(f"{folder_path}/Dense2B.csv", header=None))

        self.dense1 = Layer_Dense(784, 9, self.weights1, self.biases1)
        self.activation1 = Activation_ReLu()
        self.dense2 = Layer_Dense(9, 9, self.weights2, self.biases2)
        self.activation2 = Activation_Softmax_Loss_CategoricalCrossentropy()

    def forward(self, inputs, labels):
        self.dense1.forward(inputs)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.loss = self.activation2.forward(self.dense2.output, labels)

    def backward(self, labels):
        self.activation2.backward(self.activation2.output, labels)
        self.dense2.backward(self.activation2.dinputs)
        self.activation1.backward(self.dense2.dinputs)
        self.dense1.backward(self.activation1.dinputs)

    def predict(self):
        self.predictions = np.argmax(self.activation2.output, axis=1)
        return self.predictions

    def predict_one(self, inputs):
        self.dense1.forward(inputs)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)
        return np.argmax(self.activation2.output, axis=1)


    def get_accuracy(self, predictons ,labels):
        return np.sum(predictons == labels) / labels.size

    def print_model(self, labels, i):
        predictions = self.predict()
        self.accuracy = self.get_accuracy(predictions, labels)
        print("-" * 12)
        print("Iteration: ", i, "Loss: ", self.loss)
        print(f"{np.sum(self.predictions == labels)} / {labels.size}")
        print("Accuracy: ", self.accuracy)

    def gradientDesent(self,inputs, y_true, iterations, target_accuracy):
        sgd = Optmizer_SGD(0.001)
        for i in range(iterations):
            self.forward(inputs, y_true)
            self.backward(y_true)

            if i % 10 == 0:
                self.print_model(y_true, i)
                if self.accuracy >= target_accuracy: break

            sgd.update_params(self.dense2)
            sgd.update_params(self.dense1)

            if i != 0 and i % 500 == 0:
                self.save_to_csv()

    def save_to_csv(self):
        os.makedirs(self.folder_path, exist_ok=True)
        pd.DataFrame(self.dense1.weights).to_csv(f"{self.folder_path}/Dense1W.csv", header= None, index=None)
        pd.DataFrame(self.dense1.biases).to_csv(f"{self.folder_path}/Dense1B.csv", header= None, index=None)
        pd.DataFrame(self.dense2.weights).to_csv(f"{self.folder_path}/Dense2W.csv", header= None, index=None)
        pd.DataFrame(self.dense2.biases).to_csv(f"{self.folder_path}/Dense2B.csv", header= None, index=None)
