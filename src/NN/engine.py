# coding=utf-8
from numpy import array, exp, dot, random


class NeuralLayer:
    def __init__(self, number_of_neurons, number_of_input_per_nueron):
        # random weights to
        self.synaptic_weights = 2 * random.random((number_of_input_per_nueron, number_of_neurons)) -1


class NeuralNet:
    def __init__(self, layer1, layer2):
        self.layer_1 = layer1
        self.layer_2 = layer2

    # normalization of the weighted sum to a range 0 to 1
    @staticmethod
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # confidence level of weights
    @staticmethod
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, train_set_input, train_set_output, num_iterations):
        for iterations in range(num_iterations):
            # pass training  set to a single neuron in out net
            output_from_layer_1, output_from_layer_2 = self.think(train_set_input)

            # difference between desired output and predicted from layer 2
            layer_2_error = train_set_output - output_from_layer_2
            layer_2_delta = layer_2_error * self.sigmoid_derivative(self, output_from_layer_2)

            # calculate the error for layer 1 in comparison to error in layer 2
            layer_1_error = layer_2_delta.dot(self.layer_2.synaptic_weights.T)
            layer_1_delta = layer_1_error * self.sigmoid_derivative(self, output_from_layer_1)

            # how much to adjust the weights by
            layer_1_adjustment = train_set_input.T.dot(layer_1_delta)
            layer_2_adjustment = output_from_layer_1.T.dot(layer_2_delta)


            # adjust weights
            self.layer_1.synaptic_weights += layer_1_adjustment
            self.layer_2.synaptic_weights += layer_2_adjustment

    def think(self, inputs):
        output_from_layer_1 = self.sigmoid(self, dot(inputs, self.layer_1.synaptic_weights))
        output_from_layer_2 = self.sigmoid(self, dot(output_from_layer_1, self.layer_2.synaptic_weights))
        return output_from_layer_1, output_from_layer_2

    @staticmethod
    def print_weights(self):
        print("Layer 1 (4 neurons, each with 3 inputs): ")
        print(self.layer_1.synaptic_weights)
        print("Layer 2 (1 neuron, with 4 inputs): ")
        print(self.layer_2.synaptic_weights)

if __name__ == "__main__":
    random.seed(1)


    layer_1 = NeuralLayer(4, 3)
    layer_2 = NeuralLayer(1, 4)

    neural_net = NeuralNet(layer_1, layer_2)

    print("Stage 1: Random starting synaptic weights")
    print(neural_net.print_weights(neural_net))

    # training set with 7 data sets each with 3 inputs and i ouput value
    train_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0],
                              [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    train_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # training the net with train set
    # 60,000 iterations with adjustments at each
    neural_net.train(train_set_inputs, train_set_outputs, 60000)

    print("Stage 2: New synaptic weights after training")
    print(neural_net.print_weights(neural_net))

    # test neural net with new situation
    print("Stage3. Considering new situation [1, 1, 0] => ?: ")
    hidden_state, output = neural_net.think(array([1, 1, 0]))
    print(output)
