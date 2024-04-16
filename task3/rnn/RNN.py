import random

from neural_network.task3.rnn.ActivationNeuron import ActivationNeuron
from neural_network.task3.rnn.LinkCreator import LinkCreator
from neural_network.task3.rnn.RecurrentLayer import RecurrentLayer
from neural_network.task3.utils.SigmoidActivation import SigmoidActivation


class RNN:

    def __init__(self, neuron_layers: [int]):
        self.__neuron_layers = neuron_layers
        self.__sigmoid_activation = SigmoidActivation()
        self.__neurons = self.__create_rnn()

    def __create_rnn(self) -> [[ActivationNeuron]]:
        output_neuron = ActivationNeuron(None, self.__sigmoid_activation.get_activation_function())
        neurons = [[RecurrentLayer(output_neuron, [None], [None])]]

        current_layer = [output_neuron]
        for layer_index, layer_size in enumerate(reversed(self.__neuron_layers)):
            next_layer = current_layer.copy()
            h_t = self.__create_h(next_layer, layer_size)
            h_t_previous = self.__create_h(next_layer, layer_size)
            current_layer = self.__create_neuron_layer(
                next_layer,
                layer_index,
                layer_size
            )

            neurons.append(RecurrentLayer(current_layer, h_t, h_t_previous))
        return neurons[::-1]

    def __create_h(self, next_layer, size: int) -> [ActivationNeuron]:
        h = [ActivationNeuron]
        for i in range(size):
            h.append(ActivationNeuron(LinkCreator.create_links(next_layer, random.uniform), self.__sigmoid_activation))
        return h

    def __create_neuron_layer(self, previous_layer, layer_index, layer_size):
        current_layer = []
        for i in range(layer_size):
            links_to_next_layer = LinkCreator.create_links(previous_layer, random.uniform)
            activation_function = SigmoidActivation.x \
                if layer_index == len(self.__neuron_layers) - 1 \
                else self.__sigmoid_activation.get_activation_function()

            current_layer.append(ActivationNeuron(links_to_next_layer, activation_function))
        return current_layer

    def neurons(self) -> [[ActivationNeuron]]:
        return self.__neurons

    def layer(self, index: int) -> [ActivationNeuron]:
        return self.__neuron_layers[index]
