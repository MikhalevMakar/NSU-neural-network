from neural_network.task3.rnn import Neuron


class LinkNeuron:
    def __init__(self, neuron: Neuron, weight: float):
        self.__weight = weight
        self.next_neurons = neuron
        self.__previous_weight = 1.0

    def get_weight(self) -> float:
        return self.__weight

    def get_previous_weight(self) -> float:
        return self.__previous_weight

    def get_next_neuron(self) -> Neuron:
        return self.next_neurons

    def sum_weight_error(self, error_weight: float):
        self.__previous_weight = self.__weight
        self.__weight += error_weight

