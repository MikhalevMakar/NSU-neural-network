from neural_network.task2 import Neuron


class LinkNeuron:
    def __init__(self, neuron: Neuron, weight: float):
        self.__weight = weight
        self.next_neurons = neuron

    def get_weight(self) -> float:
        return self.__weight

    def get_next_neuron(self) -> Neuron:
        return self.next_neurons

    def sum_weight_error(self, error_weight: float):
        self.__weight += error_weight
