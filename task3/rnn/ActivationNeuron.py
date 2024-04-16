from neural_network.task3.rnn import LinkNeuron
from neural_network.task3.rnn.Neuron import Neuron


class ActivationNeuron(Neuron):

    def __init__(self, links: [LinkNeuron], activation):
        super().__init__(links)
        self._activation = activation

    def run(self):
        activation: float = self.get_activation()
        for link in self._links:
            link.next_neurons.add_sum(link.get_weight() * activation)

    def get_activation(self) -> float:
        return self._activation(self._sum)