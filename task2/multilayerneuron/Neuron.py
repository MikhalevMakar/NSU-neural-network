from neural_network.task2.multilayerneuron import LinkNeuron


class Neuron:

    def __init__(self, links: [LinkNeuron]):
        self._links = links
        self._sum: float = 0
        self.__error: float = 0

    def add_sum(self, value: float):
        self._sum += value

    def set_sum(self, value: float):
        self._sum = value

    def get_data(self):
        return self._sum

    def get_links(self) -> [LinkNeuron]:
        return self._links

    def get_error(self):
        return self.__error

    def set_error(self, error_v: float):
        self.__error = error_v
