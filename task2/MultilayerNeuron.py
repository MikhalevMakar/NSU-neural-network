from numpy import ndarray

from neural_network.task2.ActivationNeuron import ActivationNeuron


class MultilayerNeuron:
    def __init__(self, neurons: [[ActivationNeuron]]):
        self.neurons: [[ActivationNeuron]] = neurons
        self.__education_speed = 0.4

    def run(self) -> float:
        for layer in self.neurons[:-1]:
            for neuron in layer:
                neuron.run()
        return self.neurons[-1][-1].get_activation()

    @staticmethod
    def __calc_output_error(output_value: float, target_value: float) -> float:
        return (output_value - target_value) * output_value * (1 - output_value)

    @staticmethod
    def __calc_error(actual_value: float, next_error: float) -> float:
        return next_error * actual_value * (1 - actual_value)

    @staticmethod
    def count_next_error(neuron: ActivationNeuron) -> float:
        error = 0
        for link in neuron.get_links():
            next_neuron = link.get_next_neuron()
            error += next_neuron.get_error() * link.get_weight()
        return error

    def calc_full_error(self, target_value: float):
        output_neuron = self.neurons[-1][-1]
        output_error: float = self.__calc_output_error(output_neuron.get_activation(), target_value)
        output_neuron.set_error(output_error)
        for layer in reversed(self.neurons[1:-1]):
            for neuron in layer:
                next_error = self.count_next_error(neuron)
                error = self.__calc_error(neuron.get_activation(), next_error)
                neuron.set_error(error)

    def __update_weight(self, neuron: ActivationNeuron):
        for link in neuron.get_links():
            next_neuron = link.get_next_neuron()
            link.sum_weight_error(-self.__education_speed * next_neuron.get_error() * neuron.get_activation())

    def education_model(self):
        for layer in reversed(self.neurons[:-1]):
            for neuron in layer:
                self.__update_weight(neuron)

    def update_input_values(self, values: ndarray):
        for i, inp_neuron in enumerate(self.neurons[0]):
            inp_neuron.set_sum(values[i+1])

    def zeroing_net(self):
        for layer in self.neurons:
            for neuron in layer:
                neuron.set_sum(0)
