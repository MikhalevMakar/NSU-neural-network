from neural_network.task3.rnn.ActivationNeuron import ActivationNeuron


class RecurrentLayer:

    def __init__(self, x, h_t: [ActivationNeuron], h_t_previous: [ActivationNeuron]):
        self.__x = x
        self.__h_t = h_t
        self.__h_t_previous = h_t_previous

    def get_x(self):
        return self.__x

    def get_h_t(self):
        return self.__h_t

    def get_h_t_previous(self):
        return self.__h_t_previous
