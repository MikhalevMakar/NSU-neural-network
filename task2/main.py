import random

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from neural_network.task2.ActivationNeuron import ActivationNeuron
from neural_network.task2.LinkNeuron import LinkNeuron
from neural_network.task2.MultilayerNeuron import MultilayerNeuron
from neural_network.task2.SigmoidActivation import SigmoidActivation


def create_net(laptops_columns: int) -> [[ActivationNeuron]]:
    sigmoid_activation = SigmoidActivation()

    output_neuron = ActivationNeuron(None, sigmoid_activation.get_activation_function())
    input_neurons = []
    layer_1_neurons = []
    layer_2_neurons = []
    # layer_3_neurons = []
    #
    # for h1 in range(5):
    #     link_neuron = [LinkNeuron(output_neuron, random.uniform(-1, 1))]
    #     neuron = ActivationNeuron(link_neuron, sigmoid_activation.get_activation_function())
    #     layer_3_neurons.append(neuron)
    for h1 in range(5):
        # links_to_3 = []
        # for n3 in layer_3_neurons:
        #     links_to_3.append(LinkNeuron(n3, random.uniform(-1, 1)))
        link_neuron = [LinkNeuron(output_neuron, random.uniform(-1, 1))]
        neuron = ActivationNeuron(link_neuron, sigmoid_activation.get_activation_function())
        layer_2_neurons.append(neuron)
    for h2 in range(5):
        links_2 = []
        for n2 in layer_2_neurons:
            links_2.append(LinkNeuron(n2, random.uniform(-1, 1)))
        neuron = ActivationNeuron(links_2, sigmoid_activation.get_activation_function())
        layer_1_neurons.append(neuron)
    for ind in range(laptops_columns - 2):
        links = []
        for n in layer_1_neurons:
            link_1 = LinkNeuron(n, random.uniform(-1, 1))
            links.append(link_1)
        neuron = ActivationNeuron(links, SigmoidActivation.x)
        input_neurons.append(neuron)
    return [input_neurons, layer_1_neurons, layer_2_neurons, [output_neuron]]


def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)


def load_and_normalize_data(filename):
    data = pd.read_csv(filename)
    normalized_data = normalize_data(data)
    return normalized_data


def un_normalize(val: float) -> float:
    return (max_value - min_value) * val + min_value


def run(per: MultilayerNeuron):
    max_diff = 0.0
    for i in range(len(laptops.values)):
        per.update_input_values(laptops.values[i])
        dif = abs(un_normalize(per.run()) - laptops1.values[i][-1])
        if dif > max_diff:
            max_diff = dif
        per.calc_full_error(laptops.values[i][-1])
        per.education_model()
        per.zeroing_net()
    print("iteration: " + str(iteration) + " max diff: " + str(max_diff))


if __name__ == "__main__":
    laptops1 = pd.read_csv('Laptop_price2.csv')
    max_value = laptops1['Price'].max()
    min_value = laptops1['Price'].min()
    laptops = load_and_normalize_data('Laptop_price2.csv')
    multilayer_neuron = MultilayerNeuron(create_net(len(laptops.columns)))
    for iteration in range(1000):
        run(multilayer_neuron)
