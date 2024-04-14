import random

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from neural_network.task2.multilayerneuron.ActivationNeuron import ActivationNeuron
from neural_network.task2.multilayerneuron.LinkCreator import LinkCreator
from neural_network.task2.multilayerneuron.MultilayerNeuron import MultilayerNeuron
from neural_network.task2.utils.SigmoidActivation import SigmoidActivation
from sklearn.model_selection import train_test_split


def create_neuron_layer(neuron_layers, previous_layer, layer_index, layer_size, sigmoid_activation):
    current_layer = []
    for i in range(layer_size):
        links_to_next_layer = LinkCreator.create_links(previous_layer, random.uniform)

        activation_function = SigmoidActivation.x \
            if layer_index == len(neuron_layers) - 1 \
            else sigmoid_activation.get_activation_function()

        current_layer.append(ActivationNeuron(links_to_next_layer, activation_function))
    return current_layer


def create_net(neuron_layers: [int]) -> [[ActivationNeuron]]:
    sigmoid_activation = SigmoidActivation()
    output_neuron = ActivationNeuron(None, sigmoid_activation.get_activation_function())
    neurons = [[output_neuron]]
    current_layer = [output_neuron]
    for layer_index, layer_size in enumerate(reversed(neuron_layers)):
        next_layer = current_layer.copy()

        current_layer = create_neuron_layer(
            neuron_layers,
            next_layer,
            layer_index,
            layer_size,
            sigmoid_activation
        )

        neurons.append(current_layer)
    return neurons[::-1]


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


def education(multilayerNeuron: MultilayerNeuron, education_set_norm: pd.DataFrame):
    for i in range(len(education_set_norm.values)):
        multilayerNeuron.update_input_values(education_set_norm.values[i])
        multilayerNeuron.run()
        multilayerNeuron.calc_full_error(float(education_set_norm.values[i][-1]))
        multilayerNeuron.education_model()
        multilayerNeuron.zeroing_net()


def test(multilayerNeuron: MultilayerNeuron, test_set_norm: pd.DataFrame, test_set: pd.DataFrame):
    for i in range(len(test_set.values)):
        multilayerNeuron.update_input_values(test_set_norm.values[i])
        un_result = un_normalize(multilayerNeuron.run())
        print("result " + str(un_result) + " " + " target " + str(
            test_set.values[i][-1]) + " predication " + classification_mushroom(int(un_result))
              )
        multilayerNeuron.zeroing_net()


def education_test(dataset_origin: pd.DataFrame, dataset_normal: pd.DataFrame):
    x_train, x_test = train_test_split(dataset_origin, test_size=0.25, random_state=42)
    x_train_norm, x_test_norm, = train_test_split(dataset_normal, test_size=0.25, random_state=42)
    print(str(len(x_train.values)) + " " + str(len(x_test.values)))

    multilayer_neuron = MultilayerNeuron(create_net([len(x_train.columns) - 1, 5, 3]))
    print("part  education: ")

    for i in range(200):
        education(multilayer_neuron, x_train_norm)

    print("part test: ")
    test(multilayer_neuron, x_test_norm, x_test)


if __name__ == "__main__":
    dataset = pd.read_csv('laptop_price.csv')
    max_value = dataset['Price'].max()
    min_value = dataset['Price'].min()
    dataset_normalize = load_and_normalize_data('laptop_price.csv')
    education_test(dataset, dataset_normalize)
