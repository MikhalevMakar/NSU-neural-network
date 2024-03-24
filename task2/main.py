import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from neural_network.task2.ActivationNeuron import ActivationNeuron
from neural_network.task2.LinkNeuron import LinkNeuron
from neural_network.task2.MultilayerNeuron import MultilayerNeuron
from neural_network.task2.SigmoidActivation import SigmoidActivation
from sklearn.model_selection import train_test_split


# def create_net(laptop_normalize_columns: int) -> [[ActivationNeuron]]:
#     sigmoid_activation = SigmoidActivation()
#
#     output_neuron = ActivationNeuron(None, sigmoid_activation.get_activation_function())
#     input_neurons = []
#     layer_1_neurons = []
#     layer_2_neurons = []
#     layer_3_neurons = []
#
#     for h1 in range(6):
#         link_neuron = [LinkNeuron(output_neuron, random.uniform(-1, 1))]
#         neuron = ActivationNeuron(link_neuron, sigmoid_activation.get_activation_function())
#         layer_3_neurons.append(neuron)
#     for h1 in range(5):
#         links_3 = []
#         for n3 in layer_3_neurons:
#             links_3.append(LinkNeuron(n3, random.uniform(-1, 1)))
#         # link_neuron = [LinkNeuron(links_to_3, random.uniform(-1, 1))]
#         neuron = ActivationNeuron(links_3, sigmoid_activation.get_activation_function())
#         layer_2_neurons.append(neuron)
#     for h2 in range(5):
#         links_2 = []
#         for n2 in layer_2_neurons:
#             links_2.append(LinkNeuron(n2, random.uniform(-1, 1)))
#         neuron = ActivationNeuron(links_2, sigmoid_activation.get_activation_function())
#         layer_1_neurons.append(neuron)
#     for ind in range(laptop_normalize_columns - 2):
#         links = []
#         for n in layer_1_neurons:
#             link_1 = LinkNeuron(n, random.uniform(-1, 1))
#             links.append(link_1)
#         neuron = ActivationNeuron(links, SigmoidActivation.x)
#         input_neurons.append(neuron)
#     return [input_neurons, layer_1_neurons, layer_2_neurons, [output_neuron]]

def classification_mushroom(prediction: int) -> str:
    if prediction == 111:
        return "p"
    return "e"


weights = [0.6107772252805626, 0.2946815159874838, -0.843037214051362, -0.6750553082518584, -0.14045021152884063,
           -0.8830705995777086, 0.28671748261283514, -0.2665059193411432, 0.09914217234148004, -0.06468763232488928,
           0.659111991511899, -0.7763788614678577, 0.37276045039909156, -0.16662309919439333, -0.5050992547348856,
           -0.44266706180053994, -0.7862465050247849, 0.769028803080368, -0.9252012176696041, -0.9415646904955857,
           0.3133794006797326, -0.5376549328299884, -0.20601150203967955, 0.3076585168777586, 0.4647630051254916,
           0.7592100384926765, 0.14489163749923972, 0.2797810975046422, -0.17771399581875946, 0.8133135868846342,
           0.6031688914414919, -0.5428108966962346, 0.38768491868284016, -0.9898244473530544, -0.843725153280509,
           0.921849879927146, -0.7434762173349574, 0.29987586629826524, 0.7066842696410083, 0.8348775779539586,
           0.5991150340030618, -0.04214436740001659, -0.11116366017491575, 0.8354317534836535, -0.3232744695998988,
           0.0956547466821056, 0.23088116313065332, -0.03657036705478989, -0.4002115053518138, 0.28344994611186225,
           0.5137599014905436, 0.03394170157849108, 0.9341607493275046, 0.0748205284477319, 0.09061861133161275,
           0.7014927147196777, 0.9014294424478286, -0.8813999633965104, 0.2891355199011041]


def create_net(neuron_counts: [int]) -> [[ActivationNeuron]]:
    sigmoid_activation = SigmoidActivation()
    output_neuron = ActivationNeuron(None, sigmoid_activation.get_activation_function(0))
    neurons = [[output_neuron]]
    cur_layer = [output_neuron]
    index_weight = 0
    for index, count in enumerate(reversed(neuron_counts)):
        next_layer = cur_layer.copy()
        cur_layer = []
        for i in range(count):
            links_to_next_layer = []
            for n in next_layer:
                # links_to_next_layer.append(LinkNeuron(n, weights[index_weight]))
                links_to_next_layer.append(LinkNeuron(n, weights[index_weight]))
                index_weight += 1
            func = SigmoidActivation.x if index == len(
                neuron_counts) - 1 else sigmoid_activation.get_activation_function(i)
            cur_layer.append(ActivationNeuron(links_to_next_layer, func))
        neurons.append(cur_layer)
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


def education(multilayerNeuron: MultilayerNeuron, education_set_norm: pd.DataFrame, education_set: pd.DataFrame):
    for i in range(len(education_set_norm.values)):
        multilayerNeuron.update_input_values(education_set_norm.values[i])
        un_result = un_normalize(multilayerNeuron.run())
        print("result " + str(un_result) + " " + " target " + str(education_set.values[i][-1]) + " predication " + classification_mushroom(int(un_result)))
        multilayerNeuron.calc_full_error(float(education_set.values[i][-1]))
        multilayerNeuron.education_model()
        multilayerNeuron.zeroing_net()


# def test(multilayerNeuron: MultilayerNeuron, test_set_norm: pd.DataFrame, test_set: pd.DataFrame):
#     for i in range(len(test_set.values)):
#         multilayerNeuron.update_input_values(test_set_norm.values[i])
#         un_result = un_normalize(multilayerNeuron.run())
#         print("result " + str(un_result) + " " + " target " + str(test_set.values[i][-1]) + " predication " + classification_mushroom(int(un_result)))
#         multilayerNeuron.zeroing_net()
#
#
# def education_test():
#     x_train, x_test = train_test_split(dataset, test_size=0.25, random_state=42)
#     x_train_norm, x_test_norm,  = train_test_split(dataset_normalize, test_size=0.25, random_state=42)
#     print(str(len(x_train.values)) + " " + str(len(x_test.values)))
#
#     multilayer_neuron = MultilayerNeuron(create_net([len(x_train.columns) - 1, 4, 3]))
#     print("part  education: ")
#     education(multilayer_neuron, x_train_norm, x_train)
#     print("part test: ")
#     test(multilayer_neuron, x_test_norm, x_test)

# def run(multilayerNeuron: MultilayerNeuron, dataset_normalize: pd.DataFrame, dataset: pd.DataFrame):
#     max_diff = 0.0
#
#     for i in range(len(dataset.values)):
#         multilayerNeuron.update_input_values(dataset_normalize.values[i])
#         result = multilayerNeuron.run()
#         un_result = un_normalize(result)
#         # print("result " + str(un_result) + " " + " target " + str(dataset.values[i][-1]) +
#         #       " predication " + classification_mushroom(int(un_result)))
#         dif = abs(un_result - dataset.values[i][-1])
#         if dif > max_diff:
#             max_diff = dif
#         multilayerNeuron.calc_full_error(float(dataset.values[i][-1]))
#         multilayerNeuron.education_model()
#         multilayerNeuron.zeroing_net()
#     print("iteration: " + " max diff: " + str(max_diff))
#
# if __name__ == "__main__":
#     dataset = pd.read_csv('laptop_price.csv')
#     max_value = dataset['Price'].max()
#     min_value = dataset['Price'].min()
#     dataset_normalize = load_and_normalize_data('mushrooms_filtered.csv')
#     multilayer_neuron = MultilayerNeuron(create_net([len(dataset_normalize.columns) - 1, 5, 5]))
    # for iteration in range(1000):
    # x_train, x_test = train_test_split(dataset, test_size=0.25, random_state=42)
    # x_train_norm, x_test_norm,  = train_test_split(dataset_normalize, test_size=0.25, random_state=42)
    # for iteration in range(1000):
    #     run(multilayer_neuron, dataset_normalize, dataset)

    # education_test()

def run(per: MultilayerNeuron):
    max_diff = 0.0
    for i in range(len(laptops.values)):
        per.update_input_values(laptops.values[i])
        un_result = un_normalize(per.run())
        dif = abs(un_result - laptops1.values[i][-1])
        print("result " + str(un_result) + " " + " target " + str(laptops1.values[i][-1]) +
              " predication " + classification_mushroom(int(un_result)))

        if dif > max_diff:
            max_diff = dif
        per.calc_full_error(laptops.values[i][-1])
        per.education_model()
        per.zeroing_net()
    print("iteration: " + " max diff: " + str(max_diff))


if __name__ == "__main__":
    laptops1 = pd.read_csv('mushrooms_filtered.csv')
    max_value = laptops1['poisonous'].max()
    min_value = laptops1['poisonous'].min()
    laptops = load_and_normalize_data('mushrooms_filtered.csv')
    multilayer_neuron = MultilayerNeuron(create_net([len(laptops1.columns) - 1, 4, 3]))
    # for iteration in range(1000):
    run(multilayer_neuron)