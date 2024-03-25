import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from neural_network.task2.multilayerneuron.ActivationNeuron import ActivationNeuron
from neural_network.task2.multilayerneuron.LinkCreator import LinkCreator
from neural_network.task2.multilayerneuron.MultilayerNeuron import MultilayerNeuron
from neural_network.task2.utils.SigmoidActivation import SigmoidActivation
from sklearn.model_selection import train_test_split


def classification_mushroom(prediction: int) -> chr:
    if prediction == 111:
        return 'p'
    return 'e'


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


def create_neuron_layer(neuron_layers, previous_layer, layer_index, layer_size, sigmoid_activation):
    current_layer = []
    for i in range(layer_size):
        links_to_next_layer = LinkCreator.create_links(previous_layer, weights)

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


def print_mera_accuracy(tp_count: int, total: int):
    print("tp count " + str(tp_count) + " mera accuracy " + str(tp_count / total))


def test(multilayerNeuron: MultilayerNeuron, test_set_norm: pd.DataFrame, test_set: pd.DataFrame):
    tp_count = 0
    for i in range(len(test_set.values)):
        multilayerNeuron.update_input_values(test_set_norm.values[i])
        un_result = un_normalize(multilayerNeuron.run())
        classification_result: chr = classification_mushroom(int(un_result))

        if ord(classification_result) == test_set.values[i][-1]:
            tp_count += 1

        print("result " + str(un_result) + " " + " target " + str(
            test_set.values[i][-1]) + " predication " + classification_mushroom(int(un_result))
              )
        multilayerNeuron.zeroing_net()

    print_mera_accuracy(tp_count, len(test_set.values))


def education_test(dataset_origin: pd.DataFrame, dataset_normal: pd.DataFrame):
    x_train, x_test = train_test_split(dataset_origin, test_size=0.25, random_state=42)
    x_train_norm, x_test_norm, = train_test_split(dataset_normal, test_size=0.25, random_state=42)
    print(str(len(x_train.values)) + " " + str(len(x_test.values)))

    multilayer_neuron = MultilayerNeuron(create_net([len(x_train.columns) - 1, 4, 3]))
    print("part  education: ")
    education(multilayer_neuron, x_train_norm)
    print("part test: ")
    test(multilayer_neuron, x_test_norm, x_test)


if __name__ == "__main__":
    dataset = pd.read_csv('mushrooms_filtered.csv')
    max_value = dataset['poisonous'].max()
    min_value = dataset['poisonous'].min()
    dataset_normalize = load_and_normalize_data('mushrooms_filtered.csv')
    education_test(dataset, dataset_normalize)
