from neural_network.task3.rnn.LinkNeuron import LinkNeuron


class LinkCreator:
    index_weight = 0

    @staticmethod
    def create_links(neurons: [], weight):
        links = []
        for neuron in neurons:
            links.append(LinkNeuron(neuron, weight))
            LinkCreator.index_weight += 1
        return links
