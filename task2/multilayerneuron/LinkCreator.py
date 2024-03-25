from neural_network.task2.multilayerneuron.LinkNeuron import LinkNeuron


class LinkCreator:
    index_weight = 0

    @staticmethod
    def create_links(neurons: [], weights: []):
        links = []
        for neuron in neurons:
            links.append(LinkNeuron(neuron, weights[LinkCreator.index_weight]))
            LinkCreator.index_weight += 1
        return links

