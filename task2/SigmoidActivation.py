import random
import math


class SigmoidActivation:
    def __init__(self):
        self.activations_func = [
            self.activation1, self.activation2, self.activation3, self.activation4,
            self.activation5, self.activation6, self.activation7, self.activation8
        ]

    @staticmethod
    def sigmoid(x: float, alpha: float) -> float:
        return 1 / (1 + math.exp(-alpha * x))

    def activation1(self, x: float) -> float:
        return self.sigmoid(x, 0.3)

    def activation2(self, x: float) -> float:
        return self.sigmoid(x, 0.5)

    def activation3(self, x: float) -> float:
        return self.sigmoid(x, 0.7)

    def activation4(self, x: float) -> float:
        return self.sigmoid(x, -0.3)

    def activation5(self, x: float) -> float:
        return self.sigmoid(x, 0.1)

    def activation6(self, x: float) -> float:
        return self.sigmoid(x, -0.2)

    def activation7(self, x: float) -> float:
        return self.sigmoid(x, 0.25)

    def activation8(self, x: float) -> float:
        return self.sigmoid(x, 0.9)

    def get_activation_function(self):
        return self.activations_func[random.randint(0, len(self.activations_func) - 1)]

    @staticmethod
    def x(x: float) -> float:
        return x
