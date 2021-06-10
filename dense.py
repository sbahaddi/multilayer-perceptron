import numpy as np


class Dense():
    def __init__(self, input_units, output_units):
        """
        A dense layer is a layer which performs a learned affine transformation:
        f(x) = <W*x> + b
        """
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(
                                            2/(input_units+output_units)),
                                        size=(input_units, output_units)).astype(float)
        self.biases = np.zeros(output_units)

    def __str__(self):
        return f"Dense({self.weights.shape[0]})"

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output, lr):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.sum(axis=0)

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step.
        self.weights = self.weights - lr * grad_weights
        self.biases = self.biases - lr * grad_biases

        return grad_input
