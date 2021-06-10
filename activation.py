import numpy as np


# Rectified Linear Unit
class ReLU:
    def __init__(self):
        pass

    def __str__(self):
        return "ReLU"

    def forward(self, inputs):
        return np.maximum(0, inputs)

    def backward(self, inputs, grad_output=None, lr=None):
        # Compute gradient of loss w.r.t. ReLU input
        relu_grad = inputs > 0
        return grad_output*relu_grad


class Softmax:
    def __init__(self):
        pass

    def __str__(self):
        return "Softmax"

    def forward(self, inputs):
        exps = np.exp(inputs)
        return exps / np.sum(exps, axis=1).reshape(len(exps), 1)

    def grad(self, pred_logits, y):
        return (pred_logits - y) / pred_logits.shape[0]
