import numpy as np

class ReLU():
    def __init__(self):
        pass

    def __str__(self):
        return "ReLU"

    def forward(self, input_):
        return np.maximum(1e-3 * input_, input_)

    def backward(self, input_, grad_output=None, lr=None):
        relu_grad = input_ > 0
        relu_grad = np.array([[1 if l else 1e-3 for l in i]
                             for i in relu_grad])
        return grad_output * relu_grad


class Softmax():
    def __init__(self):
        pass

    def __str__(self):
        return "Softmax"

    def forward(self, input_):
        exps = np.exp(input_ - np.max(input_))
        return exps / np.sum(exps, axis=-1).reshape(len(exps), 1)

    def grad(self, pred_logits, y):
        return (pred_logits - y) / pred_logits.shape[0]

    # def softmax(self, x):
    #     exps = np.exp(x - np.max(x))
    #     return exps / np.sum(exps, axis=-1).reshape(len(exps), 1)
