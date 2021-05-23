import numpy as np
import config as cfg
from dense import Dense
from activation import ReLU, Softmax


class Model:
    def __init__(self, layers, data):
        self.network = self.__create_network(layers)
        self.data = data
        self.train_cost_log = None
        self.val_cost_log = None
        self.train_log = None
        self.val_log = None
        self.lr_log = None

    @staticmethod
    def __create_network(layers):
        network = []
        n_layers = len(layers) - 1
        for i in range(n_layers):
            network.append(Dense(*layers[i:i+2]))
            if i + 1 < n_layers:
                network.append(ReLU())
            else:
                network.append(Softmax())
        return network

    def train(self):
        batch_size = cfg.batch_size
        epochs = cfg.epochs
        lr = cfg.learning_rate
        dyn_lr = cfg.dynamic_learning_rate
        lr_multiplier = cfg.learning_rate_multiplier
        lr_n_changes = cfg.learning_rate_change_number
        lr_epoch_change = epochs // lr_n_changes

        train_log = []
        val_log = []
        train_cost_log = []
        val_cost_log = []
        lr_log = []

        for ep in range(epochs):
            train_cost = []
            val_cost = []
            for X_batch, y_batch in self.get_minibatches(self.data.X_train, self.data.y_train, batch_size):
                train_cost.append(self.train_batch(X_batch, y_batch, lr))
                val_cost.append(self.compute_loss(
                    self.data.X_val, self.data.y_val))

            train_log.append(self.score(self.predict(
                self.data.X_train), self.data.y_train))
            val_log.append(self.score(self.predict(
                self.data.X_val), self.data.y_val))

            train_cost_log.append(np.mean(train_cost))
            val_cost_log.append(np.mean(val_cost))

            lr_log.append(lr)

            print(
                f"epoch {ep}/{epochs} - loss: {train_cost_log[-1]} - val_loss: {val_cost_log[-1]}")

        self.train_cost_log = train_cost_log
        self.val_cost_log = val_cost_log
        self.train_log = train_log
        self.val_log = val_log
        self.lr_log = lr_log

        return train_cost_log, val_cost_log, train_log, val_log, lr_log

    def compute_loss(self, X, y):
        val_preds = self.forward(X)[-1]
        val_loss = self.softmax_crossentropy_logits(val_preds[:, 1], y[:, 1])
        return val_loss

    @staticmethod
    def get_minibatches(X, y, batch_size):
        indices = np.random.permutation(len(y))
        for i in range(0, len(y) - batch_size + 1, batch_size):
            selection = indices[i:i+batch_size]
            yield X[selection], y[selection]

    def train_batch(self, X, y, lr):
        activations = self.forward(X)
        inputs = [X] + activations
        logits = activations[-1]

        loss = self.softmax_crossentropy_logits(logits[:, 1], y[:, 1])

        loss_grad = self.network[-1].grad(logits, y)

        for i in range(len(self.network) - 1)[::-1]:
            layer = self.network[i]
            loss_grad = layer.backward(inputs[i], loss_grad, lr)

        return loss

    def forward(self, X):
        activations = []
        input = X
        for l in self.network:
            activations.append(l.forward(input))
            input = activations[-1]
        assert len(activations) == len(self.network), "Error"
        return activations

    @staticmethod
    def softmax_crossentropy_logits(pred_logits, y):
        if len(pred_logits.shape) == 1:
            pred_logits = pred_logits.reshape((pred_logits.shape[0], 1))
            y = y.reshape((y.shape[0], 1))
        a = y * np.log(pred_logits + 1e-15)
        b = (1 - y) * np.log(1 - pred_logits + 1e-15)
        c = (a + b).sum(axis=1)
        if -np.sum(c) / len(y) == np.NaN:
            exit()

        return -np.sum(c) / len(y)

    def score(self, y_pred, y_true):
        return np.mean(y_pred == y_true[:, 1])

    def predict(self, X):
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)

# len = 9
# batch = 2
# 1 2 3 4 5 6 7 8 9
