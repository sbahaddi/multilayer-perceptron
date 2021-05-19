from dense import Dense
from activation import ReLU, Softmax

class Model:
    def __init__(self, layers):
        self.network = self.__create_network(layers)
        self.test = True
        self.x_max = None
        self.x_min = None
        self.cost_log = None
        self.train_log = None
        self.val_log = None
        self.lr_log = None

    def scale_data(self, X):
        if type(self.x_max):
            self.x_max = X.max(axis=0)
            self.x_min = X.min(axis=0)
        return (X - self.x_min)/self.x_max

    @staticmethod
    def __create_network(layers):
        network=[]
        n_units = layers
        n_layers = len(n_units) - 1
        for i in range(n_layers):
            network.append(Dense(*n_units[i:i+2]))
            if i + 1 < n_layers:
                network.append(ReLU)
            else:
                network.append(Softmax)
        return network