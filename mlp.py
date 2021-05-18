
class Model:
    def __init__(self, network, config):
        self.network = self.__build_network(network, config)
        self.test = True
        self.xmax = None
        self.xmin = None
        self.cost_log = None
        self.train_log = None
        self.val_log = None
        self.lr_log = None
        self.config = config
