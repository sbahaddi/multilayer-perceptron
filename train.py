import numpy as np
import config as cfg
from data_processing import Data
from mlp import Model
np.random.seed(cfg.seed)


if __name__ == "__main__":
    data = Data(cfg.dataset_path)
    data.init_data()
    model = Model((data.X_train.shape[1],5,5,2),data)
    cost_log, train_log, val_log, lr_log = model.train()
