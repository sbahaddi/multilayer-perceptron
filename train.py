import numpy as np
import config as cfg
from data_processing import Data


if __name__ == "__main__":
    data = Data(cfg.dataset_path)
    # print(data.raw_data)
    # train,val = data.split_data()
    # print(train)
    # data.init_data()
    # print(data.raw_data)
