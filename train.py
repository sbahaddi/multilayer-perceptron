import numpy as np
from data_processing import Data

if __name__ == "__main__":
    data = Data("datasets/data.csv")
    data.init_data()
    print(data.raw_data)
    print("--------------------------")
    np.random.shuffle(data.raw_data)
    print(data.raw_data)
