import numpy as np
from data_processing import Data

if __name__ == "__main__":
    data = Data("datasets/data.csv")
    print(data.raw_data)
    data.init_data()
    print("---\n", data.raw_data)
