import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

# A histogram provides a visual representation of the distribution of a dataset: location,
# spread and skewness of the data; it also helps to visualize whether the distribution is symmetric or skewed left or right.

# In addition, if it is unimodal, bimodal or multimodal. It can also show any outliers or gaps in the data.

# In brief, a histogram summarizes the distribution properties of a continuous numerical variable.


def save_histograme(M_data, B_data, i):
    plt.clf()
    plt.hist(M_data[:, i].tolist(), label="Malignant", fill=True, alpha=0.5)
    plt.hist(B_data[:, i].tolist(), label="Benign", fill=True, alpha=0.5)
    plt.xlabel("feature " + str(i + 1))
    plt.ylabel("Count")
    plt.title("Histogram of feature " + str(i + 1))
    plt.legend(loc="upper right")
    target = "Histograms/" + "%s" % ("feature " + str(i + 1)) + ".png"
    plt.savefig(target)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        type=str,
        help="Path to .csv file",
    )
    args = parser.parse_args()

    try:
        data = pd.read_csv(
            args.filename, header=None, usecols=[i for i in range(1, 32)]
        )
        data = data.to_numpy()
    except Exception:
        print("Can't open file.")
        exit()

    n_feature = len(data[0]) - 1
    B_data = data[data[:, 0] == "M"][:, 1:]
    M_data = data[data[:, 0] == "B"][:, 1:]

    if not os.path.exists("Histograms/"):
        os.mkdir("Histograms/")

    print("Histograms will be saved as images in 'Histograms/' folder")
    for i in range(n_feature):
        save_histograme(M_data, B_data, i)
