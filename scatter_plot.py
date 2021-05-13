import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

# A scatter plot is an excellent tool for comparing pairs of values to see if they are related.


def save_scatter_plot(M_data, B_data, i, j):
    plt.clf()
    plt.scatter(
        M_data[:, i].tolist(),
        M_data[:, j].tolist(),
        label="Malignant",
        alpha=0.5,
    )
    plt.scatter(
        B_data[:, i].tolist(),
        B_data[:, j].tolist(),
        label="Benign",
        alpha=0.5,
    )
    i_feature = "feature " + str(i + 1)
    j_feature = "feature " + str(j + 1)
    plt.xlabel(i_feature)
    plt.ylabel(j_feature)
    plt.title('%s , %s' % (i_feature, j_feature))
    plt.legend(loc='upper right')
    target = "Scatter_plot/" + '%s , %s' % (i_feature, j_feature) + "png"
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

    if not os.path.exists("Scatter_plot/"):
        os.mkdir("Scatter_plot/")

    print("scatter plots will be saved as images in 'Scatter_plot/' folder.")
    for i in range(0, n_feature):
        for j in range(i + 1, n_feature):
            save_scatter_plot(M_data, B_data, i, j)
