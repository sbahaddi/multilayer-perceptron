import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os


def save_pairPlots(M_data, B_data):
    fig, ax = plt.subplots(30, 30, figsize=(30, 30))
    for i in range(0, n_feature):
        for j in range(0, n_feature):
            if (i == j):
                ax[i][j].hist(M_data[:, i].tolist(),
                              label="Malignant", fill=True, alpha=0.5)
                ax[i][j].hist(B_data[:, i].tolist(),
                              label="Benign", fill=True, alpha=0.5)
            else:
                ax[i][j].scatter(M_data[:, i].tolist(), M_data[:,
                                                               j].tolist(), label="Malignant", alpha=0.5,)
                ax[i][j].scatter(B_data[:, i].tolist(), B_data[:,
                                                               j].tolist(), label="Benign", alpha=0.5,)
            if i == 0:
                ax[i][j].set_title("feature " + str(j + 1))
            if j == 0:
                ax[i][j].set_ylabel("feature " + str(i + 1))
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            print('creating pair plot feature %02d vs feature %02d ' %
                  (i + 1, j + 1))
    fig.tight_layout()
    plt.savefig("pair_plots/pair_plot.png")
    print("Pair Plot saved..")

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

    if not os.path.exists("pair_plots/"):
        os.mkdir("pair_plots/")

    print("pair plots will be saved as image in 'pair_plots/' folder")
    save_pairPlots(M_data, B_data)
