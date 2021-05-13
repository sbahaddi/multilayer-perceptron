import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

# From this visualization, what features are you going to use for your logistic regression?

# Features: "Arithmancy and care of magic creatures" are homogenous across the classes and can be eliminated
# "Astronomy and Defense against the dark arts are correlated", this means we can eliminate one of them.

# so we eliminate Arithmancy, care of magic creatures and Astronomy
# Eliminating features speeds up the algorithm


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
        #data = data.to_numpy()
    except Exception:
        print("Can't open file.")
        exit()

    # print(data[1])
    sns.set_theme(font_scale=0.63)
    g = sns.pairplot(data, hue=1, palette="husl", height=1.1)
    plt.show()
