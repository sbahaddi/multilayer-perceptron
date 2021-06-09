import numpy as np
import config as cfg
import argparse
import os
from data_processing import Data
from mlp import Model
import matplotlib.pyplot as plt

try:
    np.random.seed(cfg.seed)
except:
    pass


def one_hot(data, n):
    ret = np.zeros((len(data), n))
    for i, val in enumerate(data.flat):
        ret[i, val] = 1
    return ret


def confusion_matrix(val_pred, val, val_raw):
    predicted = one_hot(val_pred, 2)
    print()
    print(f"total = {len(val_pred)}")
    for i, label in enumerate(cfg.categories):
        preds = predicted[:, i]
        y = val[:, i]
        raw_preds = val_raw[:, i]

        true_positives = np.sum((preds == 1) * (y == 1))
        true_negative = np.sum((preds == 0) * (y == 0))
        false_positive = np.sum(preds > y)
        false_negative = np.sum(preds < y)
        pred_count = len(preds)

        # Accuracy: Overall, how often is the classifier correct?
        # (TP+TN)/total
        accuracy = (true_positives + true_negative) / pred_count

        # Precision: When it predicts yes, how often is it correct?
        # TP/(TP+FP)predicted yes
        if true_positives + false_positive == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positive)

        # When it's actually yes, how often does it predict yes?
        #  TP/(TP+FN)actual yes
        if true_positives + false_negative == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negative)

        """
        It is difficult to compare two models with low precision and high recall or vice versa.
        So to make them comparable, we use F-Score.
        F-score helps to measure Recall and Precision at the same time.
        It uses Harmonic Mean in place of Arithmetic Mean by punishing the extreme values more.
        """
        if precision + recall == 0:
            F1 = 0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)

        loss = Model.binary_crossentropy_error(raw_preds, y)
        print()
        print(f"Label {label}")
        print(
            f"\tAccuracy={accuracy:.3f} Precision={precision:.3f} Recall={recall:.3f} F1={F1:.3f} loss={loss:.3f}"
        )
        print(
            f"\tTP = {true_positives}, FP = {false_positive}, TN = {true_negative}, FN = {false_negative}"
        )


def plot_logs(train_log, val_log, train_cost_log, val_cost_log, args):
    plt.figure(figsize=(10, 7.5))
    plt.subplot(212)
    plt.title("Accuracy")
    plt.ylabel("%")
    plt.plot(train_log, label="Train")
    plt.plot(val_log, label="Validation")
    plt.legend()
    plt.grid()

    plt.subplot(221)
    plt.title("Train Loss")
    plt.ylabel("cross-entropy")
    plt.xlabel("epochs")
    plt.plot(range(len(train_cost_log)), train_cost_log, "tab:red", label="Train")
    plt.legend()
    plt.grid()

    plt.subplot(222)
    plt.title("Val Loss")
    plt.ylabel("cross-entropy")
    plt.xlabel("epochs")
    plt.plot(range(len(val_cost_log)), val_cost_log, "tab:red", label="Validation")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    if not os.path.exists("training_plots/"):
        os.mkdir("training_plots/")
    plt.savefig(f"training_plots/{args.savemodel}.png")

    if args.plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cm", action="store_true", help="use a confusion matrix")
    parser.add_argument("-p", "--plot", help="plot training logs", action="store_true")
    parser.add_argument(
        "-s", "--savemodel", help="save model to file", default="model.mlp"
    )
    args = parser.parse_args()

    data = Data(cfg.train_data)
    data.init_data()
    network = (data.X_train.shape[1],) + cfg.layers
    model = Model(network)

    data.X_train = model.scale_data(data.X_train)
    data.X_val = model.scale_data(data.X_val)

    train_cost_log, val_cost_log, train_log, val_log, lr_log = model.train(
        data.X_train, data.y_train, data.X_val, data.y_val
    )

    model.save_to_file(name=args.savemodel)

    if args.cm:
        confusion_matrix(
            model.predict(data.X_val), data.y_val, model.forward(data.X_val)[-1]
        )

    plot_logs(train_log, val_log, train_cost_log, val_cost_log, args)
