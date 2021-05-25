
import config as cfg
import numpy as np
from mlp import Model
from data_processing import Data


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

        loss = Model.softmax_crossentropy_logits(raw_preds, y)
        print()
        print(f"Label {label}")
        print(
            f"\tAccuracy={accuracy:.3f} Precision={precision:.3f} Recall={recall:.3f} F1={F1:.3f} loss={loss:.3f}"
        )
        print(
            f"\tTP = {true_positives}, FP = {false_positive}, TN = {true_negative}, FN = {false_negative}")


def one_hot(data, n):
    ret = np.zeros((len(data), n))
    for i, val in enumerate(data.flat):
        ret[i, val] = 1
    return ret


if __name__ == "__main__":
    model = Model.load("networks/mymodel.mlp")
    data = Data("datasets/data.csv")
    X = model.scale_data(data.raw_x)
    preds = model.predict(X)
    y = data.one_hot_encoding((data.raw_y))
    confusion_matrix(preds, y, model.forward(X)[-1])
