import numpy as np
import config as cfg
from data_processing import Data
from mlp import Model

np.random.seed(cfg.seed)


def one_hot(data, n):
    ret = np.zeros((len(data), n))
    for i, val in enumerate(data.flat):
        ret[i, val] = 1
    return ret


def validate_model(val_pred, val, val_raw):
    predicted = one_hot(val_pred, 2)
    for i, label in enumerate(cfg.categories):
        preds = predicted[:, i]
        y = val[:, i]
        raw_preds = val_raw[:, i]

        true_positives = np.sum((preds == 1) * (y == 1))
        true_negative = np.sum((preds == 0) * (y == 0))
        false_positive = np.sum(preds > y)
        false_negative = np.sum(preds < y)
        pred_count = len(preds)

        accuracy = (true_negative + true_positives) / pred_count

        if true_positives + false_positive == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positive)

        if true_positives + false_negative == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negative)

        if precision + recall == 0:
            F1 = 0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)

        loss = Model.softmax_crossentropy_logits(raw_preds, y)
        print(
            f"Label {label}: Accuracy={accuracy:.3f} Precision={precision:.3f} Recall={recall:.3f} F1={F1:.3f} loss={loss:.3f}")


if __name__ == "__main__":
    data = Data(cfg.dataset_path)
    data.init_data()
    model = Model((data.X_train.shape[1], 5, 5, 2), data)
    train_cost_log, val_cost_log, train_log, val_log, lr_log = model.train()
    validate_model(model.predict(data.X_val), data.y_val,
                   model.forward(data.X_val)[-1])
