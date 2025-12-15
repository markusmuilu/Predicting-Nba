import numpy as np


def evaluate( y_pred, y_true):
    """
        To replica official models, lets return Accuracy, precision, recall and f1 score to show
        model performance.
    """


    TP, FP, FN, TN = compute_confusion(y_true, y_pred)

    acc = accuracy_score(TP, FP, FN, TN)
    precision = precision_score(TP, FP)
    recall = recall_score(TP, FN)
    f1 = f1_score(precision, recall)

    print("\nEvaluation Results")
    print("------------------")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("------------------\n")

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN)
    }

def compute_confusion(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    return TP, FP, FN, TN


def accuracy_score(TP, FP, FN, TN):
    return (TP + TN) / (TP + FP + FN + TN + 1e-8)


def precision_score(TP, FP):
    return TP / (TP + FP + 1e-8)


def recall_score(TP, FN):
    return TP / (TP + FN + 1e-8)


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-8)
