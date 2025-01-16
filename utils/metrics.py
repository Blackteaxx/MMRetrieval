import numpy as np


def compute_f1(col):
    def f1(row):
        n = len(np.intersect1d(row["label"], row[col]))
        return 2 * n / (len(row["label"]) + len(row[col]))

    return f1


def compute_recall(col):
    def recall(row):
        n = len(np.intersect1d(row["label"], row[col]))
        return n / len(row["label"])

    return recall


def compute_precision(col):
    def precision(row):
        n = len(np.intersect1d(row["label"], row[col]))
        return n / len(row[col])

    return precision


def compute_precision_K(col, K):
    def precision(row):
        n = len(np.intersect1d(row["label"], row[col][:K]))
        return n / K

    return precision


def compute_AP(col, N):
    """compute average precision"""

    def AP(row):
        n = len(np.intersect1d(row["label"], row[col]))
        max_n = min(len(row[col]), N)
        if n == 0:
            return 0
        return (
            sum(
                [
                    compute_precision_K(col, i)(row)
                    for i in range(1, max_n + 1)
                    if row[col][i - 1] in row["label"]
                ]
            )
            / max_n
        )

    return AP
