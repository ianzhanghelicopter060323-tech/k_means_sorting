import numpy as np
from numpy.typing import NDArray
from sklearn import datasets


def get_dataset() -> NDArray[np.float64]:
    """返回形状为 (150, 4) 的鸢尾花特征数组。"""
    X, __ = datasets.load_iris(return_X_y=True)
    return X[:, :4]
