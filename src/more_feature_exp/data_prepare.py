from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def get_dataset() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """读取 150 个鸢尾花样本；真实品种标签仅用于训练后的评价。"""
    X, y = datasets.load_iris(return_X_y=True)
    return X, y


def select_features(
    X: NDArray[np.float64],
    columns: tuple[int, ...],
    standardize: bool,
) -> NDArray[np.float64]:
    """按列选择特征，并按实验设置决定是否标准化。"""
    # 原始列依次为：萼片长度、萼片宽度、花瓣长度、花瓣宽度。
    features = X[:, columns]
    if standardize:
        features = StandardScaler().fit_transform(features)
    return features
