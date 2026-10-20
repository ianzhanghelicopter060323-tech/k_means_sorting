import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def model(n_clusters: int) -> KMeans:
    """
    定义了聚类模型
    参数 n_clusters: int 定义聚类簇个数

    """
    estimator = KMeans(n_clusters=n_clusters, n_init=10)  # 构造聚类器
    return estimator


def train(estimator: KMeans, X: NDArray[np.float64]) -> None:
    """使用二维特征数组 X 训练传入的模型，直接更新 estimator。"""
    estimator.fit(X) 
