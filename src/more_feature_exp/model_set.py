from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix, silhouette_score


def model(n_clusters: int, n_init: int, random_state: int) -> KMeans:
    """显式固定初始化方法和随机种子，保持不同实验的参数可比。"""
    return KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=n_init,
        max_iter=300,
        tol=1e-4,
        random_state=random_state,
        algorithm='lloyd',
    )


def train(estimator: KMeans, X: NDArray[np.float64]) -> None:
    """仅使用特征训练聚类模型，不传入真实品种标签。"""
    estimator.fit(X)


def evaluate(
    estimator: KMeans,
    X: NDArray[np.float64],
    y: NDArray[np.int64],
) -> tuple[dict[str, float], NDArray[np.int64]]:
    """计算聚类指标，返回评价用的标签匹配结果。"""
    labels = estimator.labels_
    # 簇编号没有品种语义，先寻找一对一的最优匹配，再计算准确率。
    # 匹配只用于事后评价和统一绘图颜色，不反馈给聚类模型。
    counts = confusion_matrix(y, labels, labels=[0, 1, 2])
    true_ids, cluster_ids = linear_sum_assignment(-counts)
    mapping = np.empty(3, dtype=np.int64)
    mapping[cluster_ids] = true_ids
    aligned_labels = mapping[labels]

    metrics = {
        'ari': float(adjusted_rand_score(y, labels)),
        'accuracy': float(np.mean(aligned_labels == y)),
        'silhouette': float(silhouette_score(X, labels)),
        'inertia': float(estimator.inertia_),
    }
    return metrics, aligned_labels
