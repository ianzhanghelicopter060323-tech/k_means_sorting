from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def _save_and_show(fig: Figure, filename: str) -> None:
    """保存到 output/baseline，先保存再显示，关闭后释放画布。"""
    output_dir = Path(__file__).resolve().parent.parent / 'output' / 'baseline'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_dir / filename), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_dataset(X: NDArray[np.float64]) -> None:
    """绘制原始样本的前两个特征，保存并显示数据分布图。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    # 取前两个维度（萼片长度、萼片宽度），绘制数据分布图
    ax.scatter(
        X[:, 0], X[:, 1],
        c='red', marker=MarkerStyle('o'), label='Samples',
    )

    ax.set_title('Iris: feature distribution')
    ax.set_xlabel('Sepal length (cm)')
    ax.set_ylabel('Sepal width (cm)')
    ax.legend(loc=2)
    ax.grid(alpha=0.25)
    _save_and_show(fig, 'dataset.png')


def plot_after_traning(
    estimator: KMeans,
    X: NDArray[np.float64],
) -> None:
    """绘制三个簇在前两个特征上的投影，簇编号不代表真实品种。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    label_pred = estimator.labels_  # 获取聚类标签

    # 沿用任务书的三簇分组；模型使用全部四个特征，图中只展示前两个。
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    x2 = X[label_pred == 2]
    ax.scatter(
        x0[:, 0], x0[:, 1],
        c='red', marker=MarkerStyle('o'), label='Cluster 0',
    )
    ax.scatter(
        x1[:, 0], x1[:, 1],
        c='green', marker=MarkerStyle('*'), label='Cluster 1',
    )
    ax.scatter(
        x2[:, 0], x2[:, 1],
        c='blue', marker=MarkerStyle('+'), label='Cluster 2',
    )

    ax.set_title('K-means: 3 clusters (trained on 4 features)')
    ax.set_xlabel('Sepal length (cm)')
    ax.set_ylabel('Sepal width (cm)')
    ax.legend(loc=2)
    ax.grid(alpha=0.25)
    _save_and_show(fig, 'kmeans_result.png')
