from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray


def _save(fig: Figure, output_dir: Path, filename: str) -> None:
    """保存实验图片并释放画布，适用于无桌面环境的批量运行。"""
    fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_metrics(summary: list[dict], output_dir: Path) -> None:
    """展示固定种子结果与多随机种子均值，误差线为样本标准差。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    positions = np.arange(len(summary))
    names = [row['plot_label'] for row in summary]
    for ax, metric, title in zip(
        axes, ('ari', 'accuracy'), ('Adjusted Rand index', 'Matched clustering accuracy'),
    ):
        ax.bar(
            positions - 0.18, [row[f'{metric}_seed42'] for row in summary],
            width=0.36, label='Seed 42', color='steelblue',
        )
        ax.bar(
            positions + 0.18, [row[f'{metric}_mean'] for row in summary],
            yerr=[row[f'{metric}_std'] for row in summary],
            width=0.36, capsize=3, label='31 seeds: mean +/- SD', color='darkorange',
        )
        ax.set_xticks(positions, names, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle('Iris K-means: feature selection, initialization and scaling')
    _save(fig, output_dir, 'metric_comparison.png')


def plot_clusters(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    predictions: dict[str, NDArray[np.int64]],
    output_dir: Path,
) -> None:
    """将不同维数模型放在相同坐标系展示，避免投影不同造成误判。"""
    panels = [
        ('Ground truth', y),
        ('2 features: petals', predictions['petal_2_n50']),
        ('3 features: petals + sepal length', predictions['petal_sepal_3_n50']),
        ('4 features: all', predictions['raw_4_n50']),
    ]
    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8), sharex='row', sharey='row', constrained_layout=True,
    )
    for row, (x_column, y_column, x_label, y_label) in enumerate([
        (0, 1, 'Sepal length (cm)', 'Sepal width (cm)'),
        (2, 3, 'Petal length (cm)', 'Petal width (cm)'),
    ]):
        for column, (title, labels) in enumerate(panels):
            ax = axes[row, column]
            for group, color, marker in zip(range(3), ('red', 'green', 'blue'), ('o', '*', '+')):
                mask = labels == group
                ax.scatter(
                    X[mask, x_column], X[mask, y_column], c=color, marker=marker, s=28,
                    label=('Setosa', 'Versicolor', 'Virginica')[group],
                )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(alpha=0.25)
    fig.suptitle('Seed 42, n_init=50; cluster colors aligned to truth for display only')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.035), ncol=3)
    _save(fig, output_dir, 'cluster_comparison.png')
