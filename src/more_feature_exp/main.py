from __future__ import annotations

import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from data_prepare import get_dataset, select_features
from model_set import evaluate, model, train
from visualization import plot_clusters, plot_metrics


OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'output' / 'more_feature_exp'
RANDOM_SEEDS = [*range(30), 42]

# 每项依次为：名称、图中标签、特征列、是否标准化、初始化次数。
# 2 -> 3 -> 4 特征对照均使用原始厘米值、n_init=50 和相同种子。
EXPERIMENTS = [
    ('raw_4_n10', '4D raw\nn_init=10', (0, 1, 2, 3), False, 10),
    ('raw_4_n50', '4D raw\nn_init=50', (0, 1, 2, 3), False, 50),
    ('scaled_4_n50', '4D scaled\nn_init=50', (0, 1, 2, 3), True, 50),
    ('petal_2_n50', '2D petals\nn_init=50', (2, 3), False, 50),
    ('petal_sepal_3_n50', '3D petals + SL\nn_init=50', (2, 3, 0), False, 50),
]


def save_csv(path: Path, rows: list[dict]) -> None:
    """保留未四舍五入的数值，方便独立复核和后续制表。"""
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiments() -> None:
    """完成特征数、初始化次数和标准化对照，保存指标及绘图数据。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = get_dataset()
    all_results = []
    summary = []
    predictions = {}
    matched_predictions = {}

    # 小数据使用单线程，避免线程调度开销；不同设置使用相同的一组种子。
    with threadpool_limits(limits=1):
        for name, plot_label, columns, standardize, n_init in EXPERIMENTS:
            features = select_features(X, columns, standardize)
            case_results = []
            for seed in RANDOM_SEEDS:
                estimator = model(n_clusters=3, n_init=n_init, random_state=seed)
                train(estimator, features)
                metrics, aligned_labels = evaluate(estimator, features, y)
                result = {
                    'experiment': name,
                    'n_features': len(columns),
                    'feature_indices': ','.join(map(str, columns)),
                    'standardized': standardize,
                    'n_init': n_init,
                    'random_state': seed,
                    'n_samples': len(y),
                    'n_iter': int(estimator.n_iter_),
                    **metrics,
                }
                case_results.append(result)
                if seed == 42:
                    predictions[name] = estimator.labels_.copy()
                    matched_predictions[name] = aligned_labels

            all_results.extend(case_results)
            reference = next(row for row in case_results if row['random_state'] == 42)
            row = {'experiment': name, 'plot_label': plot_label, 'n_runs': len(RANDOM_SEEDS)}
            for metric in ('ari', 'accuracy', 'silhouette', 'inertia'):
                values = np.array([result[metric] for result in case_results])
                row.update({
                    f'{metric}_seed42': reference[metric],
                    f'{metric}_mean': float(values.mean()),
                    f'{metric}_std': float(values.std(ddof=1)),
                    f'{metric}_min': float(values.min()),
                    f'{metric}_max': float(values.max()),
                })
            summary.append(row)

    save_csv(OUTPUT_DIR / 'metrics.csv', all_results)
    save_csv(OUTPUT_DIR / 'summary.csv', summary)
    prediction_rows = []
    for index in range(len(y)):
        # 预测记录同时包含整数标签和浮点特征，不能按纯整数字典推断。
        prediction_row: dict[str, float] = {
            'sample_index': index,
            'true_label': int(y[index]),
        }
        prediction_row.update({f'feature_{j}': float(X[index, j]) for j in range(4)})
        for name in predictions:
            prediction_row[f'{name}_cluster'] = int(predictions[name][index])
            prediction_row[f'{name}_matched_label'] = int(matched_predictions[name][index])
        prediction_rows.append(prediction_row)
    save_csv(OUTPUT_DIR / 'predictions_seed42.csv', prediction_rows)

    parameters = {
        'dataset': 'sklearn.datasets.load_iris',
        'shape': list(X.shape),
        'feature_names': ['sepal length', 'sepal width', 'petal length', 'petal width'],
        'feature_unit': 'cm',
        'random_seeds': RANDOM_SEEDS,
        'kmeans': {'n_clusters': 3, 'init': 'k-means++', 'max_iter': 300, 'tol': 1e-4, 'algorithm': 'lloyd'},
        'experiments': [
            {'name': name, 'feature_indices': columns, 'standardized': scaled, 'n_init': n_init}
            for name, _, columns, scaled, n_init in EXPERIMENTS
        ],
        'evaluation': 'All 150 samples; labels only for post-fit evaluation and display; no held-out test set.',
        'std_ddof': 1,
        'versions': {
            'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'matplotlib': version('matplotlib'),
        },
    }
    (OUTPUT_DIR / 'parameters.json').write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )
    plot_metrics(summary, OUTPUT_DIR)
    plot_clusters(X, y, matched_predictions, OUTPUT_DIR)
    save_description(summary)


def save_description(summary: list[dict]) -> None:
    """保存实际结果、运行方法和指标限制，不预先写定实验结论。"""
    lines = [
        '鸢尾花 K-means 特征数量对照实验',
        '运行（项目根目录）：MPLBACKEND=Agg .venv/bin/python src/more_feature_exp/main.py',
        '数据：150 个样本，4 个原始厘米特征；全部样本用于聚类与事后评价。',
        '随机种子：0 至 29，以及 42；共 31 次。标准差使用 ddof=1。',
        'ARI：调整兰德指数；ACC：簇编号与真实品种一对一最优匹配后的准确率。',
        '真实标签不参与特征预处理或模型训练；ACC 不是独立测试集分类准确率。',
        '',
        '设置 | seed42 ARI | seed42 ACC | 多种子 ARI 均值±标准差 | ACC 均值±标准差',
    ]
    for row in summary:
        lines.append(
            f"{row['experiment']} | {row['ari_seed42']:.4f} | {row['accuracy_seed42']:.4f} | "
            f"{row['ari_mean']:.4f}±{row['ari_std']:.4f} | "
            f"{row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}"
        )
    by_name = {row['experiment']: row for row in summary}
    two = by_name['petal_2_n50']
    four = by_name['raw_4_n50']
    lines.extend([
        '',
        '主要受控对照：花瓣长度+花瓣宽度 -> 再加萼片长度 -> 再加萼片宽度。',
        '三组均使用 n_init=50、原始厘米特征和相同种子。其余两组分别检查初始化次数与标准化。',
        f"从 2 特征增加到 4 特征，多种子平均 ARI 变化为 {four['ari_mean'] - two['ari_mean']:+.4f}，"
        f"平均 ACC 变化为 {four['accuracy_mean'] - two['accuracy_mean']:+.4f}。",
        '本实验说明更多输入特征不保证更高的品种一致性；不推出增加特征一定无效或有害。',
        '特征选择属于探索性分析；多种子只检查初始化稳定性，不代表独立样本或泛化验证。',
        'silhouette 在各自训练空间计算；inertia 也依赖维数与尺度，不能跨特征设置直接据此排名。',
        '图片使用同一原始坐标，簇颜色仅在训练结束后对齐，图上的重叠不等同于聚类错误。',
        '图片仅保存文件，不弹出窗口。summary.csv 为汇总，metrics.csv 为逐次运行记录，',
        'predictions_seed42.csv 保存原始特征、真实标签、原始簇编号及匹配标签。',
    ])
    description = '\n'.join(lines) + '\n'
    (OUTPUT_DIR / 'results.txt').write_text(description, encoding='utf-8')
    print(description)


if __name__ == '__main__':
    run_experiments()
