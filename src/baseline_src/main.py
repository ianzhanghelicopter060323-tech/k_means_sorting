from data_prepare import get_dataset
from visualization import plot_dataset, plot_after_traning
from model_set import model, train


if __name__ == '__main__':
    dataset = get_dataset()

    # 绘制并保存原始数据分布图。
    plot_dataset(dataset)

    # 训练
    estimator = model(n_clusters=3)
    train(estimator, dataset)

    # 训练后画图
    plot_after_traning(estimator=estimator, X=dataset)
