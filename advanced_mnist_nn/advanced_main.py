import time

import matplotlib.pyplot as plt
import numpy as np

from advanced_nn_mnist_loader import MNISTLoader
from advanced_nn_mnist_network import NeuralNetwork


def visualize_predictions(network, X, y, num_samples=10):
    """
    可视化网络的预测结果
    展示原始图像、真实标签和预测标签的对比
    """
    predictions = network.predict(X)
    true_labels = np.argmax(y, axis=1)

    plt.figure(figsize=(20, 4))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap="gray")
        plt.title(f"True: {true_labels[i]}\nPred: {predictions[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def analyze_performance(network, X, y):
    """
    详细分析网络在测试集上的表现
    计算混淆矩阵和各个数字的识别准确率
    """
    predictions = network.predict(X)
    true_labels = np.argmax(y, axis=1)

    # 计算混淆矩阵
    conf_matrix = np.zeros((10, 10))
    for t, p in zip(true_labels, predictions):
        conf_matrix[t, p] += 1

    # 计算每个数字的准确率
    per_digit_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # 添加数值标注
    for i in range(10):
        for j in range(10):
            plt.text(j, i, int(conf_matrix[i, j]), ha="center", va="center")

    plt.figure(figsize=(10, 5))
    plt.bar(range(10), per_digit_accuracy)
    plt.ylim(0, 1)
    plt.xlabel("Digit")
    plt.ylabel("Accuracy")
    plt.title("Recognition Accuracy by Digit")

    for i, acc in enumerate(per_digit_accuracy):
        plt.text(i, acc, f"{acc:.2%}", ha="center", va="bottom")

    plt.show()

    return conf_matrix, per_digit_accuracy


def main():
    # 1. 加载并预处理数据
    print("正在加载MNIST数据集...")
    loader = MNISTLoader()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_data()

    # 2. 创建神经网络
    # 使用三层架构：784(输入) -> 128 -> 64 -> 10(输出)
    print("\n创建神经网络...")
    network = NeuralNetwork([784, 128, 64, 10])

    # 3. 设置训练参数
    training_params = {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.001,  # 使用更小的初始学习率
        "verbose": True,
    }

    # 4. 训练网络
    print("\n开始训练...")
    start_time = time.time()
    network.train(X_train, y_train, X_val, y_val, **training_params)
    training_time = time.time() - start_time
    print(f"\n训练完成！总用时: {training_time:.2f}秒")

    # 5. 显示训练历史
    print("\n绘制训练历史...")
    network.plot_training_history()

    # 6. 在测试集上评估
    print("\n在测试集上评估模型...")
    test_accuracy = network.compute_accuracy(X_test, y_test)
    print(f"\n测试集准确率: {test_accuracy:.2%}")

    # 7. 详细性能分析
    print("\n生成详细的性能分析...")
    conf_matrix, per_digit_accuracy = analyze_performance(network, X_test, y_test)

    # 8. 可视化一些预测结果
    print("\n显示一些预测示例...")
    visualize_predictions(network, X_test, y_test)

    # 9. 保存一些有趣的统计数据
    print("\n模型统计信息:")
    print(f"每类数字的识别准确率:")
    for digit, acc in enumerate(per_digit_accuracy):
        print(f"数字 {digit}: {acc:.2%}")

    # 计算最容易和最难识别的数字
    easiest_digit = np.argmax(per_digit_accuracy)
    hardest_digit = np.argmin(per_digit_accuracy)
    print(
        f"\n最容易识别的数字: {easiest_digit} (准确率: {per_digit_accuracy[easiest_digit]:.2%})"
    )
    print(
        f"最难识别的数字: {hardest_digit} (准确率: {per_digit_accuracy[hardest_digit]:.2%})"
    )

    # 10. 错误案例分析
    print("\n分析一些错误案例...")
    predictions = network.predict(X_test)
    true_labels = np.argmax(y_test, axis=1)
    errors = predictions != true_labels
    error_indices = np.where(errors)[0]

    if len(error_indices) > 0:
        plt.figure(figsize=(20, 4))
        for i in range(min(10, len(error_indices))):
            idx = error_indices[i]
            plt.subplot(2, 5, i + 1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
            plt.title(f"True: {true_labels[idx]}\nPred: {predictions[idx]}")
            plt.axis("off")
        plt.suptitle("Error Cases Analysis")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
