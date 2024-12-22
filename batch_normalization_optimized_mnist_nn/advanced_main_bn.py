import time
import matplotlib.pyplot as plt
import numpy as np

from advanced_nn_mnist_loader import MNISTLoader
from advanced_nn_mnist_network_with_bn import NeuralNetwork
from visualized_utils import VisualizationUtils


def train_network(X_train, y_train, X_val, y_val, network_config):
    """
    训练神经网络的主要函数

    Args:
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据
        y_val: 验证标签
        network_config: 网络配置参数字典

    Returns:
        训练好的神经网络模型
    """
    print("\n创建并训练神经网络...")
    network = NeuralNetwork(
        layer_sizes=network_config["layer_sizes"],
        use_batchnorm=network_config["use_batchnorm"],
    )

    # 记录训练开始时间
    start_time = time.time()

    # 训练网络
    network.train(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=network_config["epochs"],
        batch_size=network_config["batch_size"],
        learning_rate=network_config["learning_rate"],
        verbose=True,
    )

    # 计算训练时间
    training_time = time.time() - start_time
    print(f"\n训练完成！总用时: {training_time:.2f}秒")

    return network


def evaluate_and_visualize(network, X_test, y_test, viz_utils):
    """
    评估模型性能并生成可视化结果

    Args:
        network: 训练好的神经网络模型
        X_test: 测试数据
        y_test: 测试标签
        viz_utils: 可视化工具实例
    """
    # 模型评估
    print("\n进行模型评估...")
    evaluation_results = network.evaluate(X_test, y_test)

    # 打印评估结果
    print("\n模型评估结果:")
    for metric, value in evaluation_results.items():
        if isinstance(value, list):
            print(f"\n{metric}:")
            for i, v in enumerate(value):
                print(f"  类别 {i}: {v:.4f}")
        else:
            print(f"{metric}: {value:.4f}")

    # 生成预测和置信度
    print("\n生成预测及置信度分析...")
    predictions, confidences = network.predict_with_confidence(X_test)
    true_labels = np.argmax(y_test, axis=1)

    # 绘制混淆矩阵
    print("\n绘制混淆矩阵...")
    viz_utils.plot_confusion_matrix(
        true_labels, predictions, class_names=[str(i) for i in range(10)]
    )

    # 绘制学习曲线
    print("\n绘制学习曲线...")
    viz_utils.plot_learning_curves(network.training_history)

    # 绘制预测置信度分布
    print("\n绘制预测置信度分布...")
    viz_utils.plot_prediction_confidence(predictions, confidences, true_labels)

    # 分析网络权重
    print("\n分析网络权重分布...")
    for i, layer in enumerate(network.layers):
        viz_utils.visualize_layer_weights(layer.weights, f"Layer {i+1}")

    # 展示预测示例
    print("\n展示预测示例...")
    visualize_predictions(X_test, y_test, predictions, confidences)


def visualize_predictions(X_test, y_test, predictions, confidences, num_samples=10):
    """
    可视化模型预测结果

    Args:
        X_test: 测试数据
        y_test: 测试标签
        predictions: 模型预测结果
        confidences: 预测置信度
        num_samples: 要显示的样本数量
    """
    true_labels = np.argmax(y_test, axis=1)
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        plt.title(
            f"True: {true_labels[idx]}\n"
            f"Pred: {predictions[idx]}\n"
            f"Conf: {confidences[idx]:.2f}"
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def test_model_persistence(network, X_test, y_test, model_path):
    """
    测试模型的保存和加载功能

    Args:
        network: 训练好的神经网络模型
        X_test: 测试数据
        y_test: 测试标签
        model_path: 模型保存路径
    """
    print("\n测试模型保存和加载功能...")

    # 保存模型
    print("\n保存模型...")
    network.save_model(model_path)

    # 加载模型
    print("\n加载模型...")
    loaded_network = NeuralNetwork.load_model(model_path)

    # 比较性能
    print("\n比较原始模型和加载后模型的性能:")
    original_accuracy = network.compute_accuracy(X_test, y_test)
    loaded_accuracy = loaded_network.compute_accuracy(X_test, y_test)

    print(f"原始模型准确率: {original_accuracy:.4f}")
    print(f"加载后模型准确率: {loaded_accuracy:.4f}")

    # 验证准确率是否相同
    assert np.abs(original_accuracy - loaded_accuracy) < 1e-6, "模型加载后性能不一致！"


def main():
    """主函数，协调整个训练和评估过程"""

    # 网络配置
    network_config = {
        "layer_sizes": [784, 256, 128, 10],  # 使用更大的隐藏层
        "use_batchnorm": True,  # 启用批量归一化
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.001,
    }

    # 1. 加载数据
    print("正在加载MNIST数据集...")
    loader = MNISTLoader()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_data()

    # 2. 创建可视化工具实例
    viz_utils = VisualizationUtils()

    # 3. 训练网络
    network = train_network(X_train, y_train, X_val, y_val, network_config)

    # 4. 评估和可视化结果
    evaluate_and_visualize(network, X_test, y_test, viz_utils)

    # 5. 测试模型保存和加载
    test_model_persistence(network, X_test, y_test, "models/mnist_model_bn.json")


if __name__ == "__main__":
    main()
