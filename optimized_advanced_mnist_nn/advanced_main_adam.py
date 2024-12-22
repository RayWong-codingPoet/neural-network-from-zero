import matplotlib.pyplot as plt
import numpy as np

from advanced_mnist_nn.advanced_nn_mnist_loader import MNISTLoader
from optimized_advanced_mnist_nn.advanced_nn_mnist_network_with_adam import (
    NeuralNetwork,
)
from optimized_advanced_mnist_nn.visualized_utils import VisualizationUtils


def main():
    # 1. 加载数据
    print("正在加载MNIST数据集...")
    loader = MNISTLoader()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_data()

    # 2. 创建并训练网络
    print("\n创建并训练神经网络...")
    network = NeuralNetwork([784, 128, 64, 10])
    network.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=128)

    # 3. 保存训练好的模型
    print("\n保存模型...")
    network.save_model("models/mnist_model_adam.json")

    # 4. 进行详细的模型评估
    print("\n进行模型评估...")
    evaluation_results = network.evaluate(X_test, y_test)

    print("\n模型评估结果:")
    for metric, value in evaluation_results.items():
        if isinstance(value, list):
            print(f"\n{metric}:")
            for i, v in enumerate(value):
                print(f"  类别 {i}: {v:.4f}")
        else:
            print(f"{metric}: {value:.4f}")

    # 5. 生成预测及置信度
    print("\n生成预测及置信度分析...")
    predictions, confidences = network.predict_with_confidence(X_test)
    true_labels = np.argmax(y_test, axis=1)

    # 6. 创建可视化工具实例
    viz = VisualizationUtils()

    # 7. 可视化混淆矩阵
    print("\n绘制混淆矩阵...")
    viz.plot_confusion_matrix(
        true_labels, predictions, class_names=[str(i) for i in range(10)]
    )

    # 8. 可视化学习曲线
    print("\n绘制学习曲线...")
    viz.plot_learning_curves(network.training_history)

    # 9. 可视化预测置信度
    print("\n绘制预测置信度分布...")
    viz.plot_prediction_confidence(predictions, confidences, true_labels)

    # 10. 分析网络权重
    print("\n分析网络权重分布...")
    for i, layer in enumerate(network.layers):
        viz.visualize_layer_weights(layer.weights, f"Layer {i+1}")

    # 11. 展示一些预测示例
    print("\n展示预测示例...")
    n_samples = 10
    sample_indices = np.random.choice(len(X_test), n_samples)

    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        plt.title(
            f"True: {true_labels[idx]}\nPred: {predictions[idx]}\nConf: {confidences[idx]:.2f}"
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 12. 测试模型加载功能
    print("\n测试模型加载功能...")
    loaded_network = NeuralNetwork.load_model("models/mnist_model_adam.json")
    loaded_accuracy = loaded_network.compute_accuracy(X_test, y_test)
    print(f"加载的模型测试集准确率: {loaded_accuracy:.4f}")

    # 13. 对比原始模型和加载后的模型性能
    print("\n性能对比:")
    original_accuracy = network.compute_accuracy(X_test, y_test)
    print(f"原始模型准确率: {original_accuracy:.4f}")
    print(f"加载后模型准确率: {loaded_accuracy:.4f}")


if __name__ == "__main__":
    main()
