import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import numpy as np


class VisualizationUtils:
    """用于神经网络分析和可视化的工具类"""

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None
    ) -> None:
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称列表
        """
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(y_true)))]

        conf_matrix = np.zeros((len(class_names), len(class_names)))
        for t, p in zip(y_true, y_pred):
            conf_matrix[t, p] += 1

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    @staticmethod
    def plot_learning_curves(history: Dict[str, List[float]]) -> None:
        """
        绘制学习曲线

        Args:
            history: 训练历史数据
        """
        plt.figure(figsize=(15, 5))

        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(history["loss"], label="Training Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # 准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(history["accuracy"], label="Training Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        # 学习率曲线
        plt.subplot(1, 3, 3)
        plt.plot(history["learning_rate"], label="Learning Rate")
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_prediction_confidence(
        predictions: np.ndarray, confidences: np.ndarray, true_labels: np.ndarray
    ) -> None:
        """
        绘制预测置信度分布

        Args:
            predictions: 预测的类别
            confidences: 预测的置信度
            true_labels: 真实标签
        """
        correct_pred = predictions == true_labels

        plt.figure(figsize=(10, 6))
        plt.hist(
            [confidences[correct_pred], confidences[~correct_pred]],
            label=["Correct Predictions", "Incorrect Predictions"],
            bins=20,
            alpha=0.6,
        )
        plt.title("Prediction Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def visualize_layer_weights(layer_weights: np.ndarray, layer_name: str) -> None:
        """
        可视化层的权重分布

        Args:
            layer_weights: 层的权重矩阵
            layer_name: 层的名称
        """
        plt.figure(figsize=(10, 6))
        plt.hist(layer_weights.flatten(), bins=50)
        plt.title(f"Weight Distribution - {layer_name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Count")
        plt.grid(True)
        plt.show()

        # 计算一些统计信息
        print(f"\nLayer {layer_name} Statistics:")
        print(f"Mean: {np.mean(layer_weights):.6f}")
        print(f"Std: {np.std(layer_weights):.6f}")
        print(f"Min: {np.min(layer_weights):.6f}")
        print(f"Max: {np.max(layer_weights):.6f}")
