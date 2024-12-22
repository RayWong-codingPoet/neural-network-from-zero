import json
import os
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np


class AdamOptimizer:
    """
    实现Adam优化器，用于参数更新
    Adam结合了动量法和RMSprop的优点，能自适应地调整学习率
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1  # 一阶矩估计的指数衰减率
        self.beta2 = beta2  # 二阶矩估计的指数衰减率
        self.epsilon = epsilon
        self.m = None  # 一阶矩估计（动量）
        self.v = None  # 二阶矩估计（RMSprop）
        self.t = 0  # 时间步

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """参数更新函数"""
        # 首次调用时初始化动量
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # 梯度裁剪，防止梯度爆炸
        grads = np.clip(grads, -5, 5)

        # 更新动量估计
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)

        # 偏差修正
        m_hat = self.m / (1 - np.power(self.beta1, self.t))
        v_hat = self.v / (1 - np.power(self.beta2, self.t))

        # 参数更新
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params - update


class BatchNorm:
    """
    批量归一化层
    通过归一化每个mini-batch来加速训练并提供轻微的正则化效果
    """

    def __init__(self, num_features: int, epsilon: float = 1e-8, momentum: float = 0.9):
        # 可学习参数
        self.gamma = np.ones(num_features)  # 缩放参数
        self.beta = np.zeros(num_features)  # 平移参数

        # 优化器
        self.gamma_optimizer = AdamOptimizer()
        self.beta_optimizer = AdamOptimizer()

        # 运行时统计量
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # 超参数
        self.epsilon = epsilon
        self.momentum = momentum

        # 缓存前向传播的中间值
        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        if training:
            # 计算mini-batch统计量
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0) + self.epsilon

            # 归一化
            x_norm = (x - batch_mean) / np.sqrt(batch_var)

            # 更新运行时统计量
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            # 缓存中间值
            self.cache = {
                "x": x,
                "mean": batch_mean,
                "var": batch_var,
                "x_norm": x_norm,
                "gamma": self.gamma,
                "beta": self.beta,
            }
        else:
            # 测试时使用运行时统计量
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # 缩放和平移
        out = self.gamma * x_norm + self.beta
        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """反向传播计算梯度"""
        x = self.cache["x"]
        mean = self.cache["mean"]
        var = self.cache["var"]
        x_norm = self.cache["x_norm"]
        N = x.shape[0]

        # 计算各参数的梯度
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        dx_norm = dout * self.gamma

        # 计算关于均值和方差的梯度
        dvar = np.sum(
            dx_norm * (x - mean) * -0.5 * (var + self.epsilon) ** (-1.5), axis=0
        )
        dmean = np.sum(
            dx_norm * -1 / np.sqrt(var + self.epsilon), axis=0
        ) + dvar * np.mean(-2 * (x - mean), axis=0)

        # 计算输入的梯度
        dx = (
            dx_norm / np.sqrt(var + self.epsilon)
            + dvar * 2 * (x - mean) / N
            + dmean / N
        )

        return dx, dgamma, dbeta


class Layer:
    """神经网络层，集成了批量归一化和Adam优化器"""

    def __init__(self, input_size: int, output_size: int, use_batchnorm: bool = True):
        # He初始化权重
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.zeros((1, output_size))

        # 优化器
        self.weight_optimizer = AdamOptimizer()
        self.bias_optimizer = AdamOptimizer()

        # 批量归一化
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = BatchNorm(output_size)

        # 缓存
        self.input = None
        self.output = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        self.input = x

        # 线性变换
        z = np.dot(x, self.weights) + self.biases

        # 批量归一化
        if self.use_batchnorm:
            z = self.bn.forward(z, training)

        # 保存输出，这很重要
        self.output = z
        return z

    def backward(self, delta: np.ndarray, learning_rate: float = 0.001) -> np.ndarray:
        """反向传播"""
        if self.use_batchnorm:
            delta, dgamma, dbeta = self.bn.backward(delta)

            # 更新批量归一化参数
            self.bn.gamma = self.bn.gamma_optimizer.update(self.bn.gamma, dgamma)
            self.bn.beta = self.bn.beta_optimizer.update(self.bn.beta, dbeta)

        # 计算梯度
        m = self.input.shape[0]
        dW = np.dot(self.input.T, delta) / m
        db = np.mean(delta, axis=0, keepdims=True)

        # 更新参数
        self.weights = self.weight_optimizer.update(self.weights, dW)
        self.biases = self.bias_optimizer.update(self.biases, db)

        # 计算前一层的梯度
        return np.dot(delta, self.weights.T)


class NeuralNetwork:
    """完整的神经网络实现，包含批量归一化和Adam优化"""

    def __init__(self, layer_sizes: List[int], use_batchnorm: bool = True):
        self.layers = []
        self.layer_sizes = layer_sizes

        # 创建网络层
        for i in range(len(layer_sizes) - 1):
            # 最后一层不使用批量归一化
            use_bn = use_batchnorm and i < len(layer_sizes) - 2
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], use_bn)
            self.layers.append(layer)

        # 初始化训练历史
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "gradient_norm": [],
        }

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        current_input = X

        # 添加输入检查
        assert not np.isnan(X).any(), "输入包含NaN值"

        for layer in self.layers[:-1]:
            z = layer.forward(current_input, training)
            current_input = np.maximum(0, z)
            # 添加中间值检查
            assert not np.isnan(current_input).any(), "ReLU输出包含NaN值"

        z = self.layers[-1].forward(current_input, training)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        softmax_output = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # 添加输出检查
        assert not np.isnan(softmax_output).any(), "Softmax输出包含NaN值"

        self.layers[-1].output = softmax_output
        return softmax_output

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """反向传播"""
        # 使用保存的softmax输出计算初始梯度
        delta = self.layers[-1].output - y  # 现在这个值应该可以正确计算
        total_gradient_norm = 0

        # 反向传播通过所有层
        for i in range(len(self.layers) - 1, -1, -1):
            # 获取前一层的激活值
            prev_activation = self.layers[i - 1].output if i > 0 else X

            # 更新参数并获取梯度
            delta = self.layers[i].backward(delta, learning_rate)

            # 如果不是最后一层，应用ReLU导数
            if i > 0:
                delta = delta * (prev_activation > 0)  # ReLU导数

            # 计算梯度范数
            total_gradient_norm += np.sqrt(np.sum(np.square(delta)))

        return total_gradient_norm

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        verbose: bool = True,
    ):
        """训练网络"""
        n_batches = len(X_train) // batch_size
        best_val_accuracy = 0
        patience = 10
        no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_gradient_norm = 0

            # 打乱训练数据
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Mini-batch训练
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]

                # 前向传播
                predictions = self.forward(X_batch, training=True)
                batch_loss = self.compute_loss(y_batch, predictions)
                epoch_loss += batch_loss

                # 反向传播
                gradient_norm = self.backward(X_batch, y_batch, learning_rate)
                epoch_gradient_norm += gradient_norm

            # 计算评估指标
            train_accuracy = self.compute_accuracy(X_train, y_train)
            val_predictions = self.forward(X_val, training=False)
            val_loss = self.compute_loss(y_val, val_predictions)
            val_accuracy = self.compute_accuracy(X_val, y_val)

            # 更新训练历史
            self.training_history["loss"].append(epoch_loss / n_batches)
            self.training_history["accuracy"].append(train_accuracy)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_accuracy"].append(val_accuracy)
            self.training_history["learning_rate"].append(learning_rate)
            self.training_history["gradient_norm"].append(
                epoch_gradient_norm / n_batches
            )

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs}, "
                    f"Loss: {epoch_loss/n_batches:.4f}, "
                    f"Accuracy: {train_accuracy:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )

            # 早停检查
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    def compute_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15
    ) -> float:
        """计算交叉熵损失"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        predictions = self.forward(X, training=False)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(predicted_classes == true_classes)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        全面评估模型性能，计算多个评估指标

        Args:
            X: 输入数据
            y: 真实标签

        Returns:
            包含各种评估指标的字典
        """
        predictions = self.forward(X, training=False)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)

        # 计算基本指标
        accuracy = np.mean(pred_classes == true_classes)
        loss = self.compute_loss(y, predictions)

        # 计算每个类别的性能指标
        n_classes = y.shape[1]
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1_scores = np.zeros(n_classes)

        for i in range(n_classes):
            # 计算真阳性、假阳性和假阴性
            true_positives = np.sum((pred_classes == i) & (true_classes == i))
            false_positives = np.sum((pred_classes == i) & (true_classes != i))
            false_negatives = np.sum((pred_classes != i) & (true_classes == i))

            # 计算精确率和召回率
            precision[i] = true_positives / (true_positives + false_positives + 1e-10)
            recall[i] = true_positives / (true_positives + false_negatives + 1e-10)

            # 计算F1分数
            f1_scores[i] = (
                2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-10)
            )

        # 汇总结果
        return {
            "accuracy": float(accuracy),
            "loss": float(loss),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_scores_per_class": f1_scores.tolist(),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1_scores)),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别概率

        Args:
            X: 输入数据

        Returns:
            每个类别的概率
        """
        return self.forward(X, training=False)

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测类别并返回置信度

        Args:
            X: 输入数据

        Returns:
            预测类别和对应的置信度
        """
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        return predictions, confidences

    def save_model(self, filepath: str):
        """
        保存模型参数和配置到文件

        Args:
            filepath: 保存模型的文件路径
        """
        model_data = {
            "layer_sizes": self.layer_sizes,
            "layers": [],
            "training_history": self.training_history,
        }

        # 保存每一层的参数
        for layer in self.layers:
            layer_data = {
                "weights": layer.weights.tolist(),
                "biases": layer.biases.tolist(),
                "use_batchnorm": layer.use_batchnorm,
                "optimizer_states": {
                    "weight_optimizer": {
                        "m": (
                            layer.weight_optimizer.m.tolist()
                            if layer.weight_optimizer.m is not None
                            else None
                        ),
                        "v": (
                            layer.weight_optimizer.v.tolist()
                            if layer.weight_optimizer.v is not None
                            else None
                        ),
                        "t": layer.weight_optimizer.t,
                    },
                    "bias_optimizer": {
                        "m": (
                            layer.bias_optimizer.m.tolist()
                            if layer.bias_optimizer.m is not None
                            else None
                        ),
                        "v": (
                            layer.bias_optimizer.v.tolist()
                            if layer.bias_optimizer.v is not None
                            else None
                        ),
                        "t": layer.bias_optimizer.t,
                    },
                },
            }

            # 如果使用了批量归一化，保存相关参数
            if layer.use_batchnorm:
                layer_data["batch_norm"] = {
                    "gamma": layer.bn.gamma.tolist(),
                    "beta": layer.bn.beta.tolist(),
                    "running_mean": layer.bn.running_mean.tolist(),
                    "running_var": layer.bn.running_var.tolist(),
                    "optimizer_states": {
                        "gamma_optimizer": {
                            "m": (
                                layer.bn.gamma_optimizer.m.tolist()
                                if layer.bn.gamma_optimizer.m is not None
                                else None
                            ),
                            "v": (
                                layer.bn.gamma_optimizer.v.tolist()
                                if layer.bn.gamma_optimizer.v is not None
                                else None
                            ),
                            "t": layer.bn.gamma_optimizer.t,
                        },
                        "beta_optimizer": {
                            "m": (
                                layer.bn.beta_optimizer.m.tolist()
                                if layer.bn.beta_optimizer.m is not None
                                else None
                            ),
                            "v": (
                                layer.bn.beta_optimizer.v.tolist()
                                if layer.bn.beta_optimizer.v is not None
                                else None
                            ),
                            "t": layer.bn.beta_optimizer.t,
                        },
                    },
                }

            model_data["layers"].append(layer_data)

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存到文件
        with open(filepath, "w") as f:
            json.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "NeuralNetwork":
        """
        从文件加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            加载的神经网络模型
        """
        with open(filepath, "r") as f:
            model_data = json.load(f)

        # 创建新的神经网络实例
        network = cls(model_data["layer_sizes"])

        # 恢复训练历史
        network.training_history = model_data["training_history"]

        # 恢复每一层的参数
        for i, layer_data in enumerate(model_data["layers"]):
            # 恢复基本参数
            network.layers[i].weights = np.array(layer_data["weights"])
            network.layers[i].biases = np.array(layer_data["biases"])

            # 恢复优化器状态
            w_opt_state = layer_data["optimizer_states"]["weight_optimizer"]
            b_opt_state = layer_data["optimizer_states"]["bias_optimizer"]

            if w_opt_state["m"] is not None:
                network.layers[i].weight_optimizer.m = np.array(w_opt_state["m"])
                network.layers[i].weight_optimizer.v = np.array(w_opt_state["v"])
                network.layers[i].weight_optimizer.t = w_opt_state["t"]

            if b_opt_state["m"] is not None:
                network.layers[i].bias_optimizer.m = np.array(b_opt_state["m"])
                network.layers[i].bias_optimizer.v = np.array(b_opt_state["v"])
                network.layers[i].bias_optimizer.t = b_opt_state["t"]

            # 恢复批量归一化参数（如果有）
            if layer_data.get("batch_norm"):
                bn_data = layer_data["batch_norm"]
                network.layers[i].bn.gamma = np.array(bn_data["gamma"])
                network.layers[i].bn.beta = np.array(bn_data["beta"])
                network.layers[i].bn.running_mean = np.array(bn_data["running_mean"])
                network.layers[i].bn.running_var = np.array(bn_data["running_var"])

                # 恢复批量归一化优化器状态
                gamma_opt_state = bn_data["optimizer_states"]["gamma_optimizer"]
                beta_opt_state = bn_data["optimizer_states"]["beta_optimizer"]

                if gamma_opt_state["m"] is not None:
                    network.layers[i].bn.gamma_optimizer.m = np.array(
                        gamma_opt_state["m"]
                    )
                    network.layers[i].bn.gamma_optimizer.v = np.array(
                        gamma_opt_state["v"]
                    )
                    network.layers[i].bn.gamma_optimizer.t = gamma_opt_state["t"]

                if beta_opt_state["m"] is not None:
                    network.layers[i].bn.beta_optimizer.m = np.array(
                        beta_opt_state["m"]
                    )
                    network.layers[i].bn.beta_optimizer.v = np.array(
                        beta_opt_state["v"]
                    )
                    network.layers[i].bn.beta_optimizer.t = beta_opt_state["t"]

        return network

    def plot_training_history(self):
        """绘制训练历史，包括损失、准确率和梯度范数"""
        plt.figure(figsize=(15, 10))

        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.training_history["loss"], label="Training Loss")
        plt.plot(self.training_history["val_loss"], label="Validation Loss")
        plt.title("Model Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # 绘制准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.training_history["accuracy"], label="Training Accuracy")
        plt.plot(self.training_history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        # 绘制学习率曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.training_history["learning_rate"], label="Learning Rate")
        plt.title("Learning Rate Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True)

        # 绘制梯度范数
        plt.subplot(2, 2, 4)
        plt.plot(self.training_history["gradient_norm"], label="Gradient Norm")
        plt.title("Gradient Norm Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
