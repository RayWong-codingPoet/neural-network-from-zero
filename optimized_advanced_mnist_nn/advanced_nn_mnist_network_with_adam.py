import json
import os
import time
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np


class AdamOptimizer:
    """
    Enhanced Adam (Adaptive Moment Estimation) optimizer implementation
    with improved numerical stability and proper bias correction
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        # Parameter validation
        if not (0.0 < learning_rate < 1.0):
            raise ValueError("Learning rate must be between 0 and 1")
        if not (0.0 < beta1 < 1.0) or not (0.0 < beta2 < 1.0):
            raise ValueError("Beta values must be between 0 and 1")

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0  # Time step

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Update parameters using Adam optimization

        Args:
            params: Current parameters
            grads: Parameter gradients

        Returns:
            Updated parameters
        """
        # Initialize moments if first update
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Clip gradients to prevent exploding gradients
        grads = np.clip(grads, -5, 5)

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * np.square(grads)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1.0 - np.power(self.beta1, self.t))

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1.0 - np.power(self.beta2, self.t))

        # Compute the update
        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Apply update with gradient clipping
        return params - np.clip(update, -1, 1)


class Layer:
    """Neural network layer with improved initialization and optimization"""

    def __init__(self, input_size: int, output_size: int):
        # He initialization with improved numerical stability
        scale = np.sqrt(2.0 / float(input_size))
        self.weights = np.random.normal(0.0, scale, (input_size, output_size))
        self.biases = np.zeros((1, output_size))

        # Separate optimizers for weights and biases
        self.weight_optimizer = AdamOptimizer()
        self.bias_optimizer = AdamOptimizer()

        # Cache for forward pass
        self.input = None
        self.output = None
        self.pre_activation = None

    def forward(self, x: np.ndarray, activation_func) -> np.ndarray:
        """Forward pass with improved numerical stability"""
        self.input = x
        # Clip inputs to prevent numerical instability
        x_clipped = np.clip(x, -100, 100)
        self.pre_activation = np.dot(x_clipped, self.weights) + self.biases
        self.output = activation_func(self.pre_activation)
        return self.output


class Activations:
    """Activation functions with improved numerical stability"""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Numerically stable ReLU"""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU derivative with improved numerical stability"""
        return np.where(x > 0, 1.0, 0.0)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax implementation"""
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class LearningRateScheduler:
    """Enhanced learning rate scheduler with multiple decay options"""

    def __init__(self, initial_lr: float = 0.001, decay_type: str = "cosine"):
        self.initial_lr = initial_lr
        self.decay_type = decay_type

    def __call__(self, epoch: int, total_epochs: int) -> float:
        if self.decay_type == "cosine":
            return self.cosine_decay(epoch, total_epochs)
        elif self.decay_type == "step":
            return self.step_decay(epoch)
        return self.initial_lr

    def cosine_decay(self, epoch: int, total_epochs: int) -> float:
        """Cosine annealing learning rate schedule"""
        return self.initial_lr * (1 + np.cos(np.pi * epoch / total_epochs)) / 2

    def step_decay(self, epoch: int) -> float:
        """Step decay learning rate schedule"""
        drop_rate = 0.5
        epochs_drop = 10.0
        return self.initial_lr * np.power(
            drop_rate, np.floor((1 + epoch) / epochs_drop)
        )


class NeuralNetwork:
    """Enhanced neural network with improved training and monitoring"""

    def __init__(self, layer_sizes: List[int]):
        self.layers = []
        self.layer_sizes = layer_sizes
        self.lr_scheduler = LearningRateScheduler(initial_lr=0.001, decay_type="cosine")

        # Initialize layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

        # Training history with additional metrics
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "gradient_norm": [],
        }

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass with improved error handling"""
        try:
            current_input = X
            for layer in self.layers[:-1]:
                current_input = layer.forward(current_input, Activations.relu)
            return self.layers[-1].forward(current_input, Activations.softmax)
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise

    def backward(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """Backward pass with gradient norm monitoring"""
        m = X.shape[0]
        delta = self.layers[-1].output - y
        total_gradient_norm = 0

        for i in range(len(self.layers) - 1, -1, -1):
            prev_activation = self.layers[i - 1].output if i > 0 else X

            # Compute gradients
            dW = np.dot(prev_activation.T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)

            # Monitor gradient norm
            total_gradient_norm += np.sqrt(
                np.sum(np.square(dW)) + np.sum(np.square(db))
            )

            # Update parameters using Adam
            self.layers[i].weights = self.layers[i].weight_optimizer.update(
                self.layers[i].weights, dW
            )
            self.layers[i].biases = self.layers[i].bias_optimizer.update(
                self.layers[i].biases, db
            )

            if i > 0:
                delta = np.dot(
                    delta, self.layers[i].weights.T
                ) * Activations.relu_derivative(self.layers[i - 1].pre_activation)

        return total_gradient_norm

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        verbose: bool = True,
    ):
        """Enhanced training loop with improved monitoring and error handling"""
        try:
            n_batches = len(X_train) // batch_size
            best_val_accuracy = 0
            patience = 10
            no_improve = 0
            start_time = time.time()

            for epoch in range(epochs):
                epoch_loss = 0
                epoch_gradient_norm = 0
                current_lr = self.lr_scheduler(epoch, epochs)

                # Shuffle training data
                indices = np.random.permutation(len(X_train))
                X_train = X_train[indices]
                y_train = y_train[indices]

                # Mini-batch training
                for batch in range(n_batches):
                    start_idx = batch * batch_size
                    end_idx = start_idx + batch_size

                    X_batch = X_train[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]

                    # Forward pass
                    predictions = self.forward(X_batch)
                    batch_loss = self.compute_loss(y_batch, predictions)
                    epoch_loss += batch_loss

                    # Backward pass
                    gradient_norm = self.backward(X_batch, y_batch, current_lr)
                    epoch_gradient_norm += gradient_norm

                # Calculate metrics
                train_accuracy = self.compute_accuracy(X_train, y_train)
                val_predictions = self.forward(X_val)
                val_loss = self.compute_loss(y_val, val_predictions)
                val_accuracy = self.compute_accuracy(X_val, y_val)

                # Update history
                self.training_history["loss"].append(epoch_loss / n_batches)
                self.training_history["accuracy"].append(train_accuracy)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_accuracy"].append(val_accuracy)
                self.training_history["learning_rate"].append(current_lr)
                self.training_history["gradient_norm"].append(
                    epoch_gradient_norm / n_batches
                )

                if verbose and (epoch % 1 == 0):
                    print(
                        f"Epoch {epoch+1}/{epochs}, "
                        f"Loss: {epoch_loss/n_batches:.4f}, "
                        f"Accuracy: {train_accuracy:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val Accuracy: {val_accuracy:.4f}, "
                        f"LR: {current_lr:.6f}, "
                        f"Grad Norm: {epoch_gradient_norm/n_batches:.6f}"
                    )

                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

            training_time = time.time() - start_time
            print(f"\nTraining completed! Total time: {training_time:.2f} seconds")
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def plot_training_history(self):
        """Enhanced visualization of training metrics"""
        plt.figure(figsize=(15, 10))

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.training_history["loss"], label="Training Loss")
        plt.plot(self.training_history["val_loss"], label="Validation Loss")
        plt.title("Model Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.training_history["accuracy"], label="Training Accuracy")
        plt.plot(self.training_history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(self.training_history["learning_rate"])
        plt.title("Learning Rate Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid(True)

        # Plot gradient norm
        plt.subplot(2, 2, 4)
        plt.plot(self.training_history["gradient_norm"])
        plt.title("Gradient Norm Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def compute_loss(
        self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15
    ) -> float:
        """Compute cross-entropy loss with improved numerical stability"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy"""
        predictions = self.forward(X)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(predicted_classes == true_classes)

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
        for i, layer in enumerate(self.layers):
            layer_data = {
                "weights": layer.weights.tolist(),
                "biases": layer.biases.tolist(),
                "optimizer_state": {
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
            network.layers[i].weights = np.array(layer_data["weights"])
            network.layers[i].biases = np.array(layer_data["biases"])

            # 恢复优化器状态
            if layer_data["optimizer_state"]["weight_optimizer"]["m"] is not None:
                network.layers[i].weight_optimizer.m = np.array(
                    layer_data["optimizer_state"]["weight_optimizer"]["m"]
                )
                network.layers[i].weight_optimizer.v = np.array(
                    layer_data["optimizer_state"]["weight_optimizer"]["v"]
                )
                network.layers[i].weight_optimizer.t = layer_data["optimizer_state"][
                    "weight_optimizer"
                ]["t"]

            if layer_data["optimizer_state"]["bias_optimizer"]["m"] is not None:
                network.layers[i].bias_optimizer.m = np.array(
                    layer_data["optimizer_state"]["bias_optimizer"]["m"]
                )
                network.layers[i].bias_optimizer.v = np.array(
                    layer_data["optimizer_state"]["bias_optimizer"]["v"]
                )
                network.layers[i].bias_optimizer.t = layer_data["optimizer_state"][
                    "bias_optimizer"
                ]["t"]

        return network

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        全面评估模型性能

        Args:
            X: 输入数据
            y: 真实标签

        Returns:
            包含各种评估指标的字典
        """
        predictions = self.forward(X)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)

        # 计算准确率
        accuracy = np.mean(pred_classes == true_classes)

        # 计算每个类别的精确率和召回率
        n_classes = y.shape[1]
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1_scores = np.zeros(n_classes)

        for i in range(n_classes):
            true_positives = np.sum((pred_classes == i) & (true_classes == i))
            false_positives = np.sum((pred_classes == i) & (true_classes != i))
            false_negatives = np.sum((pred_classes != i) & (true_classes == i))

            # 计算精确率
            precision[i] = true_positives / (true_positives + false_positives + 1e-10)

            # 计算召回率
            recall[i] = true_positives / (true_positives + false_negatives + 1e-10)

            # 计算F1分数
            f1_scores[i] = (
                2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-10)
            )

        # 计算损失
        loss = self.compute_loss(y, predictions)

        # 汇总评估结果
        evaluation_results = {
            "accuracy": float(accuracy),
            "loss": float(loss),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_scores_per_class": f1_scores.tolist(),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1_scores)),
        }

        return evaluation_results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别概率

        Args:
            X: 输入数据

        Returns:
            每个类别的概率
        """
        return self.forward(X)

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
