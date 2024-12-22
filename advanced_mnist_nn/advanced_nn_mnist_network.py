import numpy as np
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, input_size, output_size):
        # 使用He初始化
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.zeros((1, output_size))
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.biases)

    def forward(self, x, activation_func):
        self.input = x
        # 添加数值稳定性检查
        if np.any(np.isnan(x)):
            raise ValueError("Layer input contains NaN values")

        pre_activation = np.clip(np.dot(x, self.weights) + self.biases, -100, 100)
        self.pre_activation = pre_activation
        self.output = activation_func(pre_activation)

        return self.output


class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """初始化神经网络

        Args:
            layer_sizes: 一个列表，指定每层的节点数，例如[784, 128, 64, 10]
        """
        self.layers = []
        self.layer_sizes = layer_sizes

        # 创建网络的各个层
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

        # 初始化训练历史记录
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

    def forward(self, X):
        """前向传播

        Args:
            X: 输入数据，shape为(batch_size, input_features)

        Returns:
            网络的输出预测
        """
        current_input = X

        # 对除最后一层外的所有层使用ReLU
        for layer in self.layers[:-1]:
            current_input = layer.forward(current_input, Activations.relu)

        # 最后一层使用Softmax
        output = self.layers[-1].forward(current_input, Activations.softmax)
        return output

    def backward(self, X, y, learning_rate):
        """反向传播

        Args:
            X: 输入数据
            y: 真实标签
            learning_rate: 学习率
        """
        m = X.shape[0]

        # 计算输出层的梯度
        delta = self.layers[-1].output - y  # 交叉熵损失对softmax的导数

        # 反向传播通过所有层
        for i in range(len(self.layers) - 1, -1, -1):
            # 确定前一层的激活值
            prev_activation = self.layers[i - 1].output if i > 0 else X

            # 如果不是最后一层，需要计算ReLU的导数
            if i < len(self.layers) - 1:
                delta = delta * Activations.relu_derivative(
                    self.layers[i].pre_activation
                )

            # 计算梯度
            dW = np.dot(prev_activation.T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)

            # 使用动量更新参数
            momentum = 0.9
            self.layers[i].velocity_w = (
                momentum * self.layers[i].velocity_w - learning_rate * dW
            )
            self.layers[i].velocity_b = (
                momentum * self.layers[i].velocity_b - learning_rate * db
            )

            self.layers[i].weights += self.layers[i].velocity_w
            self.layers[i].biases += self.layers[i].velocity_b

            # 为下一层计算delta
            if i > 0:
                delta = np.dot(delta, self.layers[i].weights.T)

    def compute_loss(self, y_true, y_pred, epsilon=1e-15):
        """计算交叉熵损失

        Args:
            y_true: 真实标签
            y_pred: 预测值
            epsilon: 数值稳定性的小量

        Returns:
            损失值
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def compute_accuracy(self, X, y):
        """计算准确率

        Args:
            X: 输入数据
            y: 真实标签

        Returns:
            准确率
        """
        predictions = self.forward(X)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        return np.mean(predicted_classes == true_classes)

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=128,
        learning_rate=0.001,
        verbose=True,
    ):
        """训练网络

        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批量大小
            learning_rate: 学习率
            verbose: 是否打印训练过程
        """
        n_batches = len(X_train) // batch_size
        best_val_accuracy = 0
        patience = 10
        no_improve = 0

        for epoch in range(epochs):
            epoch_loss = 0
            current_lr = learning_rate * (0.95**epoch)  # 学习率衰减

            # 随机打乱训练数据
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size

                # 获取小批量数据
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # 前向传播
                predictions = self.forward(X_batch)
                batch_loss = self.compute_loss(y_batch, predictions)
                epoch_loss += batch_loss

                # 反向传播
                self.backward(X_batch, y_batch, current_lr)

            # 计算训练指标
            train_accuracy = self.compute_accuracy(X_train, y_train)

            # 计算验证集性能
            val_predictions = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_predictions)
            val_accuracy = self.compute_accuracy(X_val, y_val)

            # 保存训练历史
            self.training_history["loss"].append(epoch_loss / n_batches)
            self.training_history["accuracy"].append(train_accuracy)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_accuracy"].append(val_accuracy)
            self.training_history["learning_rate"].append(current_lr)

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

    def predict(self, X):
        """预测新数据的类别

        Args:
            X: 输入数据

        Returns:
            预测的类别
        """
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)

    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(15, 5))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history["loss"], label="Training Loss")
        plt.plot(self.training_history["val_loss"], label="Validation Loss")
        plt.title("Model Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history["accuracy"], label="Training Accuracy")
        plt.plot(self.training_history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
