import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "weight_distributions": [],
            "activation_distributions": [],
        }

        # 使用He初始化
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def compute_accuracy(self, X, y):
        predictions = self.forward(X)
        return np.mean((predictions > 0.5) == y)

    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights)):
            net = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.sigmoid(net))
        return self.activations[-1]

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        delta = self.activations[-1] - y

        if not hasattr(self, "velocity_w"):
            self.velocity_w = [np.zeros_like(w) for w in self.weights]
            self.velocity_b = [np.zeros_like(b) for b in self.biases]

        momentum = 0.9
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(
                    self.activations[i]
                )

            self.velocity_w[i] = momentum * self.velocity_w[i] - learning_rate * dW
            self.velocity_b[i] = momentum * self.velocity_b[i] - learning_rate * db

            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def train(self, X, y, epochs, learning_rate, verbose=True):
        self.training_history["loss"] = []
        self.training_history["accuracy"] = []
        self.training_snapshots = []  # 存储训练过程中的网络状态

        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)

            # 计算损失和准确率
            loss = np.mean(np.square(output - y))
            accuracy = self.compute_accuracy(X, y)

            # 存储训练历史
            self.training_history["loss"].append(loss)
            self.training_history["accuracy"].append(accuracy)

            # 存储权重和激活值的分布
            if epoch % 100 == 0:
                weight_dist = [w.flatten() for w in self.weights]
                activation_dist = [a.flatten() for a in self.activations]
                self.training_history["weight_distributions"].append(weight_dist)
                self.training_history["activation_distributions"].append(
                    activation_dist
                )

                # 存储网络快照用于动画
                self.training_snapshots.append(
                    {
                        "weights": [w.copy() for w in self.weights],
                        "biases": [b.copy() for b in self.biases],
                    }
                )

            # 反向传播
            self.backward(X, y, learning_rate)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")


class NetworkVisualizer:
    def __init__(self, network, X, y):
        self.network = network
        self.X = X
        self.y = y

    def plot_training_history(self):
        """绘制训练过程中的损失值和准确率变化"""
        plt.figure(figsize=(12, 5))

        # 损失值曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.network.training_history["loss"])
        plt.title("Training Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.network.training_history["accuracy"])
        plt.title("Training Accuracy Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_weight_distributions(self):
        """绘制权重分布的变化"""
        plt.figure(figsize=(15, 5))
        snapshots = len(self.network.training_history["weight_distributions"])

        for i, (idx, name) in enumerate([(0, "Input-Hidden"), (1, "Hidden-Output")]):
            plt.subplot(1, 2, i + 1)
            for epoch_idx in [0, snapshots // 2, snapshots - 1]:
                weights = self.network.training_history["weight_distributions"][
                    epoch_idx
                ][idx]
                sns.kdeplot(weights, label=f"Epoch {epoch_idx*100}")
            plt.title(f"{name} Layer Weights Distribution")
            plt.xlabel("Weight Value")
            plt.ylabel("Density")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def create_decision_boundary_animation(self):
        """创建决策边界的动画"""
        fig, ax = plt.subplots(figsize=(8, 8))

        x_min, x_max = self.X[:, 0].min() - 0.1, self.X[:, 0].max() + 0.1
        y_min, y_max = self.X[:, 1].min() - 0.1, self.X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )

        def animate(frame):
            ax.clear()

            # 恢复网络状态
            old_weights = self.network.weights
            old_biases = self.network.biases
            self.network.weights = self.network.training_snapshots[frame]["weights"]
            self.network.biases = self.network.training_snapshots[frame]["biases"]

            # 计算决策边界
            Z = self.network.forward(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # 绘制决策边界和数据点
            ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdBu")
            ax.scatter(
                self.X[self.y.ravel() == 0][:, 0],
                self.X[self.y.ravel() == 0][:, 1],
                c="blue",
                label="Class 0",
                alpha=0.5,
            )
            ax.scatter(
                self.X[self.y.ravel() == 1][:, 0],
                self.X[self.y.ravel() == 1][:, 1],
                c="red",
                label="Class 1",
                alpha=0.5,
            )

            ax.set_title(f"Decision Boundary (Epoch {frame*100})")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.legend()

            # 恢复原始网络状态
            self.network.weights = old_weights
            self.network.biases = old_biases

        anim = FuncAnimation(
            fig,
            animate,
            frames=len(self.network.training_snapshots),
            interval=200,
            repeat=False,
        )
        plt.show()
        return anim


def generate_data(n_samples):
    np.random.seed(42)
    X = np.random.rand(n_samples, 2)
    y = np.zeros((n_samples, 1))

    for i in range(n_samples):
        if (X[i, 0] > 0.5 and X[i, 1] < 0.5) or (X[i, 0] < 0.5 and X[i, 1] > 0.5):
            y[i] = 1

    return X, y


if __name__ == "__main__":
    # 生成数据
    X, y = generate_data(1000)

    # 创建并训练神经网络
    nn = NeuralNetwork([2, 8, 1])
    nn.train(X, y, epochs=2000, learning_rate=0.05)

    # 创建可视化器
    visualizer = NetworkVisualizer(nn, X, y)

    # 显示训练历史
    visualizer.plot_training_history()

    # 显示权重分布
    visualizer.plot_weight_distributions()

    # 创建决策边界动画
    anim = visualizer.create_decision_boundary_animation()

    # 测试网络
    test_X, test_y = generate_data(100)
    test_accuracy = nn.compute_accuracy(test_X, test_y)
    print(f"\nFinal Test Accuracy: {test_accuracy:.2%}")
