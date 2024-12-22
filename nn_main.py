import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, layers):
        """
        初始化神经网络
        layers: 一个列表,包含每层神经元的数量,[input_size, hidden_size, output_size]
        """
        self.layers = layers
        self.weights = []
        self.biases = []

        # 使用He初始化来初始化权重
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        """sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -700, 700)))  # 添加clip防止溢出

    def sigmoid_derivative(self, x):
        """sigmoid函数的导数"""
        return x * (1 - x)

    def forward(self, X):
        """前向传播"""
        self.activations = [X]

        for i in range(len(self.weights)):
            net = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.sigmoid(net))

        return self.activations[-1]

    def backward(self, X, y, learning_rate):
        """反向传播"""
        m = X.shape[0]
        delta = self.activations[-1] - y

        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(
                    self.activations[i]
                )

            # 添加动量来更新权重和偏置
            if not hasattr(self, "velocity_w"):
                self.velocity_w = [np.zeros_like(w) for w in self.weights]
                self.velocity_b = [np.zeros_like(b) for b in self.biases]

            momentum = 0.9
            self.velocity_w[i] = momentum * self.velocity_w[i] - learning_rate * dW
            self.velocity_b[i] = momentum * self.velocity_b[i] - learning_rate * db

            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def train(self, X, y, epochs, learning_rate, verbose=True):
        """训练神经网络"""
        self.loss_history = []
        best_loss = float("inf")
        patience = 50  # 早停的耐心值
        no_improve = 0

        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)

            # 计算损失
            loss = np.mean(np.square(output - y))
            self.loss_history.append(loss)

            # 早停检查
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

            # 反向传播
            self.backward(X, y, learning_rate)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


def generate_data(n_samples):
    """生成XOR问题的数据"""
    np.random.seed(42)  # 设置随机种子以确保可重复性
    X = np.random.rand(n_samples, 2)
    y = np.zeros((n_samples, 1))

    for i in range(n_samples):
        if (X[i, 0] > 0.5 and X[i, 1] < 0.5) or (X[i, 0] < 0.5 and X[i, 1] > 0.5):
            y[i] = 1

    return X, y


def visualize_decision_boundary(nn, X, y):
    """可视化决策边界"""
    plt.figure(figsize=(12, 5))

    # 创建网格点
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # 预测网格点的类别
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.forward(grid_points)
    Z = Z.reshape(xx.shape)

    # 画出决策边界
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="RdBu")
    plt.colorbar(label="Prediction Probability")

    # 在决策边界上叠加原始数据点
    plt.scatter(
        X[y.ravel() == 0][:, 0],
        X[y.ravel() == 0][:, 1],
        c="blue",
        label="Class 0",
        alpha=0.5,
    )
    plt.scatter(
        X[y.ravel() == 1][:, 0],
        X[y.ravel() == 1][:, 1],
        c="red",
        label="Class 1",
        alpha=0.5,
    )
    plt.title("Neural Network Decision Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 生成数据
    X, y = generate_data(1000)

    # 创建神经网络 [2个输入节点, 8个隐藏节点, 1个输出节点]
    nn = NeuralNetwork([2, 8, 1])

    # 训练网络
    nn.train(X, y, epochs=2000, learning_rate=0.05)

    # 测试网络
    test_X, test_y = generate_data(100)
    predictions = nn.forward(test_X)
    accuracy = np.mean((predictions > 0.5) == test_y)
    print(f"\nTest Accuracy: {accuracy:.2%}")

    # 可视化决策边界
    visualize_decision_boundary(nn, X, y)
