import numpy as np


class Activations:
    """激活函数集合"""

    @staticmethod
    def relu(x):
        """
        ReLU激活函数: f(x) = max(0, x)
        比sigmoid更适合深度网络，能缓解梯度消失问题
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """ReLU的导数：x > 0时为1，否则为0"""
        return np.where(x > 0, 1, 0)

    @staticmethod
    def softmax(x):
        """
        Softmax激活函数，用于多分类问题
        将输入转换为概率分布（所有输出的和为1）
        使用数值稳定性的技巧：减去最大值防止指数溢出
        """
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class Loss:
    """损失函数集合"""

    @staticmethod
    def cross_entropy(y_true, y_pred, epsilon=1e-15):
        """
        交叉熵损失函数，适用于多分类问题
        添加epsilon防止log(0)
        """
        # 裁剪预测值，避免数值不稳定
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def cross_entropy_derivative(y_true, y_pred):
        """交叉熵损失对softmax输出的导数"""
        return y_pred - y_true


class Layer:
    """神经网络层的基类"""

    def __init__(self, input_size, output_size):
        """
        初始化层参数
        使用He初始化来设置权重初始值，这对ReLU激活函数特别有效
        """
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.zeros((1, output_size))

        # 为动量优化器初始化速度
        self.velocity_w = np.zeros_like(self.weights)
        self.velocity_b = np.zeros_like(self.biases)

        # 保存前向传播的中间值，用于反向传播
        self.input = None
        self.output = None
        self.pre_activation = None

    def forward(self, x, activation_func):
        """
        前向传播
        x: 输入数据
        activation_func: 激活函数（relu或softmax）
        """
        self.input = x
        self.pre_activation = np.dot(x, self.weights) + self.biases
        self.output = activation_func(self.pre_activation)
        return self.output

    def backward(self, delta, prev_layer, learning_rate, momentum=0.9):
        """
        反向传播
        delta: 从上一层传回的梯度
        prev_layer: 前一层的输出
        learning_rate: 学习率
        momentum: 动量系数
        """
        # 计算梯度
        m = self.input.shape[0]  # 批量大小
        dW = np.dot(self.input.T, delta) / m
        db = np.mean(delta, axis=0, keepdims=True)

        # 使用动量更新参数
        self.velocity_w = momentum * self.velocity_w - learning_rate * dW
        self.velocity_b = momentum * self.velocity_b - learning_rate * db

        self.weights += self.velocity_w
        self.biases += self.velocity_b

        # 计算传递给前一层的梯度
        if prev_layer is not None:
            return np.dot(delta, self.weights.T)
        return None


def create_batches(X, y, batch_size):
    """
    创建小批量数据
    返回一个生成器，每次产生一个批次的数据
    """
    n_samples = len(X)
    indices = np.random.permutation(n_samples)

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i : min(i + batch_size, n_samples)]
        yield X[batch_indices], y[batch_indices]


if __name__ == "__main__":
    # 测试代码
    # 创建一些随机数据
    X = np.random.randn(100, 784)  # 100个样本，每个样本784维
    y = np.eye(10)[np.random.randint(0, 10, 100)]  # 随机标签

    # 测试层的前向传播
    layer = Layer(784, 128)
    output = layer.forward(X, Activations.relu)
    print("Layer output shape:", output.shape)

    # 测试损失函数
    dummy_pred = Activations.softmax(np.random.randn(100, 10))
    loss = Loss.cross_entropy(y, dummy_pred)
    print("Loss value:", loss)
