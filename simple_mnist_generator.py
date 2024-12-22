import numpy as np
import matplotlib.pyplot as plt


class SimpleMNISTGenerator:
    """生成类MNIST数据的简单生成器"""

    def __init__(self, num_samples=1000, image_size=28):
        self.num_samples = num_samples
        self.image_size = image_size

    def generate_digit(self, digit):
        """生成一个数字的图像"""
        image = np.zeros((self.image_size, self.image_size))

        if digit == 0:  # 生成数字0
            # 画一个圆
            center = self.image_size // 2
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if 5 < np.sqrt((i - center) ** 2 + (j - center) ** 2) < 10:
                        image[i, j] = 1

        elif digit == 1:  # 生成数字1
            # 画一条竖线
            start = self.image_size // 4
            end = 3 * self.image_size // 4
            mid = self.image_size // 2
            image[start:end, mid - 1 : mid + 1] = 1

        # 添加随机噪声
        noise = np.random.normal(0, 0.1, image.shape)
        image = np.clip(image + noise, 0, 1)

        return image.flatten()

    def generate_dataset(self, num_classes=2):
        """生成完整的数据集"""
        # 生成图像
        images = []
        labels = []
        samples_per_class = self.num_samples // num_classes

        for digit in range(num_classes):
            for _ in range(samples_per_class):
                image = self.generate_digit(digit)
                images.append(image)
                labels.append(digit)

        # 转换为numpy数组
        X = np.array(images, dtype=np.float32)
        y = np.array(labels)

        # 转换标签为one-hot编码
        y_one_hot = np.zeros((y.size, num_classes))
        y_one_hot[np.arange(y.size), y] = 1

        # 打乱数据
        indices = np.random.permutation(len(X))
        X = X[indices]
        y_one_hot = y_one_hot[indices]

        # 划分数据集
        train_size = int(0.8 * len(X))
        val_size = int(0.1 * len(X))

        train_X = X[:train_size]
        train_y = y_one_hot[:train_size]

        val_X = X[train_size : train_size + val_size]
        val_y = y_one_hot[train_size : train_size + val_size]

        test_X = X[train_size + val_size :]
        test_y = y_one_hot[train_size + val_size :]

        return (train_X, train_y), (val_X, val_y), (test_X, test_y)

    def visualize_samples(self, images, labels, num_samples=5):
        """可视化一些样本"""
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].reshape(self.image_size, self.image_size), cmap="gray")
            plt.title(f"Label: {np.argmax(labels[i])}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 创建数据生成器
    generator = SimpleMNISTGenerator(num_samples=1000)

    # 生成数据集
    (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
    ) = generator.generate_dataset()

    # 打印数据集信息
    print("Dataset shapes:")
    print(f"Training: {train_images.shape}, {train_labels.shape}")
    print(f"Validation: {val_images.shape}, {val_labels.shape}")
    print(f"Test: {test_images.shape}, {test_labels.shape}")

    # 可视化一些样本
    generator.visualize_samples(train_images, train_labels)
