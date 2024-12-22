import numpy as np
import struct
import gzip
import os
import urllib.request


class MNISTLoader:
    """MNIST数据集加载器"""

    def __init__(self, data_folder="mnist_data"):
        """
        初始化加载器
        data_folder: 存储MNIST数据的文件夹路径
        """
        self.data_folder = data_folder
        self.train_images_path = os.path.join(data_folder, "train-images-idx3-ubyte.gz")
        self.train_labels_path = os.path.join(data_folder, "train-labels-idx1-ubyte.gz")
        self.test_images_path = os.path.join(data_folder, "t10k-images-idx3-ubyte.gz")
        self.test_labels_path = os.path.join(data_folder, "t10k-labels-idx1-ubyte.gz")

        # MNIST数据集的URL
        self.urls = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        ]

    def download_mnist(self):
        """下载MNIST数据集"""
        os.makedirs(self.data_folder, exist_ok=True)

        # 下载所有文件
        for url in self.urls:
            filename = url.split("/")[-1]
            filepath = os.path.join(self.data_folder, filename)

            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)

    def read_idx_file(self, filepath, is_image=True):
        """
        读取IDX格式文件
        filepath: 文件路径
        is_image: 是否为图像文件
        """
        with gzip.open(filepath, "rb") as f:
            # 读取文件头
            magic = struct.unpack(">I", f.read(4))[0]
            expected_magic = 2051 if is_image else 2049
            assert magic == expected_magic, f"Invalid magic number {magic}"

            # 读取数据维度
            dims = struct.unpack(">" + "I" * (magic % 256), f.read(4 * (magic % 256)))

            # 读取数据
            data = np.frombuffer(f.read(), dtype=np.uint8)

            if is_image:
                # 重塑图像数据
                data = data.reshape(dims)

        return data

    def load_data(self):
        """加载MNIST数据集"""
        # 确保数据已下载
        self.download_mnist()

        # 加载训练数据
        print("Loading training data...")
        train_images = self.read_idx_file(self.train_images_path, is_image=True)
        train_labels = self.read_idx_file(self.train_labels_path, is_image=False)

        # 加载测试数据
        print("Loading test data...")
        test_images = self.read_idx_file(self.test_images_path, is_image=True)
        test_labels = self.read_idx_file(self.test_labels_path, is_image=False)

        # 数据预处理
        # 1. 将图像数据展平并归一化到[0,1]范围
        train_images = train_images.reshape(-1, 784).astype(np.float32) / 255.0
        test_images = test_images.reshape(-1, 784).astype(np.float32) / 255.0

        # 2. 将标签转换为one-hot编码
        def to_one_hot(labels, num_classes=10):
            one_hot = np.zeros((labels.size, num_classes))
            one_hot[np.arange(labels.size), labels] = 1
            return one_hot

        train_labels = to_one_hot(train_labels)
        test_labels = to_one_hot(test_labels)

        # 3. 划分验证集
        val_size = 5000
        val_images = train_images[-val_size:]
        val_labels = train_labels[-val_size:]
        train_images = train_images[:-val_size]
        train_labels = train_labels[:-val_size]

        print("Data loading completed!")
        print(f"Training set: {train_images.shape[0]} samples")
        print(f"Validation set: {val_images.shape[0]} samples")
        print(f"Test set: {test_images.shape[0]} samples")

        return (
            (train_images, train_labels),
            (val_images, val_labels),
            (test_images, test_labels),
        )


if __name__ == "__main__":
    # 测试数据加载
    loader = MNISTLoader()
    (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
    ) = loader.load_data()

    # 显示数据集的基本信息
    print("\nData shapes:")
    print(f"Training images: {train_images.shape}")
    print(f"Training labels: {train_labels.shape}")
    print(f"Validation images: {val_images.shape}")
    print(f"Validation labels: {val_labels.shape}")
    print(f"Test images: {test_images.shape}")
    print(f"Test labels: {test_labels.shape}")

    # 可视化一些示例图像
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(train_images[i].reshape(28, 28), cmap="gray")
        plt.title(f"Label: {np.argmax(train_labels[i])}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
