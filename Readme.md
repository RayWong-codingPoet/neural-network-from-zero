# Neural Network from Zero

This project implements a neural network for MNIST digit classification completely from scratch, demonstrating the progressive evolution of neural network architectures and optimization techniques. By building everything from the ground up using only NumPy, we gain deep insights into how neural networks really work and how different optimization strategies affect their performance.

## Project Structure

```
neural-network-from-zero/
├── advanced_mnist_nn/                           # Basic implementation
│   ├── __init__.py
│   ├── advanced_main.py                        # Main training script
│   ├── advanced_nn_mnist_components.py         # Neural network components
│   ├── advanced_nn_mnist_network.py            # Core network implementation
│   └── simple_mnist_generator.py               # Data generation utilities
│
├── optimized_advanced_mnist_nn/                # Adam optimizer implementation
│   ├── __init__.py
│   ├── advanced_main_adam.py                   # Main script with Adam
│   └── advanced_nn_mnist_network_with_adam.py  # Network with Adam optimization
│
├── batch_normalization_optimized_mnist_nn/     # BatchNorm implementation
│   ├── models/                                # Saved model storage
│   ├── __init__.py
│   ├── advanced_main_bn.py                    # Main script with BatchNorm
│   └── advanced_nn_mnist_network_with_bn.py   # Network with BatchNorm
│
├── advanced_nn_mnist_loader.py                # MNIST dataset loader
├── basic_nn_xor.py                           # Simple XOR network example
├── visualized_utils.py                        # Visualization utilities
└── README.md                                 # Project documentation
```

## Implementation Journey

The project evolves through three major implementations, each building on the previous one to demonstrate key concepts in deep learning:

### 1. Basic Implementation (advanced_mnist_nn)
This serves as our foundation, implementing core neural network concepts:
- Feedforward neural network with backpropagation
- ReLU activation for hidden layers and Softmax for output
- Simple gradient descent optimization
- Basic learning rate decay
- Key achievement: 95.14% test accuracy

The basic implementation helps us understand:
- How neural networks transform input data through layers
- The mathematics behind backpropagation
- Why activation functions are necessary
- How gradient descent optimizes network weights

### 2. Adam Optimization (optimized_advanced_mnist_nn)
Building on the basic version, we add the Adam optimizer:
- Adaptive moment estimation
- First and second moment tracking
- Bias correction mechanisms
- Dynamic learning rate adjustment
- Improved performance: 97.87% test accuracy

This implementation demonstrates:
- Why simple gradient descent isn't always enough
- How momentum helps overcome local minima
- Why adaptive learning rates improve training
- The importance of bias correction in optimization

### 3. Batch Normalization (batch_normalization_optimized_mnist_nn)
The final evolution adds batch normalization:
- Layer-wise normalization of activations
- Learnable scale and shift parameters
- Running statistics for inference
- Best performance: 98.10% test accuracy

This version showcases:
- How internal covariate shift affects training
- Why normalizing activations helps
- How batch statistics differ between training and inference
- The regularization effect of BatchNorm

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-network-from-zero.git
cd neural-network-from-zero

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy matplotlib
```

## Running the Examples

Each implementation can be run independently to observe the improvements:

```bash
# Basic version
python -m advanced_mnist_nn.advanced_main

# Adam optimization version
python -m optimized_advanced_mnist_nn.advanced_main_adam

# BatchNorm version
python -m batch_normalization_optimized_mnist_nn.advanced_main_bn
```

## Results and Analysis

Our implementations show progressive improvements across key metrics:

| Metric           | Basic | Adam | BatchNorm |
|-----------------|-------|------|-----------|
| Test Accuracy   | 95.14%| 97.87%| 98.10%   |
| Training Time   | 104s  | 99s  | 137s      |
| Epochs to Stop  | 50    | 43   | 26        |
| Initial Loss    | 1.25  | 0.34 | 0.23      |

Key observations:
- BatchNorm enables faster convergence despite longer per-epoch time
- Adam provides more stable training dynamics
- Each optimization reduces the accuracy gap between training and validation
- Later implementations handle difficult digits (like 5 and 8) better

## Features and Tools

The project includes several useful utilities:
- `advanced_nn_mnist_loader.py`: Handles MNIST dataset downloading and preprocessing
- `visualized_utils.py`: Provides comprehensive visualization tools
- `basic_nn_xor.py`: Simple network implementation for learning basics
- Model saving and loading capabilities
- Extensive performance analysis tools

## Development

The code is structured to be educational and extensible. Each implementation follows these principles:
- Clear separation of concerns
- Comprehensive comments explaining the mathematics
- Consistent API across implementations
- Robust error handling
- Extensive logging and visualization

## Contributing

Contributions are welcome! Some areas for potential improvement:
- Additional optimization techniques (e.g., dropout)
- More visualization tools
- Performance optimizations
- Additional datasets
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds upon foundational work in deep learning:
- The MNIST dataset from Yann LeCun and collaborators
- Adam optimizer paper by Kingma and Ba
- Batch Normalization paper by Ioffe and Szegedy
