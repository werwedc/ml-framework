namespace MobileRuntime.Models;

public enum OperatorType : ushort
{
    Conv2D,
    DepthwiseConv2D,
    FullyConnected,
    MaxPool2D,
    AvgPool2D,
    BatchNorm,
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
    Add,
    Subtract,
    Multiply,
    Divide,
    Concat,
    Reshape,
    Flatten,
    Transpose,
    MatMul,
    LeakyRelu
}
