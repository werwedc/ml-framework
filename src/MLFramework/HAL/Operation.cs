namespace MLFramework.HAL;

/// <summary>
/// Enumeration of supported tensor operations
/// </summary>
public enum Operation
{
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,

    // Linear Algebra
    MatMul,
    Transpose,

    // Reductions
    Sum,
    Mean,
    Max,
    Min,

    // Memory
    Copy,
    Fill,

    // Activation Functions
    ReLU,
    Sigmoid,
    Tanh,

    // Convolution
    Conv2D,
    MaxPool2D,

    // Misc
    Cast,
    Reshape
}
