namespace MLFramework.MobileRuntime.Backends.Cpu
{
    /// <summary>
    /// Types of operators supported by the CPU backend.
    /// </summary>
    public enum OperatorType
    {
        /// <summary>
        /// 2D Convolution
        /// </summary>
        Conv2D,

        /// <summary>
        /// 2D Max Pooling
        /// </summary>
        MaxPool2D,

        /// <summary>
        /// 2D Average Pooling
        /// </summary>
        AvgPool2D,

        /// <summary>
        /// Fully Connected / Dense layer
        /// </summary>
        FullyConnected,

        /// <summary>
        /// ReLU activation
        /// </summary>
        Relu,

        /// <summary>
        /// Sigmoid activation
        /// </summary>
        Sigmoid,

        /// <summary>
        /// Tanh activation
        /// </summary>
        Tanh,

        /// <summary>
        /// Batch Normalization
        /// </summary>
        BatchNorm,

        /// <summary>
        /// Element-wise addition
        /// </summary>
        Add,

        /// <summary>
        /// Element-wise subtraction
        /// </summary>
        Subtract,

        /// <summary>
        /// Element-wise multiplication
        /// </summary>
        Multiply,

        /// <summary>
        /// Element-wise division
        /// </summary>
        Divide,

        /// <summary>
        /// Matrix multiplication
        /// </summary>
        MatMul,

        /// <summary>
        /// Tensor concatenation
        /// </summary>
        Concat,

        /// <summary>
        /// Tensor reshape
        /// </summary>
        Reshape,

        /// <summary>
        /// Transpose operation
        /// </summary>
        Transpose,

        /// <summary>
        /// Softmax operation
        /// </summary>
        Softmax,

        /// <summary>
        /// Element-wise absolute value
        /// </summary>
        Abs,

        /// <summary>
        /// Element-wise square
        /// </summary>
        Square,

        /// <summary>
        /// Element-wise square root
        /// </summary>
        Sqrt,

        /// <summary>
        /// Element-wise exponentiation
        /// </summary>
        Exp,

        /// <summary>
        /// Element-wise logarithm
        /// </summary>
        Log
    }
}
