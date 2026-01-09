namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Types of operators supported by the Metal backend
    /// </summary>
    public enum OperatorType
    {
        /// <summary>
        /// 2D Convolution
        /// </summary>
        Conv2D,

        /// <summary>
        /// ReLU activation
        /// </summary>
        Relu,

        /// <summary>
        /// 2D Max pooling
        /// </summary>
        MaxPool2D,

        /// <summary>
        /// 2D Average pooling
        /// </summary>
        AvgPool2D,

        /// <summary>
        /// Fully connected layer
        /// </summary>
        FullyConnected,

        /// <summary>
        /// Batch normalization
        /// </summary>
        BatchNorm,

        /// <summary>
        /// Softmax
        /// </summary>
        Softmax,

        /// <summary>
        /// Addition (element-wise)
        /// </summary>
        Add,

        /// <summary>
        /// Subtraction (element-wise)
        /// </summary>
        Sub,

        /// <summary>
        /// Multiplication (element-wise)
        /// </summary>
        Mul,

        /// <summary>
        /// Division (element-wise)
        /// </summary>
        Div,

        /// <summary>
        /// Sigmoid activation
        /// </summary>
        Sigmoid,

        /// <summary>
        /// Tanh activation
        /// </summary>
        Tanh,

        /// <summary>
        /// Leaky ReLU
        /// </summary>
        LeakyRelu,

        /// <summary>
        /// Transpose
        /// </summary>
        Transpose,

        /// <summary>
        /// Reshape
        /// </summary>
        Reshape,

        /// <summary>
        /// Flatten
        /// </summary>
        Flatten,

        /// <summary>
        /// Concatenation
        /// </summary>
        Concat
    }
}
