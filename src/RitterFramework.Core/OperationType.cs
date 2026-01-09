namespace RitterFramework.Core;

/// <summary>
/// Enumeration of supported ML operation types for metadata and validation.
/// </summary>
public enum OperationType
{
    /// <summary>
    /// Matrix multiplication operation.
    /// </summary>
    MatrixMultiply,

    /// <summary>
    /// Linear (fully connected) layer operation.
    /// </summary>
    Linear,

    /// <summary>
    /// 2D convolution operation.
    /// </summary>
    Conv2D,

    /// <summary>
    /// 1D convolution operation.
    /// </summary>
    Conv1D,

    /// <summary>
    /// Max pooling 2D operation.
    /// </summary>
    MaxPool2D,

    /// <summary>
    /// Average pooling 2D operation.
    /// </summary>
    AveragePool2D,

    /// <summary>
    /// Concatenation operation along specified axis.
    /// </summary>
    Concat,

    /// <summary>
    /// Stack operation to combine tensors along new dimension.
    /// </summary>
    Stack,

    /// <summary>
    /// Sum reduction operation.
    /// </summary>
    ReduceSum,

    /// <summary>
    /// Mean reduction operation.
    /// </summary>
    ReduceMean,

    /// <summary>
    /// Transpose operation.
    /// </summary>
    Transpose,

    /// <summary>
    /// Reshape operation.
    /// </summary>
    Reshape,

    /// <summary>
    /// Flatten operation.
    /// </summary>
    Flatten,

    /// <summary>
    /// Broadcasting operation.
    /// </summary>
    Broadcast
}
