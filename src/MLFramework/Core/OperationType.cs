namespace MLFramework.Core
{
    /// <summary>
    /// Enumeration of supported operation types in the ML framework.
    /// Used for shape validation and diagnostic reporting.
    /// </summary>
    public enum OperationType
    {
        MatrixMultiply,
        Conv2D,
        Conv1D,
        MaxPool2D,
        AveragePool2D,
        Concat,
        Stack,
        Reshape,
        Transpose,
        Flatten,
        Broadcast,
        Unknown
    }
}
