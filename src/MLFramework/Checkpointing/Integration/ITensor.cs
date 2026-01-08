namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface representing a tensor with shape, data type, and size information
/// </summary>
public interface ITensor
{
    /// <summary>
    /// Gets the shape of the tensor
    /// </summary>
    long[] Shape { get; }

    /// <summary>
    /// Gets the data type of the tensor
    /// </summary>
    TensorDataType DataType { get; }

    /// <summary>
    /// Gets the size of the tensor in bytes
    /// </summary>
    long GetSizeInBytes();
}

/// <summary>
/// Tensor data types
/// </summary>
public enum TensorDataType
{
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
    BFloat16
}
