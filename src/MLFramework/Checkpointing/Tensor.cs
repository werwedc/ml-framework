namespace MLFramework.Checkpointing;

/// <summary>
/// Tensor class representing multi-dimensional data
/// </summary>
public class Tensor : IDisposable
{
    /// <summary>
    /// Gets the size of the tensor in bytes
    /// </summary>
    public long SizeInBytes { get; }

    /// <summary>
    /// Gets the total number of elements in the tensor
    /// </summary>
    public long ElementCount { get; }

    /// <summary>
    /// Gets the size of each element's data type in bytes
    /// </summary>
    public int DataTypeSize { get; }

    private bool _disposed;

    /// <summary>
    /// Creates a new tensor with specified properties
    /// </summary>
    public Tensor(long elementCount, int dataTypeSize = 4)
    {
        ElementCount = elementCount;
        DataTypeSize = dataTypeSize;
        SizeInBytes = elementCount * dataTypeSize;
        _disposed = false;
    }

    /// <summary>
    /// Disposes the tensor and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            // Release resources here in a real implementation
            _disposed = true;
        }
    }
}
