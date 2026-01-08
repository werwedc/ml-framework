namespace MLFramework.Checkpointing;

/// <summary>
/// Tensor class representing multi-dimensional data
/// </summary>
public class Tensor : IDisposable
{
    /// <summary>
    /// Gets size of tensor in bytes
    /// </summary>
    public long SizeInBytes { get; }

    /// <summary>
    /// Gets total number of elements in tensor
    /// </summary>
    public long ElementCount { get; }

    /// <summary>
    /// Gets size of each element's data type in bytes
    /// </summary>
    public int DataTypeSize { get; }

    /// <summary>
    /// Gets the tensor shape
    /// </summary>
    public int[] Shape { get; }

    private bool _disposed;

    /// <summary>
    /// Creates a new tensor with specified properties
    /// </summary>
    public Tensor(long elementCount, int dataTypeSize = 4)
    {
        ElementCount = elementCount;
        DataTypeSize = dataTypeSize;
        SizeInBytes = elementCount * dataTypeSize;
        Shape = new int[] { (int)elementCount };
        _disposed = false;
    }

    /// <summary>
    /// Creates a new tensor with data and shape
    /// </summary>
    public Tensor(float[] data, int[] shape, int dataTypeSize = 4)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));
        if (shape == null)
            throw new ArgumentNullException(nameof(shape));

        Shape = shape;
        ElementCount = data.Length;
        DataTypeSize = dataTypeSize;
        SizeInBytes = ElementCount * dataTypeSize;
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
