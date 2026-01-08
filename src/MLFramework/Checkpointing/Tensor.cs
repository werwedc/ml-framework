namespace MLFramework.Checkpointing;

/// <summary>
/// Placeholder for Tensor class
/// </summary>
public class Tensor
{
    /// <summary>
    /// Gets the size of the tensor in bytes
    /// </summary>
    public long SizeInBytes { get; set; }

    /// <summary>
    /// Creates a new tensor
    /// </summary>
    public Tensor()
    {
        SizeInBytes = 0;
    }

    /// <summary>
    /// Creates a new tensor with specified size
    /// </summary>
    public Tensor(long sizeInBytes)
    {
        SizeInBytes = sizeInBytes;
    }
}
