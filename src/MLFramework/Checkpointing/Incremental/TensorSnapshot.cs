namespace MachineLearning.Checkpointing;

/// <summary>
/// Snapshot of tensor metadata for incremental checkpointing
/// </summary>
public class TensorSnapshot
{
    /// <summary>
    /// Name of the tensor
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Shape of the tensor
    /// </summary>
    public long[] Shape { get; set; } = Array.Empty<long>();

    /// <summary>
    /// Data type of the tensor
    /// </summary>
    public string DataType { get; set; } = string.Empty;

    /// <summary>
    /// Checksum of tensor data
    /// </summary>
    public string Checksum { get; set; } = string.Empty;
}
