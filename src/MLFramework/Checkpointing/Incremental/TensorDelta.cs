namespace MachineLearning.Checkpointing;

/// <summary>
/// Delta information for changed tensor
/// </summary>
public class TensorDelta
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
    /// Actual tensor data (float array serialized)
    /// </summary>
    public float[] Data { get; set; } = Array.Empty<float>();
}
