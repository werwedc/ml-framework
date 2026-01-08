namespace MachineLearning.Checkpointing;

/// <summary>
/// Snapshot of model state for incremental checkpointing
/// </summary>
public class IncrementalSnapshot
{
    /// <summary>
    /// Timestamp when snapshot was created
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Dictionary of tensor metadata
    /// </summary>
    public Dictionary<string, TensorSnapshot> ModelTensors { get; set; } = new();

    /// <summary>
    /// Dictionary of optimizer tensor metadata
    /// </summary>
    public Dictionary<string, TensorSnapshot> OptimizerTensors { get; set; } = new();
}
