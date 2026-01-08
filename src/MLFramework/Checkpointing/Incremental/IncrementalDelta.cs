namespace MachineLearning.Checkpointing;

/// <summary>
/// Delta representing changes between snapshots
/// </summary>
public class IncrementalDelta
{
    /// <summary>
    /// Timestamp of baseline snapshot
    /// </summary>
    public DateTime BaselineTimestamp { get; set; }

    /// <summary>
    /// Timestamp of current snapshot
    /// </summary>
    public DateTime CurrentTimestamp { get; set; }

    /// <summary>
    /// List of changed tensors with their data
    /// </summary>
    public List<TensorDelta> ChangedTensors { get; set; } = new();

    /// <summary>
    /// List of changed optimizer tensors with their data
    /// </summary>
    public List<TensorDelta> ChangedOptimizerTensors { get; set; } = new();
}
