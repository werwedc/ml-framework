namespace MachineLearning.Distributed.Checkpointing;

/// <summary>
/// Represents a training checkpoint
/// </summary>
public class TrainingCheckpoint
{
    /// <summary>
    /// Unique identifier for this checkpoint
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Current training epoch
    /// </summary>
    public int Epoch { get; set; }

    /// <summary>
    /// Current training step
    /// </summary>
    public int Step { get; set; }

    /// <summary>
    /// Learning rate at checkpoint time
    /// </summary>
    public float LearningRate { get; set; }

    /// <summary>
    /// Serialized model state
    /// </summary>
    public byte[]? ModelState { get; set; }

    /// <summary>
    /// Serialized optimizer state
    /// </summary>
    public byte[]? OptimizerState { get; set; }

    /// <summary>
    /// Number of workers at checkpoint time
    /// </summary>
    public int WorkerCount { get; set; }

    /// <summary>
    /// Whether this checkpoint was created during rescaling
    /// </summary>
    public bool IsRescalingCheckpoint { get; set; }

    /// <summary>
    /// Timestamp when checkpoint was created
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Optional metadata
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
