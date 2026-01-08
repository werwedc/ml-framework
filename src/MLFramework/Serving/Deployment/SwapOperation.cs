namespace MLFramework.Serving.Deployment;

/// <summary>
/// Represents a model version swap operation
/// </summary>
public class SwapOperation
{
    /// <summary>
    /// Unique identifier for this swap operation
    /// </summary>
    public string OperationId { get; }

    /// <summary>
    /// Name of the model being swapped
    /// </summary>
    public string ModelName { get; }

    /// <summary>
    /// Source version
    /// </summary>
    public string FromVersion { get; }

    /// <summary>
    /// Target version
    /// </summary>
    public string ToVersion { get; }

    /// <summary>
    /// Current state of the swap
    /// </summary>
    public SwapState State { get; set; }

    /// <summary>
    /// When the swap started
    /// </summary>
    public DateTime StartTime { get; }

    /// <summary>
    /// When the swap ended (null if still in progress)
    /// </summary>
    public DateTime? EndTime { get; set; }

    /// <summary>
    /// Error message if the swap failed
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Create a new swap operation
    /// </summary>
    public SwapOperation(string operationId, string modelName, string fromVersion, string toVersion)
    {
        OperationId = operationId;
        ModelName = modelName;
        FromVersion = fromVersion;
        ToVersion = toVersion;
        State = SwapState.NotStarted;
        StartTime = DateTime.UtcNow;
    }
}
