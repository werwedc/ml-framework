namespace MLFramework.Serving.Deployment;

/// <summary>
/// Result of a rollback operation
/// </summary>
public class RollbackResult
{
    /// <summary>
    /// Whether the rollback was successful
    /// </summary>
    public bool Success { get; }

    /// <summary>
    /// ID of the previous deployment being rolled back to
    /// </summary>
    public string PreviousDeploymentId { get; }

    /// <summary>
    /// ID of the current deployment being rolled back from
    /// </summary>
    public string CurrentDeploymentId { get; }

    /// <summary>
    /// Timestamp when the rollback occurred
    /// </summary>
    public DateTime RollbackTime { get; }

    /// <summary>
    /// Message describing the rollback result
    /// </summary>
    public string Message { get; }

    public RollbackResult(
        bool success,
        string previousDeploymentId,
        string currentDeploymentId,
        DateTime rollbackTime,
        string message)
    {
        Success = success;
        PreviousDeploymentId = previousDeploymentId ?? throw new ArgumentNullException(nameof(previousDeploymentId));
        CurrentDeploymentId = currentDeploymentId ?? throw new ArgumentNullException(nameof(currentDeploymentId));
        RollbackTime = rollbackTime;
        Message = message ?? throw new ArgumentNullException(nameof(message));
    }
}
