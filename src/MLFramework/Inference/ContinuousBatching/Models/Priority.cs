namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Define request priority levels for scheduler.
/// </summary>
public enum Priority
{
    /// <summary>
    /// Low priority request.
    /// </summary>
    Low = 0,

    /// <summary>
    /// Normal priority request.
    /// </summary>
    Normal = 1,

    /// <summary>
    /// High priority request.
    /// </summary>
    High = 2,

    /// <summary>
    /// Urgent priority request.
    /// </summary>
    Urgent = 3
}
