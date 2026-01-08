namespace MLFramework.Data;

/// <summary>
/// Defines how errors should be handled in the data loading pipeline.
/// </summary>
public enum ErrorPolicy
{
    /// <summary>
    /// Stop the entire dataloader immediately when any error occurs.
    /// This is the most conservative policy and ensures no corrupted data is processed.
    /// </summary>
    FailFast,

    /// <summary>
    /// Skip the failed worker and continue processing with remaining workers.
    /// This allows the dataloader to continue with reduced capacity.
    /// </summary>
    Continue,

    /// <summary>
    /// Attempt to restart the failed worker up to the maximum retry limit.
    /// This provides automatic recovery from transient failures.
    /// </summary>
    Restart,

    /// <summary>
    /// Silently ignore errors (not recommended).
    /// This should only be used in specific scenarios where data integrity is not critical.
    /// </summary>
    Ignore
}
