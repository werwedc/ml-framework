namespace MachineLearning.Checkpointing;

/// <summary>
/// Global checkpoint options
/// </summary>
public class CheckpointOptions
{
    /// <summary>
    /// Storage configuration
    /// </summary>
    public StorageOptions Storage { get; set; } = new();

    /// <summary>
    /// Retry policy
    /// </summary>
    public RetryPolicy RetryPolicy { get; set; } = new();

    /// <summary>
    /// Integrity checkers
    /// </summary>
    public List<IIntegrityChecker> IntegrityCheckers { get; set; } = new();

    /// <summary>
    /// Compatibility checkers
    /// </summary>
    public List<ICompatibilityChecker> CompatibilityCheckers { get; set; } = new();

    /// <summary>
    /// Default timeout for operations
    /// </summary>
    public TimeSpan DefaultTimeout { get; set; } = TimeSpan.FromMinutes(10);
}
