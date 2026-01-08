namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for checkpoint compatibility validation
/// </summary>
public interface ICompatibilityChecker
{
    /// <summary>
    /// Check if a checkpoint is compatible with the current setup
    /// </summary>
    Task<CompatibilityCheckResult> CheckCompatibilityAsync(
        CheckpointMetadata checkpointMetadata,
        int currentWorldSize,
        bool strictMode = true,
        CancellationToken cancellationToken = default);
}

/// <summary>
/// Result of a compatibility check
/// </summary>
public class CompatibilityCheckResult
{
    /// <summary>
    /// Whether the checkpoint is compatible
    /// </summary>
    public bool IsCompatible { get; set; }

    /// <summary>
    /// Whether resharding is required
    /// </summary>
    public bool RequiresResharding { get; set; }

    /// <summary>
    /// Warning messages (non-critical issues)
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Error messages (critical issues)
    /// </summary>
    public List<string> Errors { get; set; } = new();

    /// <summary>
    /// Get a summary of the compatibility check results
    /// </summary>
    public string GetSummary()
    {
        var summary = new System.Text.StringBuilder();
        summary.AppendLine($"Compatibility Check: {(IsCompatible ? "COMPATIBLE" : "INCOMPATIBLE")}");

        if (RequiresResharding)
        {
            summary.AppendLine("Resharding is required");
        }

        if (Warnings.Count > 0)
        {
            summary.AppendLine($"Warnings ({Warnings.Count}):");
            foreach (var warning in Warnings)
            {
                summary.AppendLine($"  - {warning}");
            }
        }

        if (Errors.Count > 0)
        {
            summary.AppendLine($"Errors ({Errors.Count}):");
            foreach (var error in Errors)
            {
                summary.AppendLine($"  - {error}");
            }
        }

        return summary.ToString();
    }
}
