namespace MachineLearning.Checkpointing;

/// <summary>
/// Verification result for checkpoint integrity checks
/// </summary>
public class VerificationResult
{
    /// <summary>
    /// List of errors found during verification
    /// </summary>
    public List<string> Errors { get; } = new();

    /// <summary>
    /// List of warnings found during verification
    /// </summary>
    public List<string> Warnings { get; } = new();

    /// <summary>
    /// Whether the verification passed without errors
    /// </summary>
    public bool IsValid => Errors.Count == 0;

    /// <summary>
    /// Add an error to the verification result
    /// </summary>
    /// <param name="error">Error message</param>
    public void AddError(string error)
    {
        if (!string.IsNullOrWhiteSpace(error))
        {
            Errors.Add(error);
        }
    }

    /// <summary>
    /// Add a warning to the verification result
    /// </summary>
    /// <param name="warning">Warning message</param>
    public void AddWarning(string warning)
    {
        if (!string.IsNullOrWhiteSpace(warning))
        {
            Warnings.Add(warning);
        }
    }
}
