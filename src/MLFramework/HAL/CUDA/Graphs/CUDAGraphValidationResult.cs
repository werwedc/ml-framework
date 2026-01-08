namespace MLFramework.HAL.CUDA;

/// <summary>
/// Validation result for CUDA graphs
/// </summary>
public class CUDAGraphValidationResult
{
    /// <summary>
    /// Gets whether the graph is valid for execution
    /// </summary>
    public bool IsValid { get; init; }

    /// <summary>
    /// Gets list of validation errors (empty if valid)
    /// </summary>
    public IReadOnlyList<string> Errors { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Gets list of warnings (non-critical issues)
    /// </summary>
    public IReadOnlyList<string> Warnings { get; init; } = Array.Empty<string>();

    /// <summary>
    /// Gets the number of captured operations
    /// </summary>
    public int OperationCount { get; init; }

    /// <summary>
    /// Creates a successful validation result
    /// </summary>
    public static CUDAGraphValidationResult Success(int operationCount)
    {
        return new CUDAGraphValidationResult
        {
            IsValid = true,
            Errors = Array.Empty<string>(),
            Warnings = Array.Empty<string>(),
            OperationCount = operationCount
        };
    }

    /// <summary>
    /// Creates a failed validation result with errors
    /// </summary>
    public static CUDAGraphValidationResult Failure(IEnumerable<string> errors, int operationCount = 0)
    {
        return new CUDAGraphValidationResult
        {
            IsValid = false,
            Errors = errors.ToArray(),
            Warnings = Array.Empty<string>(),
            OperationCount = operationCount
        };
    }

    /// <summary>
    /// Creates a validation result with warnings
    /// </summary>
    public CUDAGraphValidationResult WithWarnings(IEnumerable<string> warnings)
    {
        return new CUDAGraphValidationResult
        {
            IsValid = IsValid,
            Errors = Errors,
            Warnings = warnings.ToArray(),
            OperationCount = OperationCount
        };
    }
}
