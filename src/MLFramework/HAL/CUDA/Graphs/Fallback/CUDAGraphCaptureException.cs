namespace MLFramework.HAL.CUDA.Graphs;

/// <summary>
/// Exception thrown when CUDA graph capture fails
/// </summary>
public class CUDAGraphCaptureException : Exception
{
    /// <summary>
    /// Gets the validation result if the exception was raised due to validation failure
    /// </summary>
    public CUDAGraphValidationResult? ValidationResult { get; }

    /// <summary>
    /// Initializes a new instance of the CUDAGraphCaptureException class
    /// </summary>
    /// <param name="message">Error message</param>
    public CUDAGraphCaptureException(string message)
        : base(message)
    {
        ValidationResult = null;
    }

    /// <summary>
    /// Initializes a new instance of the CUDAGraphCaptureException class
    /// </summary>
    /// <param name="message">Error message</param>
    /// <param name="innerException">Inner exception that caused this exception</param>
    public CUDAGraphCaptureException(string message, Exception innerException)
        : base(message, innerException)
    {
        ValidationResult = null;
    }

    /// <summary>
    /// Initializes a new instance of the CUDAGraphCaptureException class
    /// </summary>
    /// <param name="message">Error message</param>
    /// <param name="validationResult">Validation result that caused the failure</param>
    public CUDAGraphCaptureException(string message, CUDAGraphValidationResult validationResult)
        : base(message)
    {
        ValidationResult = validationResult;
    }

    /// <summary>
    /// Initializes a new instance of the CUDAGraphCaptureException class
    /// </summary>
    /// <param name="message">Error message</param>
    /// <param name="validationResult">Validation result that caused the failure</param>
    /// <param name="innerException">Inner exception that caused this exception</param>
    public CUDAGraphCaptureException(
        string message,
        CUDAGraphValidationResult validationResult,
        Exception innerException)
        : base(message, innerException)
    {
        ValidationResult = validationResult;
    }
}
