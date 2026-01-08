using System.Reflection;

namespace MachineLearning.Checkpointing;

/// <summary>
/// Retry policy for fault tolerance
/// </summary>
public class RetryPolicy
{
    /// <summary>
    /// Maximum number of retry attempts
    /// </summary>
    public int MaxRetries { get; set; } = 3;

    /// <summary>
    /// Initial delay between retry attempts
    /// </summary>
    public TimeSpan InitialDelay { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Maximum delay between retry attempts
    /// </summary>
    public TimeSpan MaxDelay { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Backoff multiplier for exponential backoff
    /// </summary>
    public double BackoffFactor { get; set; } = 2.0;

    /// <summary>
    /// List of exception types that should trigger a retry
    /// </summary>
    public List<Type> RetryableExceptions { get; set; } = new();

    /// <summary>
    /// Create a new retry policy
    /// </summary>
    public RetryPolicy() { }

    /// <summary>
    /// Create a new retry policy with specified parameters
    /// </summary>
    public RetryPolicy(int maxRetries, TimeSpan initialDelay, TimeSpan maxDelay, double backoffFactor)
    {
        MaxRetries = maxRetries;
        InitialDelay = initialDelay;
        MaxDelay = maxDelay;
        BackoffFactor = backoffFactor;
    }

    /// <summary>
    /// Check if an exception is retryable based on the policy
    /// </summary>
    public bool IsRetryable(Exception ex)
    {
        return RetryableExceptions.Any(type => type.IsInstanceOfType(ex));
    }
}
