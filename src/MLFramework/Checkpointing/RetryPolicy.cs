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
    /// Delay between retry attempts
    /// </summary>
    public TimeSpan RetryDelay { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Whether to use exponential backoff
    /// </summary>
    public bool UseExponentialBackoff { get; set; } = false;

    /// <summary>
    /// Maximum backoff delay (when using exponential backoff)
    /// </summary>
    public TimeSpan MaxBackoffDelay { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Create a new retry policy
    /// </summary>
    public RetryPolicy() { }

    /// <summary>
    /// Create a new retry policy with specified parameters
    /// </summary>
    public RetryPolicy(int maxRetries, TimeSpan retryDelay)
    {
        MaxRetries = maxRetries;
        RetryDelay = retryDelay;
    }
}
