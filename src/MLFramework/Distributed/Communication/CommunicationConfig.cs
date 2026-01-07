namespace MLFramework.Distributed.Communication;

/// <summary>
/// Configuration options for communication operations.
/// </summary>
public class CommunicationConfig
{
    /// <summary>
    /// Gets or sets the timeout in milliseconds for communication operations.
    /// Default is 300000 (5 minutes).
    /// </summary>
    public int TimeoutMs { get; set; } = 300000;

    /// <summary>
    /// Gets or sets whether to enable detailed logging for communication operations.
    /// Default is false.
    /// </summary>
    public bool EnableLogging { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use pinned memory for communication buffers.
    /// Default is true.
    /// </summary>
    public bool UsePinnedMemory { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of retries for failed operations.
    /// Default is 3.
    /// </summary>
    public int MaxRetries { get; set; } = 3;

    /// <summary>
    /// Gets or sets the delay between retries in milliseconds.
    /// Default is 100.
    /// </summary>
    public int RetryDelayMs { get; set; } = 100;
}
