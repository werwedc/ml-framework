namespace MachineLearning.Visualization.Storage;

/// <summary>
/// Configuration for storage backends
/// </summary>
public class StorageConfiguration
{
    /// <summary>
    /// Gets or sets the type of backend to use (e.g., "file", "memory", "remote")
    /// </summary>
    public string BackendType { get; set; } = "file";

    /// <summary>
    /// Gets or sets the connection string for the backend
    /// </summary>
    public string ConnectionString { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the write buffer size (number of events to buffer before flushing)
    /// </summary>
    public int WriteBufferSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets the flush interval (automatic flush after this duration)
    /// </summary>
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Gets or sets whether to enable asynchronous writes
    /// </summary>
    public bool EnableAsyncWrites { get; set; } = true;

    /// <summary>
    /// Validates the configuration
    /// </summary>
    /// <returns>True if valid, false otherwise</returns>
    public bool IsValid()
    {
        if (string.IsNullOrWhiteSpace(BackendType))
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(ConnectionString))
        {
            return false;
        }

        if (WriteBufferSize <= 0)
        {
            return false;
        }

        if (FlushInterval <= TimeSpan.Zero)
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Throws an exception if the configuration is invalid
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when configuration is invalid</exception>
    public void EnsureValid()
    {
        if (string.IsNullOrWhiteSpace(BackendType))
        {
            throw new InvalidOperationException("BackendType cannot be null or empty");
        }

        if (string.IsNullOrWhiteSpace(ConnectionString))
        {
            throw new InvalidOperationException("ConnectionString cannot be null or empty");
        }

        if (WriteBufferSize <= 0)
        {
            throw new InvalidOperationException("WriteBufferSize must be greater than 0");
        }

        if (FlushInterval <= TimeSpan.Zero)
        {
            throw new InvalidOperationException("FlushInterval must be greater than TimeSpan.Zero");
        }
    }
}
