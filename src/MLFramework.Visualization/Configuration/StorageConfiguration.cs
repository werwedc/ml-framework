namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Configuration for storage backend settings
/// </summary>
public class StorageConfiguration
{
    /// <summary>
    /// Type of storage backend ("file", "database", etc.)
    /// </summary>
    public string BackendType { get; set; } = "file";

    /// <summary>
    /// Directory for log storage
    /// </summary>
    public string LogDirectory { get; set; } = "./logs";

    /// <summary>
    /// Connection string for database backends
    /// </summary>
    public string ConnectionString { get; set; }

    /// <summary>
    /// Buffer size for write operations
    /// </summary>
    public int WriteBufferSize { get; set; } = 100;

    /// <summary>
    /// Interval between automatic flushes
    /// </summary>
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Enable asynchronous writes
    /// </summary>
    public bool EnableAsyncWrites { get; set; } = true;
}
