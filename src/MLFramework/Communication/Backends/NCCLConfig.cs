namespace MLFramework.Communication.Backends;

/// <summary>
/// NCCL-specific configuration
/// </summary>
public class NCCLConfig
{
    /// <summary>
    /// Use NCCL's ring-based all-reduce (default)
    /// </summary>
    public bool UseRingAllReduce { get; set; } = true;

    /// <summary>
    /// Use NCCL's tree-based all-reduce
    /// </summary>
    public bool UseTreeAllReduce { get; set; } = false;

    /// <summary>
    /// Threshold for switching from ring to tree all-reduce (bytes)
    /// </summary>
    public long TreeThresholdBytes { get; set; } = 1024 * 1024; // 1MB

    /// <summary>
    /// Number of channels for multi-rail communication
    /// </summary>
    public int NumChannels { get; set; } = 1;

    /// <summary>
    /// Enable NCCL debugging
    /// </summary>
    public bool EnableDebug { get; set; } = false;

    /// <summary>
    /// NCCL buffer size for multi-threaded communication
    /// </summary>
    public int BufferSize { get; set; } = 4194304; // 4MB

    /// <summary>
    /// Use NCCL's built-in asynchronous operations
    /// </summary>
    public bool UseAsyncOps { get; set; } = true;

    /// <summary>
    /// Timeout for NCCL operations (milliseconds)
    /// </summary>
    public int TimeoutMs { get; set; } = 300000; // 5 minutes

    /// <summary>
    /// Set NCCL environment variable
    /// </summary>
    public static void SetEnvironmentVariable(string key, string value)
    {
        Environment.SetEnvironmentVariable(key, value);
    }

    /// <summary>
    /// Apply NCCL configuration
    /// </summary>
    public void Apply()
    {
        SetEnvironmentVariable("NCCL_DEBUG", EnableDebug ? "INFO" : "WARN");
        SetEnvironmentVariable("NCCL_BUFFSIZE", BufferSize.ToString());

        if (NumChannels > 1)
        {
            SetEnvironmentVariable("NCCL_NCHANNELS", NumChannels.ToString());
        }

        // Set timeout
        SetEnvironmentVariable("NCCL_BLOCKING_WAIT", TimeoutMs.ToString());

        // Algorithm selection
        if (UseTreeAllReduce)
        {
            SetEnvironmentVariable("NCCL_ALGO", "Tree");
        }
        else if (UseRingAllReduce)
        {
            SetEnvironmentVariable("NCCL_ALGO", "Ring");
        }
    }

    /// <summary>
    /// Validate NCCL configuration
    /// </summary>
    /// <returns>True if configuration is valid</returns>
    public bool Validate()
    {
        if (BufferSize <= 0)
        {
            return false;
        }

        if (NumChannels <= 0)
        {
            return false;
        }

        if (TimeoutMs <= 0)
        {
            return false;
        }

        if (TreeThresholdBytes < 0)
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Create a default NCCL configuration optimized for typical workloads
    /// </summary>
    /// <returns>Default NCCL configuration</returns>
    public static NCCLConfig CreateDefault()
    {
        return new NCCLConfig
        {
            UseRingAllReduce = true,
            UseTreeAllReduce = false,
            TreeThresholdBytes = 1024 * 1024, // 1MB
            NumChannels = 1,
            EnableDebug = false,
            BufferSize = 4194304, // 4MB
            UseAsyncOps = true,
            TimeoutMs = 300000 // 5 minutes
        };
    }

    /// <summary>
    /// Create a high-performance NCCL configuration for large-scale training
    /// </summary>
    /// <returns>High-performance NCCL configuration</returns>
    public static NCCLConfig CreateHighPerformance()
    {
        return new NCCLConfig
        {
            UseRingAllReduce = true,
            UseTreeAllReduce = true,
            TreeThresholdBytes = 16 * 1024 * 1024, // 16MB
            NumChannels = 4, // Multi-rail
            EnableDebug = false,
            BufferSize = 16 * 1024 * 1024, // 16MB
            UseAsyncOps = true,
            TimeoutMs = 600000 // 10 minutes
        };
    }

    /// <summary>
    /// Create a debug NCCL configuration with verbose logging
    /// </summary>
    /// <returns>Debug NCCL configuration</returns>
    public static NCCLConfig CreateDebug()
    {
        return new NCCLConfig
        {
            UseRingAllReduce = true,
            UseTreeAllReduce = false,
            TreeThresholdBytes = 1024 * 1024, // 1MB
            NumChannels = 1,
            EnableDebug = true,
            BufferSize = 4194304, // 4MB
            UseAsyncOps = true,
            TimeoutMs = 600000 // 10 minutes
        };
    }
}
