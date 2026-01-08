namespace MLFramework.Communication.Backends;

/// <summary>
/// RCCL-specific configuration
/// </summary>
public class RCCLConfig
{
    /// <summary>
    /// Use RCCL's ring-based all-reduce (default)
    /// </summary>
    public bool UseRingAllReduce { get; set; } = true;

    /// <summary>
    /// Use RCCL's tree-based all-reduce
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
    /// Enable RCCL debugging
    /// </summary>
    public bool EnableDebug { get; set; } = false;

    /// <summary>
    /// RCCL buffer size for multi-threaded communication
    /// </summary>
    public int BufferSize { get; set; } = 4194304; // 4MB

    /// <summary>
    /// Use RCCL's built-in asynchronous operations
    /// </summary>
    public bool UseAsyncOps { get; set; } = true;

    /// <summary>
    /// Timeout for RCCL operations (milliseconds)
    /// </summary>
    public int TimeoutMs { get; set; } = 300000; // 5 minutes

    /// <summary>
    /// Set RCCL environment variable
    /// </summary>
    public static void SetEnvironmentVariable(string key, string value)
    {
        Environment.SetEnvironmentVariable(key, value);
    }

    /// <summary>
    /// Apply RCCL configuration
    /// </summary>
    public void Apply()
    {
        SetEnvironmentVariable("RCCL_DEBUG", EnableDebug ? "INFO" : "WARN");
        SetEnvironmentVariable("RCCL_BUFFSIZE", BufferSize.ToString());

        if (NumChannels > 1)
        {
            SetEnvironmentVariable("RCCL_NCHANNELS", NumChannels.ToString());
        }

        // Set timeout
        SetEnvironmentVariable("RCCL_BLOCKING_WAIT", TimeoutMs.ToString());

        // Algorithm selection
        if (UseTreeAllReduce)
        {
            SetEnvironmentVariable("RCCL_ALGO", "Tree");
        }
        else if (UseRingAllReduce)
        {
            SetEnvironmentVariable("RCCL_ALGO", "Ring");
        }
    }

    /// <summary>
    /// Validate RCCL configuration
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
    /// Create a default RCCL configuration optimized for typical workloads
    /// </summary>
    /// <returns>Default RCCL configuration</returns>
    public static RCCLConfig CreateDefault()
    {
        return new RCCLConfig
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
    /// Create a high-performance RCCL configuration for large-scale training
    /// </summary>
    /// <returns>High-performance RCCL configuration</returns>
    public static RCCLConfig CreateHighPerformance()
    {
        return new RCCLConfig
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
    /// Create a debug RCCL configuration with verbose logging
    /// </summary>
    /// <returns>Debug RCCL configuration</returns>
    public static RCCLConfig CreateDebug()
    {
        return new RCCLConfig
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
