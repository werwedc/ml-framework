namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Configuration for performance profiling
/// </summary>
public class ProfilingConfiguration
{
    /// <summary>
    /// Enable profiling
    /// </summary>
    public bool EnableProfiling { get; set; } = true;

    /// <summary>
    /// Profile forward pass operations
    /// </summary>
    public bool ProfileForwardPass { get; set; } = true;

    /// <summary>
    /// Profile backward pass operations
    /// </summary>
    public bool ProfileBackwardPass { get; set; } = true;

    /// <summary>
    /// Profile optimizer step operations
    /// </summary>
    public bool ProfileOptimizerStep { get; set; } = false;

    /// <summary>
    /// Profile CPU operations
    /// </summary>
    public bool ProfileCPU { get; set; } = true;

    /// <summary>
    /// Profile GPU operations
    /// </summary>
    public bool ProfileGPU { get; set; } = true;

    /// <summary>
    /// Maximum number of operations to store
    /// </summary>
    public int MaxStoredOperations { get; set; } = 10000;
}
