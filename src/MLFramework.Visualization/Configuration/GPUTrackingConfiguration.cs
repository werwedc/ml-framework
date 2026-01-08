namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Configuration for GPU tracking
/// </summary>
public class GPUTrackingConfiguration
{
    /// <summary>
    /// Enable GPU tracking
    /// </summary>
    public bool EnableGPUTracking { get; set; } = false;

    /// <summary>
    /// Sampling interval in milliseconds
    /// </summary>
    public int SamplingIntervalMs { get; set; } = 1000;

    /// <summary>
    /// Track GPU temperature
    /// </summary>
    public bool TrackTemperature { get; set; } = true;

    /// <summary>
    /// Track GPU power consumption
    /// </summary>
    public bool TrackPower { get; set; } = true;
}
