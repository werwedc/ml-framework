namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Main configuration class for visualization and profiling
/// </summary>
public class VisualizationConfiguration
{
    /// <summary>
    /// Storage backend configuration
    /// </summary>
    public StorageConfiguration Storage { get; set; } = new StorageConfiguration();

    /// <summary>
    /// Logging configuration
    /// </summary>
    public LoggingConfiguration Logging { get; set; } = new LoggingConfiguration();

    /// <summary>
    /// Profiling configuration
    /// </summary>
    public ProfilingConfiguration Profiling { get; set; } = new ProfilingConfiguration();

    /// <summary>
    /// Memory profiling configuration
    /// </summary>
    public MemoryProfilingConfiguration MemoryProfiling { get; set; } = new MemoryProfilingConfiguration();

    /// <summary>
    /// GPU tracking configuration
    /// </summary>
    public GPUTrackingConfiguration GPUTracking { get; set; } = new GPUTrackingConfiguration();

    /// <summary>
    /// Event collection configuration
    /// </summary>
    public EventCollectionConfiguration EventCollection { get; set; } = new EventCollectionConfiguration();

    /// <summary>
    /// Enable visualization and profiling
    /// </summary>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Enable verbose logging
    /// </summary>
    public bool VerboseLogging { get; set; } = false;
}
