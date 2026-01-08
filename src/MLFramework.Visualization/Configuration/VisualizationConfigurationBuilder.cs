namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Fluent builder for VisualizationConfiguration
/// </summary>
public class VisualizationConfigurationBuilder
{
    private VisualizationConfiguration _config;

    public VisualizationConfigurationBuilder()
    {
        _config = new VisualizationConfiguration();
    }

    /// <summary>
    /// Set the storage configuration
    /// </summary>
    public VisualizationConfigurationBuilder WithStorage(StorageConfiguration storage)
    {
        _config.Storage = storage ?? new StorageConfiguration();
        return this;
    }

    /// <summary>
    /// Set the storage directory
    /// </summary>
    public VisualizationConfigurationBuilder WithStorageDirectory(string directory)
    {
        _config.Storage.LogDirectory = directory;
        return this;
    }

    /// <summary>
    /// Set the storage backend type
    /// </summary>
    public VisualizationConfigurationBuilder WithStorageBackend(string backendType)
    {
        _config.Storage.BackendType = backendType;
        return this;
    }

    /// <summary>
    /// Set the logging configuration
    /// </summary>
    public VisualizationConfigurationBuilder WithLogging(LoggingConfiguration logging)
    {
        _config.Logging = logging ?? new LoggingConfiguration();
        return this;
    }

    /// <summary>
    /// Set the log prefix for scalars
    /// </summary>
    public VisualizationConfigurationBuilder WithLogPrefix(string prefix)
    {
        _config.Logging.ScalarLogPrefix = prefix;
        return this;
    }

    /// <summary>
    /// Enable or disable scalar logging
    /// </summary>
    public VisualizationConfigurationBuilder WithScalarLogging(bool enabled)
    {
        _config.Logging.LogScalars = enabled;
        return this;
    }

    /// <summary>
    /// Set the histogram bin count
    /// </summary>
    public VisualizationConfigurationBuilder WithHistogramBinCount(int binCount)
    {
        _config.Logging.HistogramBinCount = binCount;
        return this;
    }

    /// <summary>
    /// Set the profiling configuration
    /// </summary>
    public VisualizationConfigurationBuilder WithProfiling(ProfilingConfiguration profiling)
    {
        _config.Profiling = profiling ?? new ProfilingConfiguration();
        return this;
    }

    /// <summary>
    /// Enable or disable profiling
    /// </summary>
    public VisualizationConfigurationBuilder EnableProfiling(bool enabled)
    {
        _config.Profiling.EnableProfiling = enabled;
        return this;
    }

    /// <summary>
    /// Enable or disable CPU profiling
    /// </summary>
    public VisualizationConfigurationBuilder EnableCPUProfiling(bool enabled)
    {
        _config.Profiling.ProfileCPU = enabled;
        return this;
    }

    /// <summary>
    /// Enable or disable GPU profiling
    /// </summary>
    public VisualizationConfigurationBuilder EnableGPUProfiling(bool enabled)
    {
        _config.Profiling.ProfileGPU = enabled;
        return this;
    }

    /// <summary>
    /// Set the maximum stored operations for profiling
    /// </summary>
    public VisualizationConfigurationBuilder WithMaxStoredOperations(int max)
    {
        _config.Profiling.MaxStoredOperations = max;
        return this;
    }

    /// <summary>
    /// Set the memory profiling configuration
    /// </summary>
    public VisualizationConfigurationBuilder WithMemoryProfiling(MemoryProfilingConfiguration memoryProfiling)
    {
        _config.MemoryProfiling = memoryProfiling ?? new MemoryProfilingConfiguration();
        return this;
    }

    /// <summary>
    /// Enable or disable memory profiling
    /// </summary>
    public VisualizationConfigurationBuilder EnableMemoryProfiling(bool enabled)
    {
        _config.MemoryProfiling.EnableMemoryProfiling = enabled;
        return this;
    }

    /// <summary>
    /// Set the memory snapshot interval
    /// </summary>
    public VisualizationConfigurationBuilder WithMemorySnapshotIntervalMs(int intervalMs)
    {
        _config.MemoryProfiling.SnapshotIntervalMs = intervalMs;
        return this;
    }

    /// <summary>
    /// Set the GPU tracking configuration
    /// </summary>
    public VisualizationConfigurationBuilder WithGPUTracking(GPUTrackingConfiguration gpuTracking)
    {
        _config.GPUTracking = gpuTracking ?? new GPUTrackingConfiguration();
        return this;
    }

    /// <summary>
    /// Enable or disable GPU tracking
    /// </summary>
    public VisualizationConfigurationBuilder EnableGPUTracking(bool enabled)
    {
        _config.GPUTracking.EnableGPUTracking = enabled;
        return this;
    }

    /// <summary>
    /// Set the GPU sampling interval
    /// </summary>
    public VisualizationConfigurationBuilder WithGPUSamplingIntervalMs(int intervalMs)
    {
        _config.GPUTracking.SamplingIntervalMs = intervalMs;
        return this;
    }

    /// <summary>
    /// Set the event collection configuration
    /// </summary>
    public VisualizationConfigurationBuilder WithEventCollection(EventCollectionConfiguration eventCollection)
    {
        _config.EventCollection = eventCollection ?? new EventCollectionConfiguration();
        return this;
    }

    /// <summary>
    /// Enable or disable async event collection
    /// </summary>
    public VisualizationConfigurationBuilder EnableAsyncEventCollection(bool enabled)
    {
        _config.EventCollection.EnableAsync = enabled;
        return this;
    }

    /// <summary>
    /// Set the event buffer capacity
    /// </summary>
    public VisualizationConfigurationBuilder WithEventBufferCapacity(int capacity)
    {
        _config.EventCollection.BufferCapacity = capacity;
        return this;
    }

    /// <summary>
    /// Set the event batch size
    /// </summary>
    public VisualizationConfigurationBuilder WithEventBatchSize(int batchSize)
    {
        _config.EventCollection.BatchSize = batchSize;
        return this;
    }

    /// <summary>
    /// Enable or disable the visualization globally
    /// </summary>
    public VisualizationConfigurationBuilder Enable(bool enabled)
    {
        _config.IsEnabled = enabled;
        return this;
    }

    /// <summary>
    /// Enable or disable verbose logging
    /// </summary>
    public VisualizationConfigurationBuilder WithVerboseLogging(bool enabled)
    {
        _config.VerboseLogging = enabled;
        return this;
    }

    /// <summary>
    /// Build the configuration and validate it
    /// </summary>
    public VisualizationConfiguration Build()
    {
        var validator = new ConfigurationValidator();
        validator.ValidateAndThrow(_config);
        return _config;
    }

    /// <summary>
    /// Build the configuration without validation
    /// </summary>
    public VisualizationConfiguration BuildWithoutValidation()
    {
        return _config;
    }

    /// <summary>
    /// Build the configuration and return validation result
    /// </summary>
    public (VisualizationConfiguration config, ValidationResult result) BuildWithValidation()
    {
        var validator = new ConfigurationValidator();
        var result = validator.Validate(_config);
        return (_config, result);
    }
}
