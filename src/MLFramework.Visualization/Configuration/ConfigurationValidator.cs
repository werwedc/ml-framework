namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Configuration validator for VisualizationConfiguration
/// </summary>
public class ConfigurationValidator : IConfigurationValidator
{
    /// <summary>
    /// Validate configuration and return result
    /// </summary>
    public ValidationResult Validate(VisualizationConfiguration config)
    {
        if (config == null)
        {
            return new ValidationResult(
                new List<string> { "Configuration cannot be null" },
                new List<string>()
            );
        }

        var errors = new List<string>();
        var warnings = new List<string>();

        // Validate Storage configuration
        ValidateStorage(config.Storage, errors, warnings);

        // Validate Logging configuration
        ValidateLogging(config.Logging, errors, warnings);

        // Validate Profiling configuration
        ValidateProfiling(config.Profiling, errors, warnings);

        // Validate Memory Profiling configuration
        ValidateMemoryProfiling(config.MemoryProfiling, errors, warnings);

        // Validate GPU Tracking configuration
        ValidateGPUTracking(config.GPUTracking, errors, warnings);

        // Validate Event Collection configuration
        ValidateEventCollection(config.EventCollection, errors, warnings);

        return new ValidationResult(errors, warnings);
    }

    /// <summary>
    /// Validate configuration and throw exception if invalid
    /// </summary>
    public void ValidateAndThrow(VisualizationConfiguration config)
    {
        var result = Validate(config);

        if (!result.IsValid)
        {
            var message = $"Configuration validation failed:\n{result.GetSummary()}";
            throw new InvalidOperationException(message);
        }
    }

    private void ValidateStorage(StorageConfiguration storage, List<string> errors, List<string> warnings)
    {
        if (storage == null)
        {
            errors.Add("Storage configuration cannot be null");
            return;
        }

        if (string.IsNullOrWhiteSpace(storage.LogDirectory))
        {
            errors.Add("Storage.LogDirectory cannot be null or empty");
        }

        if (storage.WriteBufferSize <= 0)
        {
            errors.Add("Storage.WriteBufferSize must be positive");
        }

        if (storage.WriteBufferSize > 100000)
        {
            warnings.Add("Storage.WriteBufferSize is very large, may consume significant memory");
        }

        if (storage.FlushInterval.TotalMilliseconds <= 0)
        {
            errors.Add("Storage.FlushInterval must be positive");
        }

        if (storage.FlushInterval.TotalSeconds > 300)
        {
            warnings.Add("Storage.FlushInterval is very long, data may be lost if process crashes");
        }

        if (string.IsNullOrWhiteSpace(storage.BackendType))
        {
            errors.Add("Storage.BackendType cannot be null or empty");
        }
    }

    private void ValidateLogging(LoggingConfiguration logging, List<string> errors, List<string> warnings)
    {
        if (logging == null)
        {
            errors.Add("Logging configuration cannot be null");
            return;
        }

        if (logging.HistogramBinCount <= 0)
        {
            errors.Add("Logging.HistogramBinCount must be positive");
        }

        if (logging.HistogramBinCount > 1000)
        {
            warnings.Add("Logging.HistogramBinCount is very large, may impact performance");
        }

        if (logging.DefaultSmoothingWindow <= 0)
        {
            errors.Add("Logging.DefaultSmoothingWindow must be positive");
        }

        if (logging.DefaultSmoothingWindow > 1000)
        {
            warnings.Add("Logging.DefaultSmoothingWindow is very large, smoothing will be very aggressive");
        }
    }

    private void ValidateProfiling(ProfilingConfiguration profiling, List<string> errors, List<string> warnings)
    {
        if (profiling == null)
        {
            errors.Add("Profiling configuration cannot be null");
            return;
        }

        if (profiling.MaxStoredOperations <= 0)
        {
            errors.Add("Profiling.MaxStoredOperations must be positive");
        }

        if (profiling.MaxStoredOperations > 100000)
        {
            warnings.Add("Profiling.MaxStoredOperations is very large, may consume significant memory");
        }

        if (!profiling.ProfileCPU && !profiling.ProfileGPU && profiling.EnableProfiling)
        {
            warnings.Add("Profiling is enabled but neither CPU nor GPU profiling is selected");
        }
    }

    private void ValidateMemoryProfiling(MemoryProfilingConfiguration memoryProfiling, List<string> errors, List<string> warnings)
    {
        if (memoryProfiling == null)
        {
            errors.Add("MemoryProfiling configuration cannot be null");
            return;
        }

        if (memoryProfiling.MaxStackTraceDepth <= 0)
        {
            errors.Add("MemoryProfiling.MaxStackTraceDepth must be positive");
        }

        if (memoryProfiling.MaxStackTraceDepth > 50)
        {
            warnings.Add("MemoryProfiling.MaxStackTraceDepth is very large, may impact performance");
        }

        if (memoryProfiling.SnapshotIntervalMs <= 0)
        {
            errors.Add("MemoryProfiling.SnapshotIntervalMs must be positive");
        }

        if (memoryProfiling.SnapshotIntervalMs < 100 && memoryProfiling.AutoSnapshot)
        {
            warnings.Add("MemoryProfiling.SnapshotIntervalMs is very small, may impact performance");
        }

        if (memoryProfiling.CaptureStackTraces && !memoryProfiling.EnableMemoryProfiling)
        {
            warnings.Add("MemoryProfiling.CaptureStackTraces is enabled but memory profiling is disabled");
        }
    }

    private void ValidateGPUTracking(GPUTrackingConfiguration gpuTracking, List<string> errors, List<string> warnings)
    {
        if (gpuTracking == null)
        {
            errors.Add("GPUTracking configuration cannot be null");
            return;
        }

        if (gpuTracking.SamplingIntervalMs <= 0)
        {
            errors.Add("GPUTracking.SamplingIntervalMs must be positive");
        }

        if (gpuTracking.SamplingIntervalMs < 100 && gpuTracking.EnableGPUTracking)
        {
            warnings.Add("GPUTracking.SamplingIntervalMs is very small, may impact performance");
        }

        if (!gpuTracking.TrackTemperature && !gpuTracking.TrackPower && gpuTracking.EnableGPUTracking)
        {
            warnings.Add("GPUTracking is enabled but neither temperature nor power tracking is selected");
        }
    }

    private void ValidateEventCollection(EventCollectionConfiguration eventCollection, List<string> errors, List<string> warnings)
    {
        if (eventCollection == null)
        {
            errors.Add("EventCollection configuration cannot be null");
            return;
        }

        if (eventCollection.BufferCapacity <= 0)
        {
            errors.Add("EventCollection.BufferCapacity must be positive");
        }

        if (eventCollection.BufferCapacity > 100000)
        {
            warnings.Add("EventCollection.BufferCapacity is very large, may consume significant memory");
        }

        if (eventCollection.BatchSize <= 0)
        {
            errors.Add("EventCollection.BatchSize must be positive");
        }

        if (eventCollection.BatchSize > eventCollection.BufferCapacity)
        {
            errors.Add("EventCollection.BatchSize cannot be larger than BufferCapacity");
        }

        if (eventCollection.MaxQueueLength <= 0)
        {
            errors.Add("EventCollection.MaxQueueLength must be positive");
        }

        if (eventCollection.BatchSize > eventCollection.MaxQueueLength)
        {
            errors.Add("EventCollection.BatchSize cannot be larger than MaxQueueLength");
        }
    }
}
