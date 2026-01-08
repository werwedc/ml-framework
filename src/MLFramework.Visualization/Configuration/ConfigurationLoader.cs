using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Configuration loader supporting multiple sources
/// </summary>
public class ConfigurationLoader : IConfigurationLoader
{
    private readonly JsonSerializerOptions _jsonOptions;

    public ConfigurationLoader()
    {
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
    }

    /// <summary>
    /// Load default configuration with environment variable overrides
    /// </summary>
    public VisualizationConfiguration Load()
    {
        var config = new VisualizationConfiguration();
        LoadFromEnvironment(config);
        return config;
    }

    /// <summary>
    /// Load configuration from a JSON file
    /// </summary>
    public VisualizationConfiguration LoadFromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Configuration file not found: {filePath}");
        }

        var json = File.ReadAllText(filePath);
        var config = LoadFromJson(json);
        LoadFromEnvironment(config); // Override with environment variables
        return config;
    }

    /// <summary>
    /// Load configuration from a JSON string
    /// </summary>
    public VisualizationConfiguration LoadFromJson(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return new VisualizationConfiguration();
        }

        try
        {
            var config = JsonSerializer.Deserialize<VisualizationConfiguration>(json, _jsonOptions);

            if (config == null)
            {
                return new VisualizationConfiguration();
            }

            // Ensure all nested objects are initialized
            config.Storage ??= new StorageConfiguration();
            config.Logging ??= new LoggingConfiguration();
            config.Profiling ??= new ProfilingConfiguration();
            config.MemoryProfiling ??= new MemoryProfilingConfiguration();
            config.GPUTracking ??= new GPUTrackingConfiguration();
            config.EventCollection ??= new EventCollectionConfiguration();

            return config;
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException($"Failed to parse JSON configuration: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Load configuration from environment variables
    /// </summary>
    public VisualizationConfiguration LoadFromEnvironment()
    {
        var config = new VisualizationConfiguration();
        LoadFromEnvironment(config);
        return config;
    }

    /// <summary>
    /// Apply environment variable overrides to configuration
    /// </summary>
    private void LoadFromEnvironment(VisualizationConfiguration config)
    {
        // Global settings
        config.IsEnabled = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_IS_ENABLED", config.IsEnabled);
        config.VerboseLogging = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_VERBOSE_LOGGING", config.VerboseLogging);

        // Storage settings
        if (config.Storage != null)
        {
            config.Storage.BackendType = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_STORAGE_BACKEND_TYPE", config.Storage.BackendType);
            config.Storage.LogDirectory = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_STORAGE_LOG_DIRECTORY", config.Storage.LogDirectory);
            config.Storage.ConnectionString = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_STORAGE_CONNECTION_STRING", config.Storage.ConnectionString);
            config.Storage.WriteBufferSize = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_STORAGE_WRITE_BUFFER_SIZE", config.Storage.WriteBufferSize);
            var flushSeconds = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_STORAGE_FLUSH_INTERVAL_SECONDS", config.Storage.FlushInterval.TotalSeconds);
            config.Storage.FlushInterval = TimeSpan.FromSeconds(flushSeconds);
            config.Storage.EnableAsyncWrites = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_STORAGE_ENABLE_ASYNC_WRITES", config.Storage.EnableAsyncWrites);
        }

        // Logging settings
        if (config.Logging != null)
        {
            config.Logging.LogScalars = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_LOGGING_LOG_SCALARS", config.Logging.LogScalars);
            config.Logging.LogHistograms = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_LOGGING_LOG_HISTOGRAMS", config.Logging.LogHistograms);
            config.Logging.LogGraphs = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_LOGGING_LOG_GRAPHS", config.Logging.LogGraphs);
            config.Logging.LogHyperparameters = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_LOGGING_LOG_HYPERPARAMETERS", config.Logging.LogHyperparameters);
            config.Logging.ScalarLogPrefix = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_LOGGING_SCALAR_LOG_PREFIX", config.Logging.ScalarLogPrefix);
            config.Logging.HistogramBinCount = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_LOGGING_HISTOGRAM_BIN_COUNT", config.Logging.HistogramBinCount);
            config.Logging.AutoSmoothScalars = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_LOGGING_AUTO_SMOOTH_SCALARS", config.Logging.AutoSmoothScalars);
            config.Logging.DefaultSmoothingWindow = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_LOGGING_DEFAULT_SMOOTHING_WINDOW", config.Logging.DefaultSmoothingWindow);
        }

        // Profiling settings
        if (config.Profiling != null)
        {
            config.Profiling.EnableProfiling = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_PROFILING_ENABLE_PROFILING", config.Profiling.EnableProfiling);
            config.Profiling.ProfileForwardPass = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_PROFILING_PROFILE_FORWARD_PASS", config.Profiling.ProfileForwardPass);
            config.Profiling.ProfileBackwardPass = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_PROFILING_PROFILE_BACKWARD_PASS", config.Profiling.ProfileBackwardPass);
            config.Profiling.ProfileOptimizerStep = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_PROFILING_PROFILE_OPTIMIZER_STEP", config.Profiling.ProfileOptimizerStep);
            config.Profiling.ProfileCPU = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_PROFILING_PROFILE_CPU", config.Profiling.ProfileCPU);
            config.Profiling.ProfileGPU = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_PROFILING_PROFILE_GPU", config.Profiling.ProfileGPU);
            config.Profiling.MaxStoredOperations = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_PROFILING_MAX_STORED_OPERATIONS", config.Profiling.MaxStoredOperations);
        }

        // Memory profiling settings
        if (config.MemoryProfiling != null)
        {
            config.MemoryProfiling.EnableMemoryProfiling = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_MEMORY_PROFILING_ENABLE_MEMORY_PROFILING", config.MemoryProfiling.EnableMemoryProfiling);
            config.MemoryProfiling.CaptureStackTraces = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_MEMORY_PROFILING_CAPTURE_STACK_TRACES", config.MemoryProfiling.CaptureStackTraces);
            config.MemoryProfiling.MaxStackTraceDepth = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_MEMORY_PROFILING_MAX_STACK_TRACE_DEPTH", config.MemoryProfiling.MaxStackTraceDepth);
            config.MemoryProfiling.SnapshotIntervalMs = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_MEMORY_PROFILING_SNAPSHOT_INTERVAL_MS", config.MemoryProfiling.SnapshotIntervalMs);
            config.MemoryProfiling.AutoSnapshot = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_MEMORY_PROFILING_AUTO_SNAPSHOT", config.MemoryProfiling.AutoSnapshot);
        }

        // GPU tracking settings
        if (config.GPUTracking != null)
        {
            config.GPUTracking.EnableGPUTracking = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_GPU_TRACKING_ENABLE_GPU_TRACKING", config.GPUTracking.EnableGPUTracking);
            config.GPUTracking.SamplingIntervalMs = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_GPU_TRACKING_SAMPLING_INTERVAL_MS", config.GPUTracking.SamplingIntervalMs);
            config.GPUTracking.TrackTemperature = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_GPU_TRACKING_TRACK_TEMPERATURE", config.GPUTracking.TrackTemperature);
            config.GPUTracking.TrackPower = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_GPU_TRACKING_TRACK_POWER", config.GPUTracking.TrackPower);
        }

        // Event collection settings
        if (config.EventCollection != null)
        {
            config.EventCollection.EnableAsync = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_EVENT_COLLECTION_ENABLE_ASYNC", config.EventCollection.EnableAsync);
            config.EventCollection.BufferCapacity = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_EVENT_COLLECTION_BUFFER_CAPACITY", config.EventCollection.BufferCapacity);
            config.EventCollection.BatchSize = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_EVENT_COLLECTION_BATCH_SIZE", config.EventCollection.BatchSize);
            config.EventCollection.EnableBackpressure = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_EVENT_COLLECTION_ENABLE_BACKPRESSURE", config.EventCollection.EnableBackpressure);
            config.EventCollection.MaxQueueLength = GetEnvironmentValue("MLFRAMEWORK_VISUALIZATION_EVENT_COLLECTION_MAX_QUEUE_LENGTH", config.EventCollection.MaxQueueLength);
        }
    }

    /// <summary>
    /// Get boolean value from environment variable, or default if not set
    /// </summary>
    private bool GetEnvironmentValue(string envVar, bool defaultValue)
    {
        var envValue = Environment.GetEnvironmentVariable(envVar);
        return !string.IsNullOrEmpty(envValue) && bool.TryParse(envValue, out var parsedValue)
            ? parsedValue
            : defaultValue;
    }

    /// <summary>
    /// Get integer value from environment variable, or default if not set
    /// </summary>
    private int GetEnvironmentValue(string envVar, int defaultValue)
    {
        var envValue = Environment.GetEnvironmentVariable(envVar);
        return !string.IsNullOrEmpty(envValue) && int.TryParse(envValue, out var parsedValue)
            ? parsedValue
            : defaultValue;
    }

    /// <summary>
    /// Get double value from environment variable, or default if not set
    /// </summary>
    private double GetEnvironmentValue(string envVar, double defaultValue)
    {
        var envValue = Environment.GetEnvironmentVariable(envVar);
        return !string.IsNullOrEmpty(envValue) && double.TryParse(envValue, out var parsedValue)
            ? parsedValue
            : defaultValue;
    }

    /// <summary>
    /// Get string value from environment variable, or default if not set
    /// </summary>
    private string GetEnvironmentValue(string envVar, string defaultValue)
    {
        var envValue = Environment.GetEnvironmentVariable(envVar);
        return !string.IsNullOrEmpty(envValue) ? envValue : defaultValue;
    }

    /// <summary>
    /// Save configuration to a JSON file
    /// </summary>
    public void Save(VisualizationConfiguration config, string filePath)
    {
        var json = SaveToJson(config);

        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Save configuration to a JSON string
    /// </summary>
    public string SaveToJson(VisualizationConfiguration config)
    {
        return JsonSerializer.Serialize(config, _jsonOptions);
    }
}
