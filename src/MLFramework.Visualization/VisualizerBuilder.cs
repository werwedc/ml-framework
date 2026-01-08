using MachineLearning.Visualization.Scalars;
using MLFramework.Visualization.Configuration;
using MLFramework.Visualization.Histograms;
using MLFramework.Visualization.Profiling;
using MachineLearning.Visualization.Storage;

namespace MLFramework.Visualization;

/// <summary>
/// Fluent builder pattern for creating TensorBoardVisualizer instances
/// </summary>
public class VisualizerBuilder
{
    private string _logDirectory = "./logs";
    private IStorageBackend? _storage;
    private StorageConfiguration? _storageConfig;
    private IScalarLogger? _scalarLogger;
    private IHistogramLogger? _histogramLogger;
    private IProfiler? _profiler;
    private bool _enableAsync = true;
    private bool _enableProfiling = true;
    private bool _isEnabled = true;
    private string _runName = "default";
    private Dictionary<string, string> _metadata = new();

    /// <summary>
    /// Creates a new VisualizerBuilder instance
    /// </summary>
    public static VisualizerBuilder Create() => new VisualizerBuilder();

    /// <summary>
    /// Private constructor to enforce factory pattern
    /// </summary>
    private VisualizerBuilder()
    {
    }

    /// <summary>
    /// Sets the log directory for the visualizer
    /// </summary>
    /// <param name="directory">Directory where logs will be stored</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder WithLogDirectory(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new ArgumentException("Log directory cannot be null or empty", nameof(directory));
        }

        _logDirectory = directory;
        _storage = null; // Clear any custom storage backend
        return this;
    }

    /// <summary>
    /// Sets a custom storage backend for the visualizer
    /// </summary>
    /// <param name="backend">Custom storage backend</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder WithStorageBackend(IStorageBackend backend)
    {
        _storage = backend ?? throw new ArgumentNullException(nameof(backend));
        _logDirectory = string.Empty; // Clear log directory
        return this;
    }

    /// <summary>
    /// Sets a custom storage configuration for the visualizer
    /// </summary>
    /// <param name="config">Storage configuration</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder WithStorageConfig(StorageConfiguration config)
    {
        _storageConfig = config ?? throw new ArgumentNullException(nameof(config));
        _storage = null; // Clear any custom storage backend
        _logDirectory = string.Empty;
        return this;
    }

    /// <summary>
    /// Sets a custom scalar logger for the visualizer
    /// </summary>
    /// <param name="logger">Custom scalar logger</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder WithScalarLogger(IScalarLogger logger)
    {
        _scalarLogger = logger;
        return this;
    }

    /// <summary>
    /// Sets a custom histogram logger for the visualizer
    /// </summary>
    /// <param name="logger">Custom histogram logger</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder WithHistogramLogger(IHistogramLogger logger)
    {
        _histogramLogger = logger;
        return this;
    }

    /// <summary>
    /// Sets a custom profiler for the visualizer
    /// </summary>
    /// <param name="profiler">Custom profiler</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder WithProfiler(IProfiler profiler)
    {
        _profiler = profiler;
        return this;
    }

    /// <summary>
    /// Enables or disables asynchronous operations
    /// </summary>
    /// <param name="enable">True to enable async, false to disable</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder EnableAsync(bool enable = true)
    {
        _enableAsync = enable;
        return this;
    }

    /// <summary>
    /// Enables or disables profiling
    /// </summary>
    /// <param name="enable">True to enable profiling, false to disable</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder EnableProfiling(bool enable = true)
    {
        _enableProfiling = enable;
        return this;
    }

    /// <summary>
    /// Enables or disables the visualizer
    /// </summary>
    /// <param name="enable">True to enable, false to disable</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder Enable(bool enable = true)
    {
        _isEnabled = enable;
        return this;
    }

    /// <summary>
    /// Sets the run name/tag for this visualization session
    /// </summary>
    /// <param name="runName">Run name</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder WithRunName(string runName)
    {
        if (string.IsNullOrWhiteSpace(runName))
        {
            throw new ArgumentException("Run name cannot be null or empty", nameof(runName));
        }

        _runName = runName;
        return this;
    }

    /// <summary>
    /// Adds metadata to the visualization session
    /// </summary>
    /// <param name="key">Metadata key</param>
    /// <param name="value">Metadata value</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder WithMetadata(string key, string value)
    {
        if (string.IsNullOrWhiteSpace(key))
        {
            throw new ArgumentException("Metadata key cannot be null or empty", nameof(key));
        }

        _metadata[key] = value ?? string.Empty;
        return this;
    }

    /// <summary>
    /// Adds multiple metadata entries to the visualization session
    /// </summary>
    /// <param name="metadata">Dictionary of metadata</param>
    /// <returns>This builder instance for fluent chaining</returns>
    public VisualizerBuilder WithMetadata(Dictionary<string, string> metadata)
    {
        if (metadata == null)
        {
            throw new ArgumentNullException(nameof(metadata));
        }

        foreach (var kvp in metadata)
        {
            _metadata[kvp.Key] = kvp.Value;
        }

        return this;
    }

    /// <summary>
    /// Builds the TensorBoardVisualizer instance with the configured options
    /// </summary>
    /// <returns>A new TensorBoardVisualizer instance</returns>
    public TensorBoardVisualizer Build()
    {
        // Determine storage backend
        IStorageBackend? storage;
        StorageConfiguration storageConfig;

        if (_storage != null)
        {
            storage = _storage;
            storageConfig = new StorageConfiguration
            {
                BackendType = "custom",
                LogDirectory = "",
                ConnectionString = "",
                WriteBufferSize = 100,
                FlushInterval = TimeSpan.FromSeconds(1),
                EnableAsyncWrites = _enableAsync
            };
        }
        else if (_storageConfig != null)
        {
            storageConfig = _storageConfig;
            var mlConfig = new MachineLearning.Visualization.Storage.StorageConfiguration
            {
                BackendType = storageConfig.BackendType,
                ConnectionString = storageConfig.ConnectionString,
                WriteBufferSize = storageConfig.WriteBufferSize,
                FlushInterval = storageConfig.FlushInterval,
                EnableAsyncWrites = storageConfig.EnableAsyncWrites
            };
            storage = new FileStorageBackend(mlConfig);
        }
        else
        {
            // Use default file-based storage with log directory
            storageConfig = new StorageConfiguration
            {
                BackendType = "file",
                LogDirectory = _logDirectory,
                ConnectionString = _logDirectory,
                WriteBufferSize = 100,
                FlushInterval = TimeSpan.FromSeconds(1),
                EnableAsyncWrites = _enableAsync
            };
            var mlConfig = new MachineLearning.Visualization.Storage.StorageConfiguration
            {
                BackendType = storageConfig.BackendType,
                ConnectionString = storageConfig.ConnectionString,
                WriteBufferSize = storageConfig.WriteBufferSize,
                FlushInterval = storageConfig.FlushInterval,
                EnableAsyncWrites = storageConfig.EnableAsyncWrites
            };
            storage = new FileStorageBackend(mlConfig);
        }

        // Validate configuration
        ValidateConfiguration();

        // Create visualizer configuration
        var config = new VisualizerConfiguration
        {
            StorageConfig = storageConfig,
            EnableAsync = _enableAsync,
            EnableProfiling = _enableProfiling,
            IsEnabled = _isEnabled,
            RunName = _runName,
            Metadata = new Dictionary<string, string>(_metadata)
        };

        // Create visualizer
        var visualizer = new TensorBoardVisualizer(config);

        // Note: Custom loggers and profiler are not currently supported in the TensorBoardVisualizer constructor
        // This is a placeholder for future enhancement
        if (_scalarLogger != null)
        {
            // TODO: Support custom scalar logger
            // For now, this is a no-op but we could add a method to replace the internal logger
        }

        if (_histogramLogger != null)
        {
            // TODO: Support custom histogram logger
        }

        if (_profiler != null)
        {
            // TODO: Support custom profiler
        }

        return visualizer;
    }

    /// <summary>
    /// Validates the builder configuration before creating the visualizer
    /// </summary>
    private void ValidateConfiguration()
    {
        // Ensure at least one storage option is specified
        if (_storage == null && _storageConfig == null && string.IsNullOrWhiteSpace(_logDirectory))
        {
            throw new InvalidOperationException(
                "Either a storage backend, storage configuration, or log directory must be specified. " +
                "Use WithLogDirectory(), WithStorageBackend(), or WithStorageConfig() to configure storage.");
        }

        // Validate storage configuration if provided
        if (_storageConfig != null)
        {
            if (string.IsNullOrWhiteSpace(_storageConfig.LogDirectory) && string.IsNullOrWhiteSpace(_storageConfig.ConnectionString))
            {
                throw new InvalidOperationException(
                    "Storage configuration must specify either LogDirectory or ConnectionString.");
            }
        }

        // Validate run name if provided
        if (string.IsNullOrWhiteSpace(_runName))
        {
            throw new InvalidOperationException("Run name cannot be empty.");
        }
    }
}
