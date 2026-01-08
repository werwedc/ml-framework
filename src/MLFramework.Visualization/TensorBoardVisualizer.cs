using MachineLearning.Visualization.Events;
using MachineLearning.Visualization.Storage;
using MachineLearning.Visualization.Scalars;
using MLFramework.Visualization.Configuration;
using MLFramework.Visualization.Graphs;
using MLFramework.Visualization.Histograms;
using MLFramework.Visualization.Profiling;

namespace MLFramework.Visualization;

/// <summary>
/// Main TensorBoardVisualizer API that provides a unified interface for all visualization and profiling functionality.
/// </summary>
public class TensorBoardVisualizer : IDisposable
{
    private readonly IStorageBackend _storage;
    private readonly IScalarLogger _scalarLogger;
    private readonly IEventPublisher _eventPublisher;
    private readonly Profiler _profiler;
    private readonly object _lock = new();
    private readonly Dictionary<string, long> _stepCounters;
    private bool _disposed;
    private long _currentStep;

    /// <summary>
    /// Gets the storage configuration
    /// </summary>
    public StorageConfiguration StorageConfig { get; set; }

    /// <summary>
    /// Gets or sets whether the visualizer is enabled
    /// </summary>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Gets whether asynchronous operations are enabled
    /// </summary>
    public bool EnableAsync { get; private set; }

    /// <summary>
    /// Gets the run name/tag for this visualization session
    /// </summary>
    public string RunName { get; }

    /// <summary>
    /// Gets additional metadata for this run
    /// </summary>
    public Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Creates a new TensorBoardVisualizer with a log directory
    /// </summary>
    /// <param name="logDirectory">Directory where logs will be stored</param>
    public TensorBoardVisualizer(string logDirectory)
        : this(new StorageConfiguration
        {
            BackendType = "file",
            LogDirectory = logDirectory,
            ConnectionString = logDirectory
        })
    {
    }

    /// <summary>
    /// Creates a new TensorBoardVisualizer with a storage configuration
    /// </summary>
    /// <param name="config">Storage configuration</param>
    public TensorBoardVisualizer(StorageConfiguration config)
        : this(CreateFileStorageBackend(config))
    {
    }

    /// <summary>
    /// Creates a new TensorBoardVisualizer with a custom storage backend
    /// </summary>
    /// <param name="storage">Custom storage backend</param>
    public TensorBoardVisualizer(IStorageBackend storage)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _eventPublisher = new MachineLearning.Visualization.Events.EventSystem();
        _scalarLogger = new ScalarLogger(_eventPublisher);
        _profiler = new Profiler(_eventPublisher, _storage);
        _stepCounters = new Dictionary<string, long>();
        StorageConfig = new StorageConfiguration
        {
            BackendType = "custom",
            LogDirectory = "",
            ConnectionString = ""
        };
        EnableAsync = true;
        RunName = "default";
        Metadata = new Dictionary<string, string>();

        _storage.Initialize(StorageConfig.ConnectionString);
    }

    /// <summary>
    /// Creates a new TensorBoardVisualizer with a complete configuration
    /// </summary>
    /// <param name="config">Visualizer configuration</param>
    public TensorBoardVisualizer(VisualizerConfiguration config)
    {
        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        config.Validate();
        StorageConfig = config.StorageConfig;
        EnableAsync = config.EnableAsync;
        IsEnabled = config.IsEnabled;
        RunName = config.RunName;
        Metadata = config.Metadata;

        _storage = CreateFileStorageBackend(StorageConfig);
        _eventPublisher = new MachineLearning.Visualization.Events.EventSystem();
        _scalarLogger = new ScalarLogger(_eventPublisher);
        _profiler = new Profiler(_eventPublisher, _storage);
        _stepCounters = new Dictionary<string, long>();

        _storage.Initialize(StorageConfig.ConnectionString);
    }

    // Scalar metrics

    /// <summary>
    /// Logs a scalar metric value
    /// </summary>
    /// <param name="name">Metric name (e.g., "train/loss")</param>
    /// <param name="value">Metric value</param>
    /// <param name="step">Training step number (auto-incremented if -1)</param>
    public void LogScalar(string name, float value, long step = -1)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        step = GetOrIncrementStep(name, step);
        _scalarLogger.LogScalar(name, value, step);
    }

    /// <summary>
    /// Logs a scalar metric value (double overload)
    /// </summary>
    /// <param name="name">Metric name (e.g., "train/loss")</param>
    /// <param name="value">Metric value</param>
    /// <param name="step">Training step number (auto-incremented if -1)</param>
    public void LogScalar(string name, double value, long step = -1)
    {
        LogScalar(name, (float)value, step);
    }

    /// <summary>
    /// Logs a scalar metric value asynchronously
    /// </summary>
    /// <param name="name">Metric name (e.g., "train/loss")</param>
    /// <param name="value">Metric value</param>
    /// <param name="step">Training step number (auto-incremented if -1)</param>
    public async Task LogScalarAsync(string name, float value, long step = -1)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (EnableAsync)
        {
            await Task.Run(() => LogScalar(name, value, step)).ConfigureAwait(false);
        }
        else
        {
            LogScalar(name, value, step);
        }
    }

    // Histograms

    /// <summary>
    /// Logs a histogram
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of histogram values</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    public void LogHistogram(string name, float[] values, long step = -1)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        if (values == null)
        {
            throw new ArgumentNullException(nameof(values));
        }

        step = GetOrIncrementStep(name, step);
        var histogramEvent = new HistogramEvent(name, values, step);
        _eventPublisher.Publish(histogramEvent);
        _storage.StoreEvent(histogramEvent);
    }

    /// <summary>
    /// Logs a histogram with custom binning configuration
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of histogram values</param>
    /// <param name="config">Binning configuration</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    public void LogHistogram(string name, float[] values, HistogramBinConfig config, long step = -1)
    {
        if (!IsEnabled)
        {
            return;
        }

        config?.Validate();
        step = GetOrIncrementStep(name, step);

        var min = config?.Min ?? (values.Length > 0 ? values.Min() : 0f);
        var max = config?.Max ?? (values.Length > 0 ? values.Max() : 0f);
        var binCount = config?.BinCount ?? 30;
        var useLogScale = config?.UseLogScale ?? false;

        var histogramEvent = new HistogramEvent(
            name,
            values,
            step,
            binCount,
            useLogScale,
            min,
            max);

        _eventPublisher.Publish(histogramEvent);
        _storage.StoreEvent(histogramEvent);
    }

    /// <summary>
    /// Logs a histogram asynchronously
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of histogram values</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    public async Task LogHistogramAsync(string name, float[] values, long step = -1)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (EnableAsync)
        {
            await Task.Run(() => LogHistogram(name, values, step)).ConfigureAwait(false);
        }
        else
        {
            LogHistogram(name, values, step);
        }
    }

    // Computational graph

    /// <summary>
    /// Logs a computational graph
    /// </summary>
    /// <param name="model">Model to visualize</param>
    public void LogGraph(IModel model)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        // Placeholder for extracting graph from model
        // In a real implementation, this would traverse the model structure
        var graph = new ComputationalGraph(model.Name);
        var graphEvent = new ComputationalGraphEvent(graph.Name, _currentStep);
        _eventPublisher.Publish(graphEvent);
        _storage.StoreEvent(graphEvent);
    }

    /// <summary>
    /// Logs a computational graph
    /// </summary>
    /// <param name="graph">Computational graph to log</param>
    public void LogGraph(ComputationalGraph graph)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (graph == null)
        {
            throw new ArgumentNullException(nameof(graph));
        }

        var graphEvent = new ComputationalGraphEvent(graph.Name, graph.Step);
        _eventPublisher.Publish(graphEvent);
        _storage.StoreEvent(graphEvent);
    }

    /// <summary>
    /// Logs a computational graph asynchronously
    /// </summary>
    /// <param name="model">Model to visualize</param>
    public async Task LogGraphAsync(IModel model)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (EnableAsync)
        {
            await Task.Run(() => LogGraph(model)).ConfigureAwait(false);
        }
        else
        {
            LogGraph(model);
        }
    }

    // Profiling

    /// <summary>
    /// Starts profiling a section of code
    /// </summary>
    /// <param name="name">Name of the section being profiled</param>
    /// <returns>A profiling scope that records duration when disposed</returns>
    public IProfilingScope StartProfile(string name)
    {
        if (!IsEnabled)
        {
            return new NullProfilingScope();
        }

        return _profiler.StartProfile(name);
    }

    /// <summary>
    /// Starts profiling a section of code with metadata
    /// </summary>
    /// <param name="name">Name of the section being profiled</param>
    /// <param name="metadata">Additional metadata</param>
    /// <returns>A profiling scope that records duration when disposed</returns>
    public IProfilingScope StartProfile(string name, Dictionary<string, string> metadata)
    {
        if (!IsEnabled)
        {
            return new NullProfilingScope();
        }

        return _profiler.StartProfile(name, metadata);
    }

    /// <summary>
    /// Records an instant event (e.g., a checkpoint or milestone)
    /// </summary>
    /// <param name="name">Name of the instant event</param>
    public void RecordInstant(string name)
    {
        if (!IsEnabled)
        {
            return;
        }

        _profiler.RecordInstant(name);
    }

    // Advanced features

    /// <summary>
    /// Logs hyperparameters
    /// </summary>
    /// <param name="hyperparams">Dictionary of hyperparameters</param>
    public void LogHyperparameters(Dictionary<string, object> hyperparams)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (hyperparams == null)
        {
            throw new ArgumentNullException(nameof(hyperparams));
        }

        // Convert hyperparameters to string metadata
        var metadata = new Dictionary<string, string>();
        foreach (var kvp in hyperparams)
        {
            metadata[kvp.Key] = kvp.Value?.ToString() ?? "null";
        }

        // Store as metadata with the current run
        foreach (var kvp in metadata)
        {
            Metadata[kvp.Key] = kvp.Value;
        }
    }

    /// <summary>
    /// Logs text data (e.g., debug notes)
    /// </summary>
    /// <param name="name">Name of the text</param>
    /// <param name="text">Text content</param>
    public void LogText(string name, string text)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        // Store text as metadata for now
        Metadata[$"text_{name}"] = text ?? string.Empty;
    }

    /// <summary>
    /// Logs an image
    /// </summary>
    /// <param name="name">Name of the image</param>
    /// <param name="imageData">Image data as byte array</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    public void LogImage(string name, byte[] imageData, long step = -1)
    {
        if (!IsEnabled)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        if (imageData == null)
        {
            throw new ArgumentNullException(nameof(imageData));
        }

        step = GetOrIncrementStep(name, step);

        // Placeholder for image logging
        // In a real implementation, this would encode the image and create an ImageEvent
        Metadata[$"image_{name}_step_{step}"] = $"Image data length: {imageData.Length}";
    }

    // Export and cleanup

    /// <summary>
    /// Exports all data (closes files and flushes buffers)
    /// </summary>
    public void Export()
    {
        if (_disposed)
        {
            return;
        }

        Flush();
        _storage.Shutdown();
    }

    /// <summary>
    /// Exports all data asynchronously
    /// </summary>
    public async Task ExportAsync()
    {
        if (_disposed)
        {
            return;
        }

        await FlushAsync().ConfigureAwait(false);
        await Task.Run(() => _storage.Shutdown()).ConfigureAwait(false);
    }

    /// <summary>
    /// Flushes any pending data to storage
    /// </summary>
    public void Flush()
    {
        if (_disposed || !IsEnabled)
        {
            return;
        }

        lock (_lock)
        {
            _storage.Flush();
        }
    }

    /// <summary>
    /// Flushes any pending data to storage asynchronously
    /// </summary>
    public async Task FlushAsync()
    {
        if (_disposed || !IsEnabled)
        {
            return;
        }

        if (EnableAsync)
        {
            await Task.Run(Flush).ConfigureAwait(false);
        }
        else
        {
            Flush();
        }
    }

    /// <summary>
    /// Disposes of resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            Flush();
            _storage.Dispose();
            _disposed = true;
        }
    }

    // Helper methods

    private static IStorageBackend CreateFileStorageBackend(StorageConfiguration config)
    {
        var mlConfig = new MachineLearning.Visualization.Storage.StorageConfiguration
        {
            BackendType = config.BackendType,
            ConnectionString = config.ConnectionString,
            WriteBufferSize = config.WriteBufferSize,
            FlushInterval = config.FlushInterval,
            EnableAsyncWrites = config.EnableAsyncWrites
        };

        return new FileStorageBackend(mlConfig);
    }

    private long GetOrIncrementStep(string name, long step)
    {
        lock (_lock)
        {
            if (step == -1)
            {
                if (!_stepCounters.TryGetValue(name, out step))
                {
                    step = 0;
                }
                _stepCounters[name] = step + 1;
            }
            else
            {
                _stepCounters[name] = step + 1;
            }

            _currentStep = Math.Max(_currentStep, step);
            return step;
        }
    }

    /// <summary>
    /// Null profiling scope for when visualizer is disabled
    /// </summary>
    private class NullProfilingScope : IProfilingScope
    {
        public string Name => "null";
        public Dictionary<string, string> Metadata => new();

        public void End()
        {
            // No-op
        }

        public void Dispose()
        {
            End();
        }
    }

    /// <summary>
    /// Internal profiler implementation
    /// </summary>
    private class Profiler : IProfiler
    {
        private readonly IEventPublisher _eventPublisher;
        private readonly IStorageBackend _storage;
        private readonly Dictionary<string, DateTime> _profileStartTimes;
        private readonly object _lock = new();

        public bool IsEnabled { get; set; } = true;

        public Profiler(IEventPublisher eventPublisher, IStorageBackend storage)
        {
            _eventPublisher = eventPublisher;
            _storage = storage;
            _profileStartTimes = new Dictionary<string, DateTime>();
        }

        public IProfilingScope StartProfile(string name)
        {
            return StartProfile(name, new Dictionary<string, string>());
        }

        public IProfilingScope StartProfile(string name, Dictionary<string, string> metadata)
        {
            if (!IsEnabled)
            {
                return new NullProfilingScope();
            }

            var startEvent = new ProfilingStartEvent(name, -1, metadata);
            _eventPublisher.Publish(startEvent);
            _storage.StoreEvent(startEvent);

            lock (_lock)
            {
                _profileStartTimes[name] = DateTime.UtcNow;
            }

            return new ProfilingScopeImpl(this, name, metadata);
        }

        public void RecordInstant(string name)
        {
            RecordInstant(name, new Dictionary<string, string>());
        }

        public void RecordInstant(string name, Dictionary<string, string> metadata)
        {
            if (!IsEnabled)
            {
                return;
            }

            // For instant events, we just record the start and end at the same time
            var startEvent = new ProfilingStartEvent(name, -1, metadata);
            var endEvent = new ProfilingEndEvent(name, -1, 0, metadata);

            _eventPublisher.Publish(startEvent);
            _eventPublisher.Publish(endEvent);
            _storage.StoreEvent(startEvent);
            _storage.StoreEvent(endEvent);
        }

        internal void EndProfile(string name, Dictionary<string, string> metadata)
        {
            DateTime startTime;
            lock (_lock)
            {
                if (!_profileStartTimes.TryGetValue(name, out startTime))
                {
                    return;
                }
                _profileStartTimes.Remove(name);
            }

            var duration = DateTime.UtcNow - startTime;
            var durationNanoseconds = (long)(duration.TotalNanoseconds);
            var endEvent = new ProfilingEndEvent(name, -1, durationNanoseconds, metadata);

            _eventPublisher.Publish(endEvent);
            _storage.StoreEvent(endEvent);
        }
    }

    /// <summary>
    /// Internal profiling scope implementation
    /// </summary>
    private class ProfilingScopeImpl : IProfilingScope
    {
        private readonly Profiler _profiler;
        private bool _ended;

        public string Name { get; }
        public Dictionary<string, string> Metadata { get; }

        public ProfilingScopeImpl(Profiler profiler, string name, Dictionary<string, string> metadata)
        {
            _profiler = profiler;
            Name = name;
            Metadata = metadata;
            _ended = false;
        }

        public void End()
        {
            if (!_ended)
            {
                _profiler.EndProfile(Name, Metadata);
                _ended = true;
            }
        }

        public void Dispose()
        {
            End();
        }
    }
}
