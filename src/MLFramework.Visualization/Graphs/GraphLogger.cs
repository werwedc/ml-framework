using MachineLearning.Visualization.Storage;
using MLFramework.Visualization.Events;
using System.Collections.Concurrent;

namespace MLFramework.Visualization.Graphs;

/// <summary>
/// Implementation of IGraphLogger for logging computational graphs.
/// </summary>
public class GraphLogger : IGraphLogger, IDisposable
{
    private readonly IStorageBackend _storage;
    private readonly GraphAnalyzer _analyzer;
    private readonly ConcurrentDictionary<string, ComputationalGraph> _loggedGraphs;

    /// <summary>
    /// Configuration for automatically capturing dynamic graphs.
    /// </summary>
    public bool AutoCaptureDynamicGraphs { get; set; } = false;

    /// <summary>
    /// Maximum depth for graph capture to avoid infinite loops.
    /// </summary>
    public int MaxCaptureDepth { get; set; } = 100;

    private ComputationalGraph? _capturedGraph;
    private int _captureDepth;
    private bool _isCapturing;
    private bool _disposed;

    /// <summary>
    /// Creates a new GraphLogger with a storage backend.
    /// </summary>
    /// <param name="storage">Storage backend for persisting graphs.</param>
    public GraphLogger(IStorageBackend storage)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _analyzer = new GraphAnalyzer();
        _loggedGraphs = new ConcurrentDictionary<string, ComputationalGraph>();
    }

    /// <summary>
    /// Logs a computational graph synchronously.
    /// </summary>
    /// <param name="graph">The graph to log.</param>
    public void LogGraph(ComputationalGraph graph)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        // Store the graph in memory
        var graphId = $"{graph.Name}_{graph.Step}_{graph.Timestamp.Ticks}";
        _loggedGraphs.TryAdd(graphId, graph);

        // Create and store the event
        var graphEvent = new ComputationalGraphEvent(graph);
        _storage.StoreAsync(graphEvent).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Logs a model by extracting its computational graph synchronously.
    /// </summary>
    /// <param name="model">The model to log.</param>
    public void LogGraph(IModel model)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));

        var graph = model.GetGraph();
        LogGraph(graph);
    }

    /// <summary>
    /// Logs a computational graph asynchronously.
    /// </summary>
    /// <param name="graph">The graph to log.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    public async Task LogGraphAsync(ComputationalGraph graph)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        // Store the graph in memory
        var graphId = $"{graph.Name}_{graph.Step}_{graph.Timestamp.Ticks}";
        _loggedGraphs.TryAdd(graphId, graph);

        // Create and store the event
        var graphEvent = new ComputationalGraphEvent(graph);
        await _storage.StoreAsync(graphEvent);
    }

    /// <summary>
    /// Logs a model by extracting its computational graph asynchronously.
    /// </summary>
    /// <param name="model">The model to log.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    public async Task LogGraphAsync(IModel model)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));

        var graph = model.GetGraph();
        await LogGraphAsync(graph);
    }

    /// <summary>
    /// Starts capturing a dynamic graph.
    /// </summary>
    /// <param name="name">Name for the captured graph.</param>
    public void StartGraphCapture(string name)
    {
        if (string.IsNullOrEmpty(name))
            throw new ArgumentException("Name cannot be null or empty.", nameof(name));

        if (_isCapturing)
        {
            throw new InvalidOperationException("Graph capture is already in progress.");
        }

        _capturedGraph = new ComputationalGraph(name);
        _captureDepth = 0;
        _isCapturing = true;
    }

    /// <summary>
    /// Stops capturing the dynamic graph.
    /// </summary>
    public void StopGraphCapture()
    {
        if (!_isCapturing)
        {
            throw new InvalidOperationException("No graph capture is in progress.");
        }

        _isCapturing = false;
    }

    /// <summary>
    /// Gets the captured dynamic graph.
    /// </summary>
    /// <returns>The captured graph, or null if no capture is in progress or completed.</returns>
    public ComputationalGraph? GetCapturedGraph()
    {
        return _capturedGraph;
    }

    /// <summary>
    /// Analyzes a logged graph and returns analysis results.
    /// </summary>
    /// <param name="graphId">ID of the graph to analyze.</param>
    /// <returns>Graph analysis results.</returns>
    public GraphAnalysis AnalyzeGraph(string graphId)
    {
        if (string.IsNullOrEmpty(graphId))
            throw new ArgumentException("Graph ID cannot be null or empty.", nameof(graphId));

        if (!_loggedGraphs.TryGetValue(graphId, out var graph))
        {
            throw new ArgumentException($"Graph with ID '{graphId}' not found.", nameof(graphId));
        }

        return _analyzer.Analyze(graph);
    }

    /// <summary>
    /// Analyzes all logged graphs and returns analysis results.
    /// </summary>
    /// <returns>Dictionary of graph IDs to analysis results.</returns>
    public Dictionary<string, GraphAnalysis> AnalyzeAllGraphs()
    {
        var results = new Dictionary<string, GraphAnalysis>();

        foreach (var kvp in _loggedGraphs)
        {
            results[kvp.Key] = _analyzer.Analyze(kvp.Value);
        }

        return results;
    }

    /// <summary>
    /// Gets all logged graphs.
    /// </summary>
    /// <returns>Dictionary of graph IDs to graphs.</returns>
    public IReadOnlyDictionary<string, ComputationalGraph> GetAllGraphs()
    {
        return _loggedGraphs;
    }

    /// <summary>
    /// Clears all logged graphs.
    /// </summary>
    public void ClearGraphs()
    {
        _loggedGraphs.Clear();
    }

    /// <summary>
    /// Disposes of resources used by the GraphLogger.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _storage?.Dispose();
            _loggedGraphs.Clear();
            _disposed = true;
        }
    }
}
