using System;
using System.Collections.Generic;
using MLFramework.HAL.CUDA.Graphs;

namespace MLFramework.HAL.CUDA.Graphs;

/// <summary>
/// Manages CUDA graph caching, lifecycle management, and coordination
/// for different training phases (forward, backward, optimizer step).
/// </summary>
public class CUDAGraphManager : IDisposable
{
    private readonly Dictionary<string, ICUDAGraph> _graphs;
    private readonly CUDAGraphMemoryPool _memoryPool;
    private readonly object _lock = new object();
    private readonly int _captureIterations;
    private int _currentIteration;
    private bool _disposed;

    /// <summary>
    /// Gets the number of cached graphs
    /// </summary>
    public int GraphCount => _graphs.Count;

    /// <summary>
    /// Gets the memory pool used by this manager
    /// </summary>
    public CUDAGraphMemoryPool MemoryPool => _memoryPool;

    /// <summary>
    /// Gets whether the capture phase is complete
    /// </summary>
    public bool IsCaptureComplete => _currentIteration >= _captureIterations;

    /// <summary>
    /// Initializes a new instance of the CUDAGraphManager class.
    /// </summary>
    /// <param name="captureIterations">Number of iterations before capturing (default: 3)</param>
    /// <param name="initialMemoryPoolSize">Initial memory pool size in bytes (default: 512MB)</param>
    public CUDAGraphManager(int captureIterations = 3, long initialMemoryPoolSize = 512 * 1024 * 1024)
    {
        _graphs = new Dictionary<string, ICUDAGraph>();
        _memoryPool = new CUDAGraphMemoryPool(initialMemoryPoolSize);
        _captureIterations = captureIterations;
        _currentIteration = 0;
        _disposed = false;
    }

    /// <summary>
    /// Gets a graph by name, creating it if it doesn't exist.
    /// </summary>
    /// <param name="graphName">Name of the graph to get or capture</param>
    /// <param name="captureAction">Action to capture for the graph</param>
    /// <param name="stream">CUDA stream for capture/execution</param>
    /// <returns>The captured graph, or null if still in warm-up phase</returns>
    public ICUDAGraph GetOrCaptureGraph(string graphName, Action<CudaStream> captureAction, CudaStream stream)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            // Check if graph already exists
            if (_graphs.TryGetValue(graphName, out var graph))
            {
                return graph;
            }

            // Check if we're still in capture phase
            if (_currentIteration < _captureIterations)
            {
                // Execute without graph to warm up
                captureAction(stream);
                _currentIteration++;

                // Return null to indicate graph not ready yet
                return null;
            }

            // Capture the graph
            using var capture = new CUDAGraphCapture();
            capture.BeginCapture(stream);

            // Execute the capture action
            captureAction(stream);

            // End capture and store the graph
            graph = capture.EndCapture();

            // Validate the graph
            var validation = graph.Validate();
            if (!validation.IsValid)
            {
                graph.Dispose();
                throw new InvalidOperationException(
                    $"Graph validation failed: {string.Join(", ", validation.Errors)}");
            }

            _graphs[graphName] = graph;
            return graph;
        }
    }

    /// <summary>
    /// Executes a graph by name, or executes the action if graph is not ready.
    /// </summary>
    /// <param name="graphName">Name of the graph to execute</param>
    /// <param name="captureAction">Action to execute if graph is not ready</param>
    /// <param name="stream">CUDA stream for execution</param>
    public void ExecuteGraphOrFallback(string graphName, Action<CudaStream> captureAction, CudaStream stream)
    {
        ThrowIfDisposed();

        var graph = GetOrCaptureGraph(graphName, captureAction, stream);

        if (graph != null)
        {
            // Execute the captured graph
            graph.Execute(stream);
        }
        else
        {
            // Fall back to regular execution during capture phase
            captureAction(stream);
        }
    }

    /// <summary>
    /// Gets a specific graph by name.
    /// </summary>
    /// <param name="graphName">Name of the graph to get</param>
    /// <returns>The graph if found, otherwise null</returns>
    public ICUDAGraph GetGraph(string graphName)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            return _graphs.TryGetValue(graphName, out var graph) ? graph : null;
        }
    }

    /// <summary>
    /// Removes a graph from the cache.
    /// </summary>
    /// <param name="graphName">Name of the graph to remove</param>
    public void RemoveGraph(string graphName)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            if (_graphs.TryGetValue(graphName, out var graph))
            {
                graph.Dispose();
                _graphs.Remove(graphName);
            }
        }
    }

    /// <summary>
    /// Clears all cached graphs.
    /// </summary>
    public void ClearGraphs()
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            foreach (var graph in _graphs.Values)
            {
                graph.Dispose();
            }
            _graphs.Clear();
        }
    }

    /// <summary>
    /// Disposes the manager and all associated resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
            return;

        lock (_lock)
        {
            ClearGraphs();
            _memoryPool.Dispose();
            _disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CUDAGraphManager));
    }
}
