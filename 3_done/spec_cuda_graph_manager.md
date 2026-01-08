# Spec: CUDA Graph Manager

## Overview
Implement a graph manager that handles caching, lifecycle management, and coordination of multiple CUDA graphs for different training phases (forward, backward, optimizer step).

## Requirements

### 1. CUDAGraphManager Class
Implement the central graph management system.

```csharp
public class CUDAGraphManager : IDisposable
{
    private readonly Dictionary<string, ICUDAGraph> _graphs;
    private readonly CUDAGraphMemoryPool _memoryPool;
    private readonly object _lock = new object();
    private readonly int _captureIterations;
    private int _currentIteration;
    private bool _disposed;

    public CUDAGraphManager(int captureIterations = 3, long initialMemoryPoolSize = 512 * 1024 * 1024)
    {
        _graphs = new Dictionary<string, ICUDAGraph>();
        _memoryPool = new CUDAGraphMemoryPool(initialMemoryPoolSize);
        _captureIterations = captureIterations;
        _currentIteration = 0;
        _disposed = false;
    }

    /// <summary>
    /// Gets a graph by name, creating it if it doesn't exist
    /// </summary>
    public ICUDAGraph GetOrCaptureGraph(string graphName, Action<CUDAStream> captureAction, CUDAStream stream)
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
                throw new InvalidOperationException(
                    $"Graph validation failed: {string.Join(", ", validation.Errors)}");
            }

            _graphs[graphName] = graph;
            return graph;
        }
    }

    /// <summary>
    /// Executes a graph by name, or executes the action if graph is not ready
    /// </summary>
    public void ExecuteGraphOrFallback(string graphName, Action<CUDAStream> captureAction, CUDAStream stream)
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
    /// Gets a specific graph by name
    /// </summary>
    public ICUDAGraph GetGraph(string graphName)
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            return _graphs.TryGetValue(graphName, out var graph) ? graph : null;
        }
    }

    /// <summary>
    /// Removes a graph from the cache
    /// </summary>
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
    /// Clears all cached graphs
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
    /// Gets the memory pool used by this manager
    /// </summary>
    public CUDAGraphMemoryPool MemoryPool => _memoryPool;

    /// <summary>
    /// Gets the number of cached graphs
    /// </summary>
    public int GraphCount => _graphs.Count;

    /// <summary>
    /// Gets whether the capture phase is complete
    /// </summary>
    public bool IsCaptureComplete => _currentIteration >= _captureIterations;

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
```

### 2. GraphPhase Enum
Define common training phases for graphs.

```csharp
public enum GraphPhase
{
    Forward,
    Backward,
    OptimizerStep,
    ForwardBackward,
    FullTrainingStep
}
```

### 3. CUDAGraphManager Extensions
Helper methods for common graph management patterns.

```csharp
public static class CUDAGraphManagerExtensions
{
    /// <summary>
    /// Gets or captures a graph for a specific phase
    /// </summary>
    public static ICUDAGraph GetOrCapturePhaseGraph(
        this CUDAGraphManager manager,
        GraphPhase phase,
        Action<CUDAStream> captureAction,
        CUDAStream stream)
    {
        return manager.GetOrCaptureGraph($"Phase_{phase}", captureAction, stream);
    }

    /// <summary>
    /// Executes a graph for a specific phase
    /// </summary>
    public static void ExecutePhaseGraph(
        this CUDAGraphManager manager,
        GraphPhase phase,
        Action<CUDAStream> captureAction,
        CUDAStream stream)
    {
        manager.ExecuteGraphOrFallback($"Phase_{phase}", captureAction, stream);
    }
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Graphs/CUDAGraphManager.cs`
- **File**: `src/CUDA/Graphs/GraphPhase.cs`
- **File**: `src/CUDA/Graphs/CUDAGraphManagerExtensions.cs`

### Dependencies
- ICUDAGraph interface (from spec_cuda_graph_core_interfaces)
- CUDAGraphMemoryPool (from spec_cuda_graph_memory_pool)
- CUDAGraphCapture (from spec_cuda_graph_capture_api)
- CUDAStream class (existing)
- System.Collections.Generic for Dictionary
- System for Action

### Graph Lifecycle
1. **Capture Phase**: Execute normally for N iterations to warm up
2. **Capture**: On iteration N, begin capture and execute
3. **Cache**: Store the captured graph for reuse
4. **Execute**: Reuse graph for all subsequent iterations

### Memory Management
- Manager owns the memory pool
- All graphs use the same memory pool
- Memory is allocated once and reused
- Dispose cleans up all graphs and memory

### Thread Safety
- Lock around all graph operations
- Thread-safe for concurrent access
- Multiple threads can request the same graph

## Success Criteria
- Manager can cache multiple graphs by name
- Capture phase executes actions normally
- Graph capture works correctly on the Nth iteration
- Captured graphs are reused for subsequent executions
- Memory pool is shared across all graphs
- Thread-safe for concurrent access
- Proper disposal of all resources

## Testing Requirements

### Unit Tests
- Test graph caching and retrieval
- Test capture phase execution (iterations 1 to N-1)
- Test graph capture on Nth iteration
- Test graph reuse after capture
- Test multiple graphs with different names
- Test graph removal and clearing
- Test fallback during capture phase
- Test concurrent access from multiple threads

### Integration Tests
- Test manager with forward/backward graphs
- Test manager with full training step graph
- Test memory pool sharing across graphs
- Test with actual model execution (requires GPU)
