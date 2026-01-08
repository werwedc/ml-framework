# Spec: Explicit Capture API

## Overview
Implement a static API for explicit CUDA graph capture. This provides a more imperative approach to graph capture, giving users fine-grained control over when and how graphs are captured and executed.

## Requirements

### 1. CUDAGraph Static Class
Provide static methods for graph capture and execution.

```csharp
public static class CUDAGraph
{
    /// <summary>
    /// Captures a CUDA graph from the specified action
    /// </summary>
    /// <param name="captureAction">Action to capture</param>
    /// <param name="stream">CUDA stream for capture</param>
    /// <returns>Captured graph</returns>
    public static ICUDAGraph Capture(Action<CUDAStream> captureAction, CUDAStream stream)
    {
        using var capture = new CUDAGraphCapture();
        capture.BeginCapture(stream);

        captureAction(stream);

        return capture.EndCapture();
    }

    /// <summary>
    /// Captures a CUDA graph with a warm-up phase
    /// </summary>
    /// <param name="captureAction">Action to capture</param>
    /// <param name="stream">CUDA stream for capture</param>
    /// <param name="warmupIterations">Number of warm-up iterations</param>
    /// <returns>Captured graph</returns>
    public static ICUDAGraph CaptureWithWarmup(
        Action<CUDAStream> captureAction,
        CUDAStream stream,
        int warmupIterations = 3)
    {
        // Warm-up iterations
        for (int i = 0; i < warmupIterations; i++)
        {
            captureAction(stream);
        }

        // Capture
        return Capture(captureAction, stream);
    }

    /// <summary>
    /// Captures a CUDA graph with options
    /// </summary>
    /// <param name="captureAction">Action to capture</param>
    /// <param name="stream">CUDA stream for capture</param>
    /// <param name="options">Capture options</param>
    /// <returns>Captured graph</returns>
    public static ICUDAGraph Capture(
        Action<CUDAStream> captureAction,
        CUDAStream stream,
        CUDAGraphCaptureOptions options)
    {
        if (options == null)
            throw new ArgumentNullException(nameof(options));

        // Apply warm-up if specified
        if (options.WarmupIterations > 0)
        {
            for (int i = 0; i < options.WarmupIterations; i++)
            {
                captureAction(stream);
            }
        }

        // Capture
        var graph = Capture(captureAction, stream);

        // Validate if requested
        if (options.ValidateOnCapture)
        {
            var validation = graph.Validate();
            if (!validation.IsValid)
            {
                throw new InvalidOperationException(
                    $"Graph validation failed: {string.Join(", ", validation.Errors)}");
            }
        }

        return graph;
    }

    /// <summary>
    /// Captures and immediately executes a graph
    /// </summary>
    /// <param name="captureAction">Action to capture</param>
    /// <param name="stream">CUDA stream for capture and execution</param>
    /// <returns>Captured graph</returns>
    public static ICUDAGraph CaptureAndExecute(
        Action<CUDAStream> captureAction,
        CUDAStream stream)
    {
        var graph = Capture(captureAction, stream);
        graph.Execute(stream);
        return graph;
    }
}
```

### 2. CUDAGraphCaptureOptions Class
Options for graph capture.

```csharp
public class CUDAGraphCaptureOptions
{
    /// <summary>
    /// Gets or sets the number of warm-up iterations
    /// </summary>
    public int WarmupIterations { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to validate the graph after capture
    /// </summary>
    public bool ValidateOnCapture { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable weight updates
    /// </summary>
    public bool EnableWeightUpdates { get; set; } = false;

    /// <summary>
    /// Gets or sets the memory pool to use
    /// </summary>
    public CUDAGraphMemoryPool MemoryPool { get; set; } = null;

    /// <summary>
    /// Gets or sets the capture mode
    /// </summary>
    public CUDACaptureMode CaptureMode { get; set; } = CUDACaptureMode.CaptureModeThreadLocal;

    public static CUDAGraphCaptureOptions Default => new CUDAGraphCaptureOptions();

    public CUDAGraphCaptureOptions WithWarmup(int iterations)
    {
        WarmupIterations = iterations;
        return this;
    }

    public CUDAGraphCaptureOptions WithValidation(bool validate = true)
    {
        ValidateOnCapture = validate;
        return this;
    }

    public CUDAGraphCaptureOptions WithWeightUpdates(bool enable = true)
    {
        EnableWeightUpdates = enable;
        return this;
    }

    public CUDAGraphCaptureOptions WithMemoryPool(CUDAGraphMemoryPool pool)
    {
        MemoryPool = pool;
        return this;
    }
}
```

### 3. Extension Methods for Graph Operations
Provide convenient extension methods for graph operations.

```csharp
public static class CUDAGraphExtensions
{
    /// <summary>
    /// Executes a graph multiple times
    /// </summary>
    public static void ExecuteMultiple(this ICUDAGraph graph, CUDAStream stream, int iterations)
    {
        for (int i = 0; i < iterations; i++)
        {
            graph.Execute(stream);
        }
    }

    /// <summary>
    /// Measures execution time of a graph
    /// </summary>
    public static TimeSpan MeasureExecutionTime(
        this ICUDAGraph graph,
        CUDAStream stream,
        int iterations = 10)
    {
        var stopwatch = Stopwatch.StartNew();

        for (int i = 0; i < iterations; i++)
        {
            graph.Execute(stream);
        }

        stream.Synchronize();

        stopwatch.Stop();
        return stopwatch.Elapsed;
    }

    /// <summary>
    /// Gets the average execution time per iteration
    /// </summary>
    public static double GetAverageExecutionTimeMs(
        this ICUDAGraph graph,
        CUDAStream stream,
        int iterations = 10)
    {
        var totalTime = MeasureExecutionTime(graph, stream, iterations);
        return totalTime.TotalMilliseconds / iterations;
    }

    /// <summary>
    /// Executes a graph with a callback before and after
    /// </summary>
    public static void ExecuteWithCallbacks(
        this ICUDAGraph graph,
        CUDAStream stream,
        Action before = null,
        Action after = null)
    {
        before?.Invoke();
        graph.Execute(stream);
        after?.Invoke();
    }
}
```

## Implementation Details

### File Structure
- **File**: `src/CUDA/Graphs/CUDAGraph.cs` (static class)
- **File**: `src/CUDA/Graphs/CUDAGraphCaptureOptions.cs`
- **File**: `src/CUDA/Graphs/CUDAGraphExtensions.cs`

### Dependencies
- CUDAGraphCapture (from spec_cuda_graph_capture_api)
- ICUDAGraph interface (from spec_cuda_graph_core_interfaces)
- CUDAGraphMemoryPool (from spec_cuda_graph_memory_pool)
- CUDAStream class (existing)
- System for Action, TimeSpan, Stopwatch

### API Design Principles
- Simple: Capture with minimal code
- Flexible: Support warm-up, validation, and options
- Safe: Validate by default
- Efficient: Support multiple executions

### Usage Examples

```csharp
// Simple capture
var graph = CUDAGraph.Capture(stream => model.Forward(input), stream);
graph.Execute(stream);

// Capture with warm-up
var graph = CUDAGraph.CaptureWithWarmup(
    stream => model.Forward(input),
    stream,
    warmupIterations: 3);

// Capture with options
var options = CUDAGraphCaptureOptions.Default
    .WithWarmup(3)
    .WithValidation(true);
var graph = CUDAGraph.Capture(stream => model.Forward(input), stream, options);

// Measure performance
var avgTimeMs = graph.GetAverageExecutionTimeMs(stream, 100);
Console.WriteLine($"Average execution time: {avgTimeMs:F2} ms");

// Execute multiple times
graph.ExecuteMultiple(stream, 1000);
```

## Success Criteria
- Static API can capture graphs
- Warm-up iterations work correctly
- Options are applied correctly
- Validation works on capture
- Extension methods work as expected
- Performance measurement is accurate

## Testing Requirements

### Unit Tests
- Test simple capture
- Test capture with warm-up
- Test capture with options
- Test validation on capture
- Test ExecuteMultiple
- Test MeasureExecutionTime
- Test GetAverageExecutionTimeMs
- Test ExecuteWithCallbacks

### Integration Tests
- Test with actual model execution (requires GPU)
- Test performance with real workloads
- Test with different capture modes
