using System.Diagnostics;

namespace MLFramework.HAL.CUDA;

/// <summary>
/// Extension methods for CUDA graph operations
/// </summary>
public static class CUDAGraphExtensions
{
    /// <summary>
    /// Executes a graph multiple times
    /// </summary>
    /// <param name="graph">The graph to execute</param>
    /// <param name="stream">CUDA stream for execution</param>
    /// <param name="iterations">Number of iterations to execute</param>
    /// <exception cref="ArgumentNullException">Thrown when graph or stream is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when iterations is less than or equal to zero</exception>
    public static void ExecuteMultiple(this ICUDAGraph graph, CudaStream stream, int iterations)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (iterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(iterations), "Iterations must be greater than zero");

        for (int i = 0; i < iterations; i++)
        {
            graph.Execute(stream);
        }
    }

    /// <summary>
    /// Measures execution time of a graph
    /// </summary>
    /// <param name="graph">The graph to measure</param>
    /// <param name="stream">CUDA stream for execution</param>
    /// <param name="iterations">Number of iterations to measure</param>
    /// <returns>Total execution time for all iterations</returns>
    /// <exception cref="ArgumentNullException">Thrown when graph or stream is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when iterations is less than or equal to zero</exception>
    public static TimeSpan MeasureExecutionTime(
        this ICUDAGraph graph,
        CudaStream stream,
        int iterations = 10)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (iterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(iterations), "Iterations must be greater than zero");

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
    /// <param name="graph">The graph to measure</param>
    /// <param name="stream">CUDA stream for execution</param>
    /// <param name="iterations">Number of iterations to measure</param>
    /// <returns>Average execution time in milliseconds</returns>
    /// <exception cref="ArgumentNullException">Thrown when graph or stream is null</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when iterations is less than or equal to zero</exception>
    public static double GetAverageExecutionTimeMs(
        this ICUDAGraph graph,
        CudaStream stream,
        int iterations = 10)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (iterations <= 0)
            throw new ArgumentOutOfRangeException(nameof(iterations), "Iterations must be greater than zero");

        var totalTime = MeasureExecutionTime(graph, stream, iterations);
        return totalTime.TotalMilliseconds / iterations;
    }

    /// <summary>
    /// Executes a graph with a callback before and after
    /// </summary>
    /// <param name="graph">The graph to execute</param>
    /// <param name="stream">CUDA stream for execution</param>
    /// <param name="before">Action to execute before the graph (optional)</param>
    /// <param name="after">Action to execute after the graph (optional)</param>
    /// <exception cref="ArgumentNullException">Thrown when graph or stream is null</exception>
    public static void ExecuteWithCallbacks(
        this ICUDAGraph graph,
        CudaStream stream,
        Action? before = null,
        Action? after = null)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        before?.Invoke();
        graph.Execute(stream);
        after?.Invoke();
    }
}
