using System;
using System.Diagnostics;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Contains the results of a performance profiling run.
/// </summary>
public class ProfilerResult
{
    /// <summary>
    /// Gets or sets the elapsed time in milliseconds.
    /// </summary>
    public long ElapsedMilliseconds { get; set; }

    /// <summary>
    /// Gets or sets the memory usage before the computation in bytes.
    /// </summary>
    public long MemoryBeforeBytes { get; set; }

    /// <summary>
    /// Gets or sets the memory usage after the computation in bytes.
    /// </summary>
    public long MemoryAfterBytes { get; set; }

    /// <summary>
    /// Gets or sets the memory delta (difference) in bytes.
    /// </summary>
    public long MemoryDeltaBytes => MemoryAfterBytes - MemoryBeforeBytes;

    /// <summary>
    /// Gets or sets whether the computation completed successfully.
    /// </summary>
    public bool CompletedSuccessfully { get; set; }

    /// <summary>
    /// Gets or sets the exception thrown if the computation failed.
    /// </summary>
    public Exception? Exception { get; set; }

    /// <summary>
    /// Gets a summary of the profiler result.
    /// </summary>
    /// <returns>A string summary.</returns>
    public string GetSummary()
    {
        var status = CompletedSuccessfully ? "Success" : $"Failed: {Exception?.Message}";
        return $"Time: {ElapsedMilliseconds}ms, Memory Delta: {MemoryDeltaBytes / 1024.0:F2} KB, Status: {status}";
    }
}

/// <summary>
/// Contains a snapshot of memory usage at a point in time.
/// </summary>
public class MemorySnapshot
{
    /// <summary>
    /// Gets or sets the total memory allocated in bytes.
    /// </summary>
    public long TotalMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the GC memory in bytes.
    /// </summary>
    public long GCMemoryBytes { get; set; }

    /// <summary>
    /// Gets or sets the timestamp of the snapshot.
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Creates a new memory snapshot.
    /// </summary>
    /// <returns>A new snapshot with current memory usage.</returns>
    public static MemorySnapshot Capture()
    {
        // Trigger GC to get accurate reading
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        return new MemorySnapshot
        {
            TotalMemoryBytes = GC.GetTotalMemory(false),
            GCMemoryBytes = GC.GetTotalMemory(false),
            Timestamp = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Calculates the memory difference between two snapshots.
    /// </summary>
    /// <param name="other">The snapshot to compare against.</param>
    /// <returns>The memory difference in bytes.</returns>
    public long DifferenceFrom(MemorySnapshot other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        return TotalMemoryBytes - other.TotalMemoryBytes;
    }
}

/// <summary>
/// Provides utilities for profiling computational performance and memory usage.
/// </summary>
public static class PerformanceProfiler
{
    /// <summary>
    /// Profiles the execution time and memory usage of a computation.
    /// </summary>
    /// <param name="computation">The computation to profile.</param>
    /// <param name="iterations">Number of iterations to run (default: 1).</param>
    /// <param name="warmupIterations">Number of warmup iterations to exclude (default: 0).</param>
    /// <returns>A profiler result with timing and memory information.</returns>
    public static ProfilerResult ProfileComputation(Action computation, int iterations = 1, int warmupIterations = 0)
    {
        if (computation == null)
            throw new ArgumentNullException(nameof(computation));

        if (iterations <= 0)
            throw new ArgumentException("Iterations must be positive", nameof(iterations));

        // Warmup iterations
        for (int i = 0; i < warmupIterations; i++)
        {
            try
            {
                computation();
            }
            catch
            {
                // Ignore warmup failures
            }
        }

        // Force GC before measurement
        GC.Collect();
        GC.WaitForPendingFinalizers();

        var memoryBefore = MemorySnapshot.Capture();
        var sw = Stopwatch.StartNew();

        Exception? caughtException = null;
        bool completed = false;

        try
        {
            for (int i = 0; i < iterations; i++)
            {
                computation();
            }
            completed = true;
        }
        catch (Exception ex)
        {
            caughtException = ex;
        }

        sw.Stop();

        // Force GC after measurement
        GC.Collect();
        GC.WaitForPendingFinalizers();

        var memoryAfter = MemorySnapshot.Capture();

        return new ProfilerResult
        {
            ElapsedMilliseconds = sw.ElapsedMilliseconds,
            MemoryBeforeBytes = memoryBefore.TotalMemoryBytes,
            MemoryAfterBytes = memoryAfter.TotalMemoryBytes,
            CompletedSuccessfully = completed,
            Exception = caughtException
        };
    }

    /// <summary>
    /// Profiles the execution time and memory usage of a function with a return value.
    /// </summary>
    /// <typeparam name="T">The return type of the computation.</typeparam>
    /// <param name="computation">The computation to profile.</param>
    /// <param name="iterations">Number of iterations to run (default: 1).</param>
    /// <param name="warmupIterations">Number of warmup iterations (default: 0).</param>
    /// <returns>A tuple containing the profiler result and the return value.</returns>
    public static (ProfilerResult result, T? value) ProfileComputation<T>(Func<T> computation, int iterations = 1, int warmupIterations = 0)
    {
        if (computation == null)
            throw new ArgumentNullException(nameof(computation));

        if (iterations <= 0)
            throw new ArgumentException("Iterations must be positive", nameof(iterations));

        // Warmup iterations
        T? result = default;
        for (int i = 0; i < warmupIterations; i++)
        {
            try
            {
                result = computation();
            }
            catch
            {
                // Ignore warmup failures
            }
        }

        // Force GC before measurement
        GC.Collect();
        GC.WaitForPendingFinalizers();

        var memoryBefore = MemorySnapshot.Capture();
        var sw = Stopwatch.StartNew();

        Exception? caughtException = null;
        bool completed = false;

        try
        {
            for (int i = 0; i < iterations; i++)
            {
                result = computation();
            }
            completed = true;
        }
        catch (Exception ex)
        {
            caughtException = ex;
        }

        sw.Stop();

        // Force GC after measurement
        GC.Collect();
        GC.WaitForPendingFinalizers();

        var memoryAfter = MemorySnapshot.Capture();

        var profilerResult = new ProfilerResult
        {
            ElapsedMilliseconds = sw.ElapsedMilliseconds,
            MemoryBeforeBytes = memoryBefore.TotalMemoryBytes,
            MemoryAfterBytes = memoryAfter.TotalMemoryBytes,
            CompletedSuccessfully = completed,
            Exception = caughtException
        };

        return (profilerResult, result);
    }

    /// <summary>
    /// Captures the current memory usage.
    /// </summary>
    /// <returns>A memory snapshot.</returns>
    public static MemorySnapshot CaptureMemoryUsage()
    {
        return MemorySnapshot.Capture();
    }

    /// <summary>
    /// Runs a benchmark that measures multiple runs and reports statistics.
    /// </summary>
    /// <param name="computation">The computation to benchmark.</param>
    /// <param name="runs">Number of benchmark runs (default: 10).</param>
    /// <param name="iterationsPerRun">Number of iterations per run (default: 10).</param>
    /// <returns>Benchmark statistics including mean, min, max, and std dev.</returns>
    public static BenchmarkStatistics RunBenchmark(Action computation, int runs = 10, int iterationsPerRun = 10)
    {
        if (computation == null)
            throw new ArgumentNullException(nameof(computation));

        if (runs <= 0 || iterationsPerRun <= 0)
            throw new ArgumentException("Runs and iterations per run must be positive");

        var timings = new long[runs];

        for (int run = 0; run < runs; run++)
        {
            // Warmup
            for (int i = 0; i < 3; i++)
            {
                try { computation(); } catch { }
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterationsPerRun; i++)
            {
                computation();
            }
            sw.Stop();

            timings[run] = sw.ElapsedMilliseconds;
        }

        // Calculate statistics
        var sum = 0L;
        var min = long.MaxValue;
        var max = long.MinValue;

        foreach (var timing in timings)
        {
            sum += timing;
            if (timing < min) min = timing;
            if (timing > max) max = timing;
        }

        var mean = (double)sum / runs;

        // Calculate standard deviation
        var variance = 0.0;
        foreach (var timing in timings)
        {
            variance += Math.Pow(timing - mean, 2);
        }
        variance /= runs;
        var stdDev = Math.Sqrt(variance);

        // Calculate median
        var sortedTimings = (long[])timings.Clone();
        Array.Sort(sortedTimings);
        var median = sortedTimings[runs / 2];

        return new BenchmarkStatistics
        {
            MeanMilliseconds = mean,
            MinMilliseconds = min,
            MaxMilliseconds = max,
            StdDevMilliseconds = stdDev,
            MedianMilliseconds = median,
            TotalRuns = runs,
            IterationsPerRun = iterationsPerRun
        };
    }
}

/// <summary>
/// Contains statistics from a benchmark run.
/// </summary>
public class BenchmarkStatistics
{
    /// <summary>
    /// Gets or sets the mean execution time in milliseconds.
    /// </summary>
    public double MeanMilliseconds { get; set; }

    /// <summary>
    /// Gets or sets the minimum execution time in milliseconds.
    /// </summary>
    public long MinMilliseconds { get; set; }

    /// <summary>
    /// Gets or sets the maximum execution time in milliseconds.
    /// </summary>
    public long MaxMilliseconds { get; set; }

    /// <summary>
    /// Gets or sets the standard deviation of execution times in milliseconds.
    /// </summary>
    public double StdDevMilliseconds { get; set; }

    /// <summary>
    /// Gets or sets the median execution time in milliseconds.
    /// </summary>
    public long MedianMilliseconds { get; set; }

    /// <summary>
    /// Gets or sets the total number of benchmark runs.
    /// </summary>
    public int TotalRuns { get; set; }

    /// <summary>
    /// Gets or sets the number of iterations per run.
    /// </summary>
    public int IterationsPerRun { get; set; }

    /// <summary>
    /// Gets a summary of the benchmark statistics.
    /// </summary>
    /// <returns>A formatted string summary.</returns>
    public string GetSummary()
    {
        return $"Benchmark Results ({TotalRuns} runs, {IterationsPerRun} iterations/run):\n" +
               $"  Mean: {MeanMilliseconds:F2} ms\n" +
               $"  Median: {MedianMilliseconds} ms\n" +
               $"  Min: {MinMilliseconds} ms\n" +
               $"  Max: {MaxMilliseconds} ms\n" +
               $"  StdDev: {StdDevMilliseconds:F2} ms";
    }
}
