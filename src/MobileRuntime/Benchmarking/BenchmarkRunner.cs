using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MobileRuntime.Benchmarking.Interfaces;
using MobileRuntime.Benchmarking.Models;
using MobileRuntime.Benchmarking.Export;

namespace MobileRuntime.Benchmarking;

public class BenchmarkRunner : IDisposable
{
    private readonly List<BenchmarkResult> _results;
    private readonly Stopwatch _stopwatch;
    private readonly IMemoryMonitor? _memoryMonitor;
    private readonly IEnergyMonitor? _energyMonitor;
    private bool _disposed;

    public BenchmarkRunner(IMemoryMonitor? memoryMonitor = null, IEnergyMonitor? energyMonitor = null)
    {
        _results = new List<BenchmarkResult>();
        _stopwatch = new Stopwatch();
        _memoryMonitor = memoryMonitor;
        _energyMonitor = energyMonitor;
    }

    public BenchmarkResult RunBenchmark(string name, Action benchmark, int iterations = 10)
    {
        if (benchmark == null)
            throw new ArgumentNullException(nameof(benchmark));

        // Warmup
        for (int i = 0; i < Math.Min(iterations / 2, 3); i++)
        {
            benchmark();
        }

        // Start monitoring
        _memoryMonitor?.StartMonitoring();
        _energyMonitor?.StartMonitoring();

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var elapsedTimes = new List<TimeSpan>(iterations);
        var memorySnapshots = new List<MemorySnapshot>();
        var energySnapshots = new List<EnergySnapshot>();

        for (int i = 0; i < iterations; i++)
        {
            _stopwatch.Restart();
            benchmark();
            _stopwatch.Stop();

            elapsedTimes.Add(_stopwatch.Elapsed);

            if (_memoryMonitor != null)
            {
                memorySnapshots.Add(_memoryMonitor.GetSnapshot());
            }

            if (_energyMonitor != null)
            {
                energySnapshots.Add(_energyMonitor.GetSnapshot());
            }
        }

        // Stop monitoring
        _memoryMonitor?.StopMonitoring();
        _energyMonitor?.StopMonitoring();

        var result = CalculateBenchmarkResult(name, iterations, elapsedTimes, memorySnapshots, energySnapshots);
        _results.Add(result);

        return result;
    }

    public BenchmarkResult RunBenchmark<T>(string name, Func<T> benchmark, int iterations = 10)
    {
        if (benchmark == null)
            throw new ArgumentNullException(nameof(benchmark));

        return RunBenchmark(name, () =>
        {
            var _ = benchmark();
        }, iterations);
    }

    public BenchmarkResults RunBenchmarkSuite(string suiteName, params BenchmarkConfig[] benchmarks)
    {
        var results = new List<BenchmarkResult>();
        int passedCount = 0;
        int failedCount = 0;

        foreach (var config in benchmarks)
        {
            try
            {
                var result = RunBenchmark(config.Name, config.Benchmark!, config.Iterations);
                results.Add(result);
                passedCount++;
            }
            catch (Exception ex)
            {
                failedCount++;
                results.Add(new BenchmarkResult
                {
                    Name = config.Name,
                    Iterations = 0,
                    Timestamp = DateTime.UtcNow,
                    Metadata = new Dictionary<string, string>
                    {
                        { "Error", ex.Message },
                        { "Failed", "true" }
                    }
                });
            }
        }

        var summary = new BenchmarkSummary
        {
            TotalTime = results.Where(r => r.Iterations > 0).Aggregate(TimeSpan.Zero, (sum, r) => sum + r.AverageTime),
            TotalMemoryBytes = results.Where(r => r.Iterations > 0).Sum(r => r.AverageMemoryBytes),
            TotalEnergyJoules = results.Where(r => r.Iterations > 0).Sum(r => r.AverageEnergyJoules),
            PassedCount = passedCount,
            FailedCount = failedCount
        };

        var benchmarkResults = new BenchmarkResults
        {
            SuiteName = suiteName,
            Results = results,
            Summary = summary
        };

        _results.AddRange(results);

        return benchmarkResults;
    }

    public BenchmarkResults GetResults()
    {
        var summary = new BenchmarkSummary
        {
            TotalTime = _results.Aggregate(TimeSpan.Zero, (sum, r) => sum + r.AverageTime),
            TotalMemoryBytes = _results.Sum(r => r.AverageMemoryBytes),
            TotalEnergyJoules = _results.Sum(r => r.AverageEnergyJoules),
            PassedCount = _results.Count(r => r.Iterations > 0),
            FailedCount = _results.Count(r => r.Iterations == 0)
        };

        return new BenchmarkResults
        {
            Results = _results.ToList(),
            Summary = summary
        };
    }

    public void Reset()
    {
        _results.Clear();
        _memoryMonitor?.Reset();
        _energyMonitor?.Reset();
    }

    public void ExportResults(string filePath, string format = ReportFormat.Json)
    {
        var results = GetResults();
        BenchmarkExporter.Export(results, filePath, format);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _memoryMonitor?.StopMonitoring();
            _energyMonitor?.StopMonitoring();
            _memoryMonitor?.Reset();
            _energyMonitor?.Reset();

            // Dispose monitors if they implement IDisposable
            if (_memoryMonitor is IDisposable disposableMemory)
                disposableMemory.Dispose();

            if (_energyMonitor is IDisposable disposableEnergy)
                disposableEnergy.Dispose();

            _stopwatch.Stop();

            _disposed = true;
        }

        GC.SuppressFinalize(this);
    }

    private BenchmarkResult CalculateBenchmarkResult(
        string name,
        int iterations,
        List<TimeSpan> elapsedTimes,
        List<MemorySnapshot> memorySnapshots,
        List<EnergySnapshot> energySnapshots)
    {
        var sortedTimes = elapsedTimes.OrderBy(t => t.TotalMilliseconds).ToList();
        var minTime = sortedTimes.First();
        var maxTime = sortedTimes.Last();
        var medianTime = sortedTimes[sortedTimes.Count / 2];
        var averageTime = TimeSpan.FromMilliseconds(sortedTimes.Average(t => t.TotalMilliseconds));

        // Calculate standard deviation
        var mean = sortedTimes.Average(t => t.TotalMilliseconds);
        var variance = sortedTimes.Average(t => Math.Pow(t.TotalMilliseconds - mean, 2));
        var stdDev = Math.Sqrt(variance);

        long minMemory = 0, maxMemory = 0, averageMemory = 0;
        if (memorySnapshots.Any())
        {
            minMemory = memorySnapshots.Min(s => s.GCMemoryBytes);
            maxMemory = memorySnapshots.Max(s => s.GCMemoryBytes);
            averageMemory = (long)memorySnapshots.Average(s => s.GCMemoryBytes);
        }

        double minEnergy = 0.0, maxEnergy = 0.0, averageEnergy = 0.0;
        if (energySnapshots.Any())
        {
            minEnergy = energySnapshots.Min(s => s.EnergyJoules);
            maxEnergy = energySnapshots.Max(s => s.EnergyJoules);
            averageEnergy = energySnapshots.Average(s => s.EnergyJoules);
        }

        return new BenchmarkResult
        {
            Name = name,
            Iterations = iterations,
            MinTime = minTime,
            MaxTime = maxTime,
            AverageTime = averageTime,
            MedianTime = medianTime,
            StdDev = stdDev,
            MinMemoryBytes = minMemory,
            MaxMemoryBytes = maxMemory,
            AverageMemoryBytes = averageMemory,
            MinEnergyJoules = minEnergy,
            MaxEnergyJoules = maxEnergy,
            AverageEnergyJoules = averageEnergy,
            Timestamp = DateTime.UtcNow
        };
    }
}
