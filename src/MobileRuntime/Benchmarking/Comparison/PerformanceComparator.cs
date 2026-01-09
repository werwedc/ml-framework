using System;
using System.Linq;
using MobileRuntime.Benchmarking.Models;

namespace MobileRuntime.Benchmarking.Comparison;

public class PerformanceComparator
{
    private const double DefaultRegressionThreshold = 5.0; // 5% change threshold

    public ComparisonReport Compare(BenchmarkResults baseline, BenchmarkResults current)
    {
        var report = new ComparisonReport
        {
            Baseline = baseline,
            Current = current
        };

        var baselineDict = baseline.Results.ToDictionary(r => r.Name);
        var currentDict = current.Results.ToDictionary(r => r.Name);

        var allNames = baselineDict.Keys.Union(currentDict.Keys).ToList();

        foreach (var name in allNames)
        {
            if (baselineDict.TryGetValue(name, out var baselineResult) &&
                currentDict.TryGetValue(name, out var currentResult))
            {
                var entry = CompareEntries(baselineResult, currentResult);
                report.Entries.Add(entry);

                if (entry.IsRegression)
                {
                    report.Regressions.Add(entry);
                }
                else if (entry.TimeChangePercent < -DefaultRegressionThreshold)
                {
                    report.Improvements.Add(entry);
                }
            }
        }

        return report;
    }

    public ComparisonReport CompareProfiles(ProfileReport baseline, ProfileReport current)
    {
        var baselineBenchmark = ConvertProfileToBenchmarkResults(baseline);
        var currentBenchmark = ConvertProfileToBenchmarkResults(current);
        return Compare(baselineBenchmark, currentBenchmark);
    }

    private ComparisonEntry CompareEntries(BenchmarkResult baseline, BenchmarkResult current)
    {
        var timeChangePercent = CalculatePercentChange(
            baseline.AverageTime.TotalMilliseconds,
            current.AverageTime.TotalMilliseconds
        );

        var memoryChangePercent = CalculatePercentChange(
            baseline.AverageMemoryBytes / (1024.0 * 1024.0),
            current.AverageMemoryBytes / (1024.0 * 1024.0)
        );

        return new ComparisonEntry
        {
            Name = baseline.Name,
            BaselineTimeMs = baseline.AverageTime.TotalMilliseconds,
            CurrentTimeMs = current.AverageTime.TotalMilliseconds,
            TimeChangePercent = timeChangePercent,
            BaselineMemoryMB = baseline.AverageMemoryBytes / (1024.0 * 1024.0),
            CurrentMemoryMB = current.AverageMemoryBytes / (1024.0 * 1024.0),
            MemoryChangePercent = memoryChangePercent,
            IsRegression = IsRegression(baseline, current, DefaultRegressionThreshold)
        };
    }

    private bool IsRegression(BenchmarkResult baseline, BenchmarkResult current, double thresholdPercent)
    {
        var timeChangePercent = CalculatePercentChange(
            baseline.AverageTime.TotalMilliseconds,
            current.AverageTime.TotalMilliseconds
        );

        return timeChangePercent > thresholdPercent;
    }

    private double CalculatePercentChange(double baseline, double current)
    {
        if (baseline == 0)
            return current > 0 ? 100.0 : 0.0;

        return ((current - baseline) / baseline) * 100.0;
    }

    private BenchmarkResults ConvertProfileToBenchmarkResults(ProfileReport profile)
    {
        return new BenchmarkResults
        {
            SuiteName = "Profile",
            Results = profile.Entries.Select(entry => new BenchmarkResult
            {
                Name = entry.Name,
                Iterations = entry.CallCount,
                AverageTime = entry.AverageTime,
                MinTime = entry.MinTime,
                MaxTime = entry.MaxTime,
                AverageMemoryBytes = entry.TotalMemoryBytes / entry.CallCount,
                Timestamp = profile.Timestamp
            }).ToList(),
            Summary = new BenchmarkSummary()
        };
    }
}
