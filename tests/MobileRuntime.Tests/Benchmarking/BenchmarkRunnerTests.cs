using System;
using System.IO;
using System.Linq;
using System.Threading;
using MobileRuntime.Benchmarking;
using MobileRuntime.Benchmarking.Comparison;
using MobileRuntime.Benchmarking.Models;
using MobileRuntime.Benchmarking.Interfaces;
using MobileRuntime.Benchmarking.Monitoring;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace MobileRuntime.Tests.Benchmarking;

[TestClass]
public class BenchmarkRunnerTests
{
    [TestMethod]
    public void RunBenchmark_ShouldMeasureCorrectly()
    {
        // Arrange
        using var runner = new BenchmarkRunner();
        var iterations = 10;

        // Act
        var result = runner.RunBenchmark("Test Benchmark", () =>
        {
            Thread.Sleep(10);
        }, iterations);

        // Assert
        Assert.Equal("Test Benchmark", result.Name);
        Assert.Equal(iterations, result.Iterations);
        Assert.True(result.MinTime.TotalMilliseconds > 0);
        Assert.True(result.MaxTime.TotalMilliseconds > 0);
        Assert.True(result.AverageTime.TotalMilliseconds > 0);
        Assert.True(result.StdDev >= 0);
    }

    [TestMethod]
    public void RunBenchmark_WithReturn_ShouldMeasureCorrectly()
    {
        // Arrange
        using var runner = new BenchmarkRunner();
        var iterations = 10;

        // Act
        var result = runner.RunBenchmark("Test Benchmark with Return", () =>
        {
            Thread.Sleep(10);
            return 42;
        }, iterations);

        // Assert
        Assert.Equal("Test Benchmark with Return", result.Name);
        Assert.Equal(iterations, result.Iterations);
        Assert.True(result.MinTime.TotalMilliseconds > 0);
    }

    [TestMethod]
    public void RunBenchmarkSuite_ShouldRunAllBenchmarks()
    {
        // Arrange
        using var runner = new BenchmarkRunner();
        var benchmarks = new BenchmarkConfig[]
        {
            new BenchmarkConfig
            {
                Name = "Benchmark 1",
                Benchmark = () => Thread.Sleep(10),
                Iterations = 5
            },
            new BenchmarkConfig
            {
                Name = "Benchmark 2",
                Benchmark = () => Thread.Sleep(20),
                Iterations = 5
            }
        };

        // Act
        var results = runner.RunBenchmarkSuite("Test Suite", benchmarks);

        // Assert
        Assert.Equal("Test Suite", results.SuiteName);
        Assert.Equal(2, results.Results.Count);
        Assert.Equal(2, results.Summary.PassedCount);
        Assert.Equal(0, results.Summary.FailedCount);
    }

    [TestMethod]
    public void GetResults_ShouldReturnAllResults()
    {
        // Arrange
        using var runner = new BenchmarkRunner();
        runner.RunBenchmark("Benchmark 1", () => Thread.Sleep(10), 5);
        runner.RunBenchmark("Benchmark 2", () => Thread.Sleep(10), 5);

        // Act
        var results = runner.GetResults();

        // Assert
        Assert.Equal(2, results.Results.Count);
    }

    [TestMethod]
    public void Reset_ShouldClearResults()
    {
        // Arrange
        using var runner = new BenchmarkRunner();
        runner.RunBenchmark("Benchmark 1", () => Thread.Sleep(10), 5);

        // Act
        runner.Reset();
        var results = runner.GetResults();

        // Assert
        Assert.Empty(results.Results);
    }

    [TestMethod]
    public void ExportResults_ShouldCreateFile()
    {
        // Arrange
        using var runner = new BenchmarkRunner();
        runner.RunBenchmark("Test Benchmark", () => Thread.Sleep(10), 5);
        var filePath = Path.Combine(Path.GetTempPath(), $"test_benchmark_{Guid.NewGuid()}.json");

        try
        {
            // Act
            runner.ExportResults(filePath, ReportFormat.Json);

            // Assert
            Assert.True(File.Exists(filePath));
            var content = File.ReadAllText(filePath);
            Assert.Contains("Test Benchmark", content);
        }
        finally
        {
            if (File.Exists(filePath))
                File.Delete(filePath);
        }
    }

    [TestMethod]
    public void RunBenchmark_WithMemoryMonitor_ShouldTrackMemory()
    {
        // Arrange
        var memoryMonitor = new MemoryMonitor();
        using var runner = new BenchmarkRunner(memoryMonitor);

        // Act
        var result = runner.RunBenchmark("Memory Benchmark", () =>
        {
            var data = new byte[1024 * 1024]; // 1MB
        }, 5);

        // Assert
        Assert.True(result.AverageMemoryBytes > 0);
    }

    [TestMethod]
    public void RunBenchmark_FailedBenchmark_ShouldBeRecorded()
    {
        // Arrange
        using var runner = new BenchmarkRunner();
        var benchmarks = new BenchmarkConfig[]
        {
            new BenchmarkConfig
            {
                Name = "Failed Benchmark",
                Benchmark = () => throw new Exception("Test exception"),
                Iterations = 5
            }
        };

        // Act
        var results = runner.RunBenchmarkSuite("Failed Suite", benchmarks);

        // Assert
        Assert.Equal(1, results.Summary.FailedCount);
        Assert.Equal(0, results.Summary.PassedCount);
        Assert.True(results.Results[0].Metadata.ContainsKey("Error"));
    }
}
