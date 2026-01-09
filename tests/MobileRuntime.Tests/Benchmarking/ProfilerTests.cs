using System;
using System.IO;
using System.Threading;
using MobileRuntime.Benchmarking;
using MobileRuntime.Benchmarking.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace MobileRuntime.Tests.Benchmarking;

[TestClass]
public class ProfilerTests
{
    [TestMethod]
    public void Profile_ShouldMeasureExecutionTime()
    {
        // Arrange
        using var profiler = new Profiler();

        // Act
        profiler.Profile("Test Operation", () =>
        {
            Thread.Sleep(10);
        });

        var report = profiler.GetReport();

        // Assert
        Assert.Single(report.Entries);
        Assert.Equal("Test Operation", report.Entries[0].Name);
        Assert.Equal(1, report.Entries[0].CallCount);
        Assert.True(report.Entries[0].TotalTime.TotalMilliseconds > 0);
    }

    [TestMethod]
    public void Profile_WithReturn_ShouldMeasureExecutionTime()
    {
        // Arrange
        using var profiler = new Profiler();

        // Act
        var result = profiler.Profile("Test Operation with Return", () =>
        {
            Thread.Sleep(10);
            return 42;
        });

        var report = profiler.GetReport();

        // Assert
        Assert.Equal(42, result);
        Assert.Single(report.Entries);
        Assert.Equal("Test Operation with Return", report.Entries[0].Name);
    }

    [TestMethod]
    public void BeginScope_ShouldCreateScope()
    {
        // Arrange
        using var profiler = new Profiler();

        // Act
        using (profiler.BeginScope("Test Scope"))
        {
            Thread.Sleep(10);
        }

        var report = profiler.GetReport();

        // Assert
        Assert.Single(report.Entries);
        Assert.Equal("Test Scope", report.Entries[0].Name);
        Assert.Equal(1, report.Entries[0].CallCount);
    }

    [TestMethod]
    public void NestedScopes_ShouldBeMeasuredCorrectly()
    {
        // Arrange
        using var profiler = new Profiler();

        // Act
        using (profiler.BeginScope("Outer Scope"))
        {
            Thread.Sleep(20);
            using (profiler.BeginScope("Inner Scope"))
            {
                Thread.Sleep(10);
            }
        }

        var report = profiler.GetReport();

        // Assert
        Assert.Equal(2, report.Entries.Count);
        Assert.Contains(report.Entries, e => e.Name == "Outer Scope");
        Assert.Contains(report.Entries, e => e.Name == "Inner Scope");
    }

    [TestMethod]
    public void MultipleCallsToSameOperation_ShouldAccumulate()
    {
        // Arrange
        using var profiler = new Profiler();

        // Act
        for (int i = 0; i < 5; i++)
        {
            profiler.Profile("Repeated Operation", () =>
            {
                Thread.Sleep(10);
            });
        }

        var report = profiler.GetReport();

        // Assert
        Assert.Single(report.Entries);
        Assert.Equal(5, report.Entries[0].CallCount);
        Assert.True(report.Entries[0].TotalTime.TotalMilliseconds > 0);
        Assert.True(report.Entries[0].AverageTime.TotalMilliseconds > 0);
    }

    [TestMethod]
    public void GetReport_ShouldReturnCorrectStats()
    {
        // Arrange
        using var profiler = new Profiler();
        profiler.Profile("Operation 1", () => Thread.Sleep(10));
        profiler.Profile("Operation 2", () => Thread.Sleep(20));

        // Act
        var report = profiler.GetReport();

        // Assert
        Assert.Equal(2, report.Entries.Count);
        Assert.True(report.TotalTime.TotalMilliseconds > 0);
        Assert.NotNull(report.Timestamp);
    }

    [TestMethod]
    public void Reset_ShouldClearAllProfiles()
    {
        // Arrange
        using var profiler = new Profiler();
        profiler.Profile("Test Operation", () => Thread.Sleep(10));

        // Act
        profiler.Reset();
        var report = profiler.GetReport();

        // Assert
        Assert.Empty(report.Entries);
    }

    [TestMethod]
    public void ExportReport_ShouldCreateFile()
    {
        // Arrange
        using var profiler = new Profiler();
        profiler.Profile("Test Operation", () => Thread.Sleep(10));
        var filePath = Path.Combine(Path.GetTempPath(), $"test_profile_{Guid.NewGuid()}.json");

        try
        {
            // Act
            profiler.ExportReport(filePath, ReportFormat.Json);

            // Assert
            Assert.True(File.Exists(filePath));
            var content = File.ReadAllText(filePath);
            Assert.Contains("Test Operation", content);
        }
        finally
        {
            if (File.Exists(filePath))
                File.Delete(filePath);
        }
    }

    [TestMethod]
    public void PrintReport_ShouldNotThrow()
    {
        // Arrange
        using var profiler = new Profiler();
        profiler.Profile("Test Operation", () => Thread.Sleep(10));

        // Act & Assert
        var exception = Record.Exception(() => profiler.PrintReport());
        Assert.Null(exception);
    }
}
