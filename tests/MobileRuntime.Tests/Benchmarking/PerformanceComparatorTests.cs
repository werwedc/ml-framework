using System.Linq;
using MobileRuntime.Benchmarking.Comparison;
using MobileRuntime.Benchmarking.Models;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace MobileRuntime.Tests.Benchmarking;

[TestClass]
public class PerformanceComparatorTests
{
    [TestMethod]
    public void Compare_ShouldDetectImprovement()
    {
        // Arrange
        var baseline = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Test Operation",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(100),
                    AverageMemoryBytes = 100 * 1024 * 1024
                }
            }
        };

        var current = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Test Operation",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(50), // Faster
                    AverageMemoryBytes = 50 * 1024 * 1024 // Less memory
                }
            }
        };

        var comparator = new PerformanceComparator();

        // Act
        var report = comparator.Compare(baseline, current);

        // Assert
        Assert.Single(report.Entries);
        Assert.Single(report.Improvements);
        Assert.Empty(report.Regressions);
        Assert.True(report.Entries[0].TimeChangePercent < 0);
    }

    [TestMethod]
    public void Compare_ShouldDetectRegression()
    {
        // Arrange
        var baseline = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Test Operation",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(50),
                    AverageMemoryBytes = 50 * 1024 * 1024
                }
            }
        };

        var current = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Test Operation",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(100), // Slower
                    AverageMemoryBytes = 100 * 1024 * 1024 // More memory
                }
            }
        };

        var comparator = new PerformanceComparator();

        // Act
        var report = comparator.Compare(baseline, current);

        // Assert
        Assert.Single(report.Entries);
        Assert.Empty(report.Improvements);
        Assert.Single(report.Regressions);
        Assert.True(report.Entries[0].TimeChangePercent > 0);
        Assert.True(report.Entries[0].IsRegression);
    }

    [TestMethod]
    public void Compare_MultipleOperations_ShouldCompareAll()
    {
        // Arrange
        var baseline = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Operation 1",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(100)
                },
                new BenchmarkResult
                {
                    Name = "Operation 2",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(200)
                }
            }
        };

        var current = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Operation 1",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(50) // Improved
                },
                new BenchmarkResult
                {
                    Name = "Operation 2",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(250) // Regressed
                }
            }
        };

        var comparator = new PerformanceComparator();

        // Act
        var report = comparator.Compare(baseline, current);

        // Assert
        Assert.Equal(2, report.Entries.Count);
        Assert.Single(report.Improvements);
        Assert.Single(report.Regressions);
    }

    [TestMethod]
    public void Compare_NewOperation_ShouldBeIncluded()
    {
        // Arrange
        var baseline = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Operation 1",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(100)
                }
            }
        };

        var current = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Operation 1",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(100)
                },
                new BenchmarkResult
                {
                    Name = "Operation 2",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(150)
                }
            }
        };

        var comparator = new PerformanceComparator();

        // Act
        var report = comparator.Compare(baseline, current);

        // Assert
        Assert.Equal(2, report.Entries.Count);
        Assert.Contains(report.Entries, e => e.Name == "Operation 2");
    }

    [TestMethod]
    public void Compare_ShouldCalculateCorrectPercentages()
    {
        // Arrange
        var baseline = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Test Operation",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(100)
                }
            }
        };

        var current = new BenchmarkResults
        {
            Results = new List<BenchmarkResult>
            {
                new BenchmarkResult
                {
                    Name = "Test Operation",
                    Iterations = 10,
                    AverageTime = TimeSpan.FromMilliseconds(110)
                }
            }
        };

        var comparator = new PerformanceComparator();

        // Act
        var report = comparator.Compare(baseline, current);

        // Assert
        Assert.InRange(report.Entries[0].TimeChangePercent, 9.9, 10.1);
    }
}
