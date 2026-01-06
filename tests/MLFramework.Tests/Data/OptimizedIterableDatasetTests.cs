using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace MLFramework.Data.Tests;

/// <summary>
/// Unit tests for OptimizedIterableDataset.
/// </summary>
public class OptimizedIterableDatasetTests
{
    [Fact]
    public void Constructor_WithNullIteratorFactory_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new OptimizedIterableDataset<int>(null!));
    }

    [Fact]
    public void Constructor_WithNegativeWorkerId_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var factory = CreateSampleFactory(10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OptimizedIterableDataset<int>(factory, enableWorkerSupport: true, workerId: -1, totalWorkers: 2));
    }

    [Fact]
    public void Constructor_WithZeroTotalWorkers_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var factory = CreateSampleFactory(10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OptimizedIterableDataset<int>(factory, enableWorkerSupport: true, workerId: 0, totalWorkers: 0));
    }

    [Fact]
    public void Constructor_WithWorkerIdGreaterThanTotalWorkers_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var factory = CreateSampleFactory(10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OptimizedIterableDataset<int>(factory, enableWorkerSupport: true, workerId: 2, totalWorkers: 2));
    }

    [Fact]
    public void GetEnumerator_WithoutWorkerSupport_ReturnsAllSamples()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5 };
        var factory = CreateSampleFactory(samples);
        var dataset = new OptimizedIterableDataset<int>(factory, enableWorkerSupport: false);

        // Act
        var result = dataset.ToList();

        // Assert
        Assert.Equal(samples, result);
    }

    [Fact]
    public void GetEnumerator_CachesIterator()
    {
        // Arrange
        int callCount = 0;
        var factory = () =>
        {
            callCount++;
            return CreateSampleEnumerator(5);
        };
        var dataset = new OptimizedIterableDataset<int>(factory);

        // Act - GetEnumerator should be called twice
        var firstIteration = dataset.ToList();
        var secondIteration = dataset.ToList();

        // Assert - Factory should only be called once due to caching
        Assert.Equal(1, callCount);
        Assert.Equal(new List<int> { 1, 2, 3, 4, 5 }, firstIteration);
        Assert.Equal(new List<int> { 1, 2, 3, 4, 5 }, secondIteration);
    }

    [Fact]
    public void Reset_ClearsCachedIterator()
    {
        // Arrange
        int callCount = 0;
        var factory = () =>
        {
            callCount++;
            return CreateSampleEnumerator(5);
        };
        var dataset = new OptimizedIterableDataset<int>(factory);

        // Act
        var firstIteration = dataset.ToList();
        dataset.Reset();
        var secondIteration = dataset.ToList();

        // Assert - Factory should be called twice due to reset
        Assert.Equal(2, callCount);
        Assert.Equal(new List<int> { 1, 2, 3, 4, 5 }, firstIteration);
        Assert.Equal(new List<int> { 1, 2, 3, 4, 5 }, secondIteration);
    }

    [Fact]
    public void GetEnumerator_WithSingleWorker_ReturnsAllSamples()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5 };
        var factory = CreateSampleFactory(samples);
        var dataset = new OptimizedIterableDataset<int>(
            factory,
            enableWorkerSupport: true,
            workerId: 0,
            totalWorkers: 1);

        // Act
        var result = dataset.ToList();

        // Assert
        Assert.Equal(samples, result);
    }

    [Fact]
    public void GetEnumerator_WithTwoWorkers_PartitionsCorrectly()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        var dataset0 = new OptimizedIterableDataset<int>(
            CreateSampleFactory(samples),
            enableWorkerSupport: true,
            workerId: 0,
            totalWorkers: 2);

        var dataset1 = new OptimizedIterableDataset<int>(
            CreateSampleFactory(samples),
            enableWorkerSupport: true,
            workerId: 1,
            totalWorkers: 2);

        // Act
        var result0 = dataset0.ToList();
        var result1 = dataset1.ToList();

        // Assert - Worker 0 gets even positions (0, 2, 4, 6, 8), Worker 1 gets odd positions (1, 3, 5, 7, 9)
        Assert.Equal(new List<int> { 1, 3, 5, 7, 9 }, result0);
        Assert.Equal(new List<int> { 2, 4, 6, 8, 10 }, result1);
    }

    [Fact]
    public void GetEnumerator_WithThreeWorkers_PartitionsCorrectly()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

        var dataset0 = new OptimizedIterableDataset<int>(
            CreateSampleFactory(samples),
            enableWorkerSupport: true,
            workerId: 0,
            totalWorkers: 3);

        var dataset1 = new OptimizedIterableDataset<int>(
            CreateSampleFactory(samples),
            enableWorkerSupport: true,
            workerId: 1,
            totalWorkers: 3);

        var dataset2 = new OptimizedIterableDataset<int>(
            CreateSampleFactory(samples),
            enableWorkerSupport: true,
            workerId: 2,
            totalWorkers: 3);

        // Act
        var result0 = dataset0.ToList();
        var result1 = dataset1.ToList();
        var result2 = dataset2.ToList();

        // Assert
        Assert.Equal(new List<int> { 1, 4, 7 }, result0);
        Assert.Equal(new List<int> { 2, 5, 8 }, result1);
        Assert.Equal(new List<int> { 3, 6, 9 }, result2);
    }

    [Fact]
    public void GetEnumerator_WithUnevenPartitioning_DistributesFairly()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5 };

        var dataset0 = new OptimizedIterableDataset<int>(
            CreateSampleFactory(samples),
            enableWorkerSupport: true,
            workerId: 0,
            totalWorkers: 2);

        var dataset1 = new OptimizedIterableDataset<int>(
            CreateSampleFactory(samples),
            enableWorkerSupport: true,
            workerId: 1,
            totalWorkers: 2);

        // Act
        var result0 = dataset0.ToList();
        var result1 = dataset1.ToList();

        // Assert
        Assert.Equal(new List<int> { 1, 3, 5 }, result0);
        Assert.Equal(new List<int> { 2, 4 }, result1);
    }

    [Fact]
    public void GetEnumerator_WorkersDoNotOverlapSamples()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var datasets = new List<OptimizedIterableDataset<int>>();

        for (int i = 0; i < 4; i++)
        {
            datasets.Add(new OptimizedIterableDataset<int>(
                CreateSampleFactory(samples),
                enableWorkerSupport: true,
                workerId: i,
                totalWorkers: 4));
        }

        // Act
        var allResults = datasets.SelectMany(d => d.ToList()).ToList();

        // Assert - No duplicates
        var uniqueResults = allResults.Distinct().ToList();
        Assert.Equal(allResults.Count, uniqueResults.Count);
        Assert.Equal(samples.Count, allResults.Count);
    }

    [Fact]
    public void GetEnumerator_AllSamplesCoveredByWorkers()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var datasets = new List<OptimizedIterableDataset<int>>();

        for (int i = 0; i < 3; i++)
        {
            datasets.Add(new OptimizedIterableDataset<int>(
                CreateSampleFactory(samples),
                enableWorkerSupport: true,
                workerId: i,
                totalWorkers: 3));
        }

        // Act
        var allResults = datasets.SelectMany(d => d.ToList()).ToList();

        // Assert - All samples covered
        Assert.Equal(samples, allResults.OrderBy(x => x).ToList());
    }

    [Fact]
    public void GetEnumerator_WithEmptyDataset_ReturnsEmpty()
    {
        // Arrange
        var factory = CreateSampleFactory(new List<int>());
        var dataset = new OptimizedIterableDataset<int>(factory);

        // Act
        var result = dataset.ToList();

        // Assert
        Assert.Empty(result);
    }

    [Fact]
    public void GetEnumerator_WithSingleElement_ReturnsCorrectElement()
    {
        // Arrange
        var factory = CreateSampleFactory(new List<int> { 42 });
        var dataset = new OptimizedIterableDataset<int>(factory);

        // Act
        var result = dataset.ToList();

        // Assert
        Assert.Single(result);
        Assert.Equal(42, result[0]);
    }

    /// <summary>
    /// Creates a sample factory that returns an enumerator for the given samples.
    /// </summary>
    private static Func<IEnumerator<int>> CreateSampleFactory(List<int> samples)
    {
        return () => samples.GetEnumerator();
    }

    /// <summary>
    /// Creates an enumerator that yields integers from 1 to count.
    /// </summary>
    private static IEnumerator<int> CreateSampleEnumerator(int count)
    {
        for (int i = 1; i <= count; i++)
        {
            yield return i;
        }
    }
}
