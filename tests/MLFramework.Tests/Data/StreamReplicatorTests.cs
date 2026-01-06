using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace MLFramework.Data.Tests;

/// <summary>
/// Unit tests for StreamReplicator.
/// </summary>
public class StreamReplicatorTests
{
    [Fact]
    public void Constructor_WithNullSourceStream_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new StreamReplicator<int>(null!, 2));
    }

    [Fact]
    public void Constructor_WithZeroReplicas_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var source = CreateSampleEnumerator(10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new StreamReplicator<int>(source, 0));
    }

    [Fact]
    public void Constructor_WithNegativeReplicas_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var source = CreateSampleEnumerator(10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new StreamReplicator<int>(source, -1));
    }

    [Fact]
    public void GetReplicaStream_WithSingleReplica_ReturnsAllSamples()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5 };
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 1);

        // Act
        var stream = replicator.GetReplicaStream(0);
        var result = ToList(stream);

        // Assert
        Assert.Equal(samples, result);
    }

    [Fact]
    public void GetReplicaStream_WithTwoReplicas_PartitionsCorrectly()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 2);

        // Act
        var stream0 = replicator.GetReplicaStream(0);
        var stream1 = replicator.GetReplicaStream(1);

        var result0 = ToList(stream0);
        var result1 = ToList(stream1);

        // Assert - Replica 0 gets even positions (0, 2, 4, 6, 8), Replica 1 gets odd positions (1, 3, 5, 7, 9)
        Assert.Equal(new List<int> { 1, 3, 5, 7, 9 }, result0);
        Assert.Equal(new List<int> { 2, 4, 6, 8, 10 }, result1);
    }

    [Fact]
    public void GetReplicaStream_WithThreeReplicas_PartitionsCorrectly()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 3);

        // Act
        var stream0 = replicator.GetReplicaStream(0);
        var stream1 = replicator.GetReplicaStream(1);
        var stream2 = replicator.GetReplicaStream(2);

        var result0 = ToList(stream0);
        var result1 = ToList(stream1);
        var result2 = ToList(stream2);

        // Assert
        Assert.Equal(new List<int> { 1, 4, 7 }, result0);
        Assert.Equal(new List<int> { 2, 5, 8 }, result1);
        Assert.Equal(new List<int> { 3, 6, 9 }, result2);
    }

    [Fact]
    public void GetReplicaStream_WithUnevenPartitioning_DistributesFairly()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5 };
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 2);

        // Act
        var stream0 = replicator.GetReplicaStream(0);
        var stream1 = replicator.GetReplicaStream(1);

        var result0 = ToList(stream0);
        var result1 = ToList(stream1);

        // Assert
        Assert.Equal(new List<int> { 1, 3, 5 }, result0);
        Assert.Equal(new List<int> { 2, 4 }, result1);
    }

    [Fact]
    public void GetReplicaStream_WithNegativeReplicaId_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var source = CreateSampleEnumerator(10);
        var replicator = new StreamReplicator<int>(source, 2);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            replicator.GetReplicaStream(-1));
    }

    [Fact]
    public void GetReplicaStream_WithReplicaIdGreaterThanNumReplicas_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var source = CreateSampleEnumerator(10);
        var replicator = new StreamReplicator<int>(source, 2);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            replicator.GetReplicaStream(2));
    }

    [Fact]
    public void GetReplicaStream_ReplicasDoNotOverlapSamples()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 4);

        // Act
        var streams = new List<IEnumerator<int>>();
        for (int i = 0; i < 4; i++)
        {
            streams.Add(replicator.GetReplicaStream(i));
        }

        var allResults = streams.SelectMany(s => ToList(s)).ToList();

        // Assert - No duplicates
        var uniqueResults = allResults.Distinct().ToList();
        Assert.Equal(allResults.Count, uniqueResults.Count);
        Assert.Equal(samples.Count, allResults.Count);
    }

    [Fact]
    public void GetReplicaStream_AllSamplesCoveredByReplicas()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 3);

        // Act
        var streams = new List<IEnumerator<int>>();
        for (int i = 0; i < 3; i++)
        {
            streams.Add(replicator.GetReplicaStream(i));
        }

        var allResults = streams.SelectMany(s => ToList(s)).ToList();

        // Assert - All samples covered
        Assert.Equal(samples, allResults.OrderBy(x => x).ToList());
    }

    [Fact]
    public void GetReplicaStream_WithEmptySource_ReturnsEmpty()
    {
        // Arrange
        var samples = new List<int>();
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 2);

        // Act
        var stream0 = replicator.GetReplicaStream(0);
        var stream1 = replicator.GetReplicaStream(1);

        var result0 = ToList(stream0);
        var result1 = ToList(stream1);

        // Assert
        Assert.Empty(result0);
        Assert.Empty(result1);
    }

    [Fact]
    public void GetReplicaStream_WithSingleElement_FirstReplicaReceivesIt()
    {
        // Arrange
        var samples = new List<int> { 42 };
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 2);

        // Act
        var stream0 = replicator.GetReplicaStream(0);
        var stream1 = replicator.GetReplicaStream(1);

        var result0 = ToList(stream0);
        var result1 = ToList(stream1);

        // Assert
        Assert.Single(result0);
        Assert.Equal(42, result0[0]);
        Assert.Empty(result1);
    }

    [Fact]
    public void ReplicaEnumerator_Reset_ThrowsNotSupportedException()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3 };
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 2);
        var stream = replicator.GetReplicaStream(0);

        // Act & Assert
        Assert.Throws<NotSupportedException>(() =>
            stream.Reset());
    }

    [Fact]
    public void GetReplicaStream_MultipleCallsForSameReplicaId_ReturnsIndependentEnumerators()
    {
        // Arrange
        var samples = new List<int> { 1, 2, 3, 4, 5 };
        var replicator = new StreamReplicator<int>(samples.GetEnumerator(), 2);

        // Act - Get multiple streams for the same replica
        var stream0a = replicator.GetReplicaStream(0);
        var stream0b = replicator.GetReplicaStream(0);

        // Consume first stream
        var result0a = ToList(stream0a);
        var result0b = ToList(stream0b);

        // Assert - Both should return empty since the source is exhausted
        Assert.Equal(new List<int> { 1, 3, 5 }, result0a);
        Assert.Empty(result0b);
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

    /// <summary>
    /// Converts an enumerator to a list.
    /// </summary>
    private static List<T> ToList<T>(IEnumerator<T> enumerator)
    {
        var result = new List<T>();
        while (enumerator.MoveNext())
        {
            result.Add(enumerator.Current);
        }
        return result;
    }
}
