using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Distributed;
using MLFramework.Distributed;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Checkpointing.Distributed;

/// <summary>
/// Tests for DistributedMemoryStats
/// </summary>
public class DistributedMemoryStatsTests
{
    [Fact]
    public void ToString_GeneratesCorrectString()
    {
        // Arrange
        var stats = new DistributedMemoryStats
        {
            TotalCurrentMemoryUsed = 1024,
            TotalPeakMemoryUsed = 2048,
            AverageMemoryPerRank = 512,
            MaxMemoryUsed = 1024,
            MinMemoryUsed = 256,
            TotalCheckpointCount = 10,
            PerRankMemoryUsed = new List<long> { 512, 512 },
            PerRankCheckpointCount = new List<int> { 5, 5 },
            Timestamp = DateTime.UtcNow
        };

        // Act
        var result = stats.ToString();

        // Assert
        Assert.Contains("Distributed Memory Statistics:", result);
        Assert.Contains("1.00KB", result);
        Assert.Contains("2.00KB", result);
        Assert.Contains("Total Checkpoints: 10", result);
    }

    [Fact]
    public void FormatBytes_FormatsBytesCorrectly()
    {
        // Arrange
        var stats = new DistributedMemoryStats
        {
            TotalCurrentMemoryUsed = 1024,
            PerRankMemoryUsed = new List<long> { 512, 2048, 1048576 }
        };

        // Act
        var result = stats.ToString();

        // Assert
        Assert.Contains("512.00B", result);
        Assert.Contains("2.00KB", result);
        Assert.Contains("1.00MB", result);
    }

    [Fact]
    public void AllProperties_AreSetCorrectly()
    {
        // Arrange & Act
        var stats = new DistributedMemoryStats
        {
            TotalCurrentMemoryUsed = 1000,
            TotalPeakMemoryUsed = 2000,
            AverageMemoryPerRank = 500,
            MaxMemoryUsed = 1000,
            MinMemoryUsed = 100,
            TotalCheckpointCount = 5,
            PerRankMemoryUsed = new List<long> { 100, 200, 300 },
            PerRankCheckpointCount = new List<int> { 1, 2, 2 },
            Timestamp = new DateTime(2024, 1, 1, 12, 0, 0)
        };

        // Assert
        Assert.Equal(1000, stats.TotalCurrentMemoryUsed);
        Assert.Equal(2000, stats.TotalPeakMemoryUsed);
        Assert.Equal(500, stats.AverageMemoryPerRank);
        Assert.Equal(1000, stats.MaxMemoryUsed);
        Assert.Equal(100, stats.MinMemoryUsed);
        Assert.Equal(5, stats.TotalCheckpointCount);
        Assert.Equal(3, stats.PerRankMemoryUsed.Count);
        Assert.Equal(3, stats.PerRankCheckpointCount.Count);
        Assert.Equal(new DateTime(2024, 1, 1, 12, 0, 0), stats.Timestamp);
    }
}
