using System;
using System.Collections.Generic;
using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Extensions;
using Xunit;

namespace MLFramework.Tests.Checkpointing;

/// <summary>
/// Tests for CheckpointStatistics class
/// </summary>
public class CheckpointStatisticsTests
{
    [Fact]
    public void ToString_GeneratesCorrectString()
    {
        var stats = new CheckpointStatistics
        {
            LayerId = "layer1",
            MemoryUsed = 1024 * 1024,
            PeakMemoryUsed = 2 * 1024 * 1024,
            RecomputationCount = 5,
            RecomputationTimeMs = 100,
            IsCheckpointingEnabled = true,
            CheckpointCount = 3,
            MemorySavings = 5 * 1024 * 1024,
            MemoryReductionPercentage = 0.75f
        };

        var result = stats.ToString();

        Assert.Contains("Checkpoint Statistics:", result);
        Assert.Contains("Memory Used:", result);
        Assert.Contains("Peak Memory:", result);
        Assert.Contains("Recomputations: 5", result);
        Assert.Contains("Recomputation Time: 100ms", result);
        Assert.Contains("Checkpoint Count: 3", result);
        Assert.Contains("Enabled: True", result);
    }

    [Fact]
    public void FormatBytes_FormatsCorrectlyForVariousSizes()
    {
        var stats = new CheckpointStatistics();

        // Test bytes
        stats.MemoryUsed = 500;
        var result = stats.ToString();
        Assert.Contains("500B", result);

        // Test KB
        stats.MemoryUsed = 2 * 1024;
        result = stats.ToString();
        Assert.Contains("2.00KB", result);

        // Test MB
        stats.MemoryUsed = 2 * 1024 * 1024;
        result = stats.ToString();
        Assert.Contains("2.00MB", result);

        // Test GB
        stats.MemoryUsed = 2L * 1024 * 1024 * 1024;
        result = stats.ToString();
        Assert.Contains("2.00GB", result);
    }

    [Fact]
    public void AllProperties_AreSetCorrectly()
    {
        var stats = new CheckpointStatistics
        {
            LayerId = "test_layer",
            MemoryUsed = 123456L,
            PeakMemoryUsed = 234567L,
            RecomputationCount = 10,
            RecomputationTimeMs = 200L,
            IsCheckpointingEnabled = false,
            CheckpointCount = 5,
            MemorySavings = 345678L,
            MemoryReductionPercentage = 0.8f,
            Timestamp = new DateTime(2024, 1, 1, 12, 0, 0)
        };

        Assert.Equal("test_layer", stats.LayerId);
        Assert.Equal(123456L, stats.MemoryUsed);
        Assert.Equal(234567L, stats.PeakMemoryUsed);
        Assert.Equal(10, stats.RecomputationCount);
        Assert.Equal(200L, stats.RecomputationTimeMs);
        Assert.False(stats.IsCheckpointingEnabled);
        Assert.Equal(5, stats.CheckpointCount);
        Assert.Equal(345678L, stats.MemorySavings);
        Assert.Equal(0.8f, stats.MemoryReductionPercentage);
        Assert.Equal(new DateTime(2024, 1, 1, 12, 0, 0), stats.Timestamp);
    }
}
