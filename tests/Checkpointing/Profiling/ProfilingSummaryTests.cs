using MLFramework.Checkpointing.Profiling;

namespace MLFramework.Tests.Checkpointing.Profiling;

public class ProfilingSummaryTests
{
    [Fact]
    public void ToString_GeneratesCorrectString()
    {
        // Arrange
        var summary = new ProfilingSummary
        {
            StartTime = DateTime.UtcNow,
            EndTime = DateTime.UtcNow.AddMilliseconds(1000),
            Duration = 1000.0,
            TotalEvents = 10,
            TotalCheckpointTime = 500,
            TotalRecomputeTime = 250,
            TotalMemorySaved = 1024 * 1024, // 1MB
            LayerProfiles = new List<LayerProfile>
            {
                new LayerProfile
                {
                    LayerId = "layer1",
                    CheckpointCount = 5,
                    TotalCheckpointTimeMs = 250,
                    RecomputeCount = 2,
                    TotalRecomputeTimeMs = 100,
                    CacheHitCount = 3,
                    TotalMemorySaved = 512 * 1024
                }
            }
        };

        // Act
        var result = summary.ToString();

        // Assert
        Assert.Contains("Checkpoint Profiling Summary", result);
        Assert.Contains("Duration: 1000.00ms", result);
        Assert.Contains("Total Events: 10", result);
        Assert.Contains("Total Checkpoint Time: 500ms", result);
        Assert.Contains("Total Recompute Time: 250ms", result);
        Assert.Contains("layer1:", result);
        Assert.Contains("Checkpoints: 5", result);
    }

    [Fact]
    public void FormatBytes_FormatsBytesCorrectly()
    {
        // Arrange
        var summary = new ProfilingSummary();

        // Act & Assert - The FormatBytes method is private, so we test through ToString
        summary.TotalMemorySaved = 512;
        var result1 = summary.ToString();
        Assert.Contains("512B", result1);

        summary.TotalMemorySaved = 1024;
        var result2 = summary.ToString();
        Assert.Contains("1.00KB", result2);

        summary.TotalMemorySaved = 1024 * 1024;
        var result3 = summary.ToString();
        Assert.Contains("1.00MB", result3);

        summary.TotalMemorySaved = (long)(1024 * 1024 * 1024 * 1.5);
        var result4 = summary.ToString();
        Assert.Contains("1.50GB", result4);
    }

    [Fact]
    public void ToString_OrdersProfilesByMemorySaved()
    {
        // Arrange
        var summary = new ProfilingSummary
        {
            StartTime = DateTime.UtcNow,
            EndTime = DateTime.UtcNow,
            Duration = 0,
            TotalEvents = 0,
            LayerProfiles = new List<LayerProfile>
            {
                new LayerProfile { LayerId = "low_memory", TotalMemorySaved = 512 },
                new LayerProfile { LayerId = "high_memory", TotalMemorySaved = 2048 },
                new LayerProfile { LayerId = "medium_memory", TotalMemorySaved = 1024 }
            }
        };

        // Act
        var result = summary.ToString();
        var highIndex = result.IndexOf("high_memory");
        var mediumIndex = result.IndexOf("medium_memory");
        var lowIndex = result.IndexOf("low_memory");

        // Assert
        Assert.True(highIndex < mediumIndex, "High memory should come first");
        Assert.True(mediumIndex < lowIndex, "Medium memory should come second");
    }
}
