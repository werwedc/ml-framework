using MLFramework.Checkpointing.Profiling;

namespace MLFramework.Tests.Checkpointing.Profiling;

public class LayerProfileTests
{
    [Fact]
    public void RecordEvent_CheckpointEvent_UpdatesStatistics()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };
        var @event = new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Checkpoint,
            DurationMs = 100,
            MemoryBytes = 1024
        };

        // Act
        profile.RecordEvent(@event);

        // Assert
        Assert.Equal(1, profile.CheckpointCount);
        Assert.Equal(100, profile.TotalCheckpointTimeMs);
        Assert.Equal(1024, profile.TotalMemorySaved);
        Assert.Equal(100.0, profile.AverageCheckpointTimeMs);
    }

    [Fact]
    public void RecordEvent_RecomputeEvent_UpdatesStatistics()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };
        var @event = new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Recompute,
            DurationMs = 50,
            MemoryBytes = 0
        };

        // Act
        profile.RecordEvent(@event);

        // Assert
        Assert.Equal(1, profile.RecomputeCount);
        Assert.Equal(50, profile.TotalRecomputeTimeMs);
        Assert.Equal(50.0, profile.AverageRecomputeTimeMs);
    }

    [Fact]
    public void RecordEvent_RetrieveEvent_UpdatesStatistics()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };
        var @event = new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Retrieve,
            DurationMs = 0,
            MemoryBytes = 0
        };

        // Act
        profile.RecordEvent(@event);

        // Assert
        Assert.Equal(1, profile.CacheHitCount);
    }

    [Fact]
    public void RecordEvent_DeallocateEvent_DoesNotUpdateStatistics()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };
        var @event = new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Deallocate,
            DurationMs = 0,
            MemoryBytes = 0
        };

        // Act
        profile.RecordEvent(@event);

        // Assert
        Assert.Equal(0, profile.CheckpointCount);
        Assert.Equal(0, profile.RecomputeCount);
        Assert.Equal(0, profile.CacheHitCount);
    }

    [Fact]
    public void AverageCheckpointTimeMs_CalculatesCorrectly()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };
        profile.RecordEvent(new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Checkpoint,
            DurationMs = 100,
            MemoryBytes = 0
        });
        profile.RecordEvent(new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Checkpoint,
            DurationMs = 200,
            MemoryBytes = 0
        });

        // Assert
        Assert.Equal(2, profile.CheckpointCount);
        Assert.Equal(300, profile.TotalCheckpointTimeMs);
        Assert.Equal(150.0, profile.AverageCheckpointTimeMs);
    }

    [Fact]
    public void AverageCheckpointTimeMs_ReturnsZeroWhenNoCheckpoints()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };

        // Assert
        Assert.Equal(0, profile.CheckpointCount);
        Assert.Equal(0.0, profile.AverageCheckpointTimeMs);
    }

    [Fact]
    public void AverageRecomputeTimeMs_CalculatesCorrectly()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };
        profile.RecordEvent(new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Recompute,
            DurationMs = 50,
            MemoryBytes = 0
        });
        profile.RecordEvent(new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Recompute,
            DurationMs = 100,
            MemoryBytes = 0
        });

        // Assert
        Assert.Equal(2, profile.RecomputeCount);
        Assert.Equal(150, profile.TotalRecomputeTimeMs);
        Assert.Equal(75.0, profile.AverageRecomputeTimeMs);
    }

    [Fact]
    public void AverageRecomputeTimeMs_ReturnsZeroWhenNoRecomputations()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };

        // Assert
        Assert.Equal(0, profile.RecomputeCount);
        Assert.Equal(0.0, profile.AverageRecomputeTimeMs);
    }

    [Fact]
    public void CacheHitRate_CalculatesCorrectly()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };
        profile.RecordEvent(new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Checkpoint,
            DurationMs = 0,
            MemoryBytes = 0
        });
        profile.RecordEvent(new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Checkpoint,
            DurationMs = 0,
            MemoryBytes = 0
        });
        profile.RecordEvent(new CheckpointEvent
        {
            LayerId = "layer1",
            EventType = CheckpointEventType.Retrieve,
            DurationMs = 0,
            MemoryBytes = 0
        });

        // Assert
        Assert.Equal(2, profile.CheckpointCount);
        Assert.Equal(1, profile.CacheHitCount);
        Assert.Equal(0.5, profile.CacheHitRate);
    }

    [Fact]
    public void CacheHitRate_ReturnsZeroWhenNoCheckpoints()
    {
        // Arrange
        var profile = new LayerProfile { LayerId = "layer1" };

        // Assert
        Assert.Equal(0, profile.CheckpointCount);
        Assert.Equal(0.0, profile.CacheHitRate);
    }
}
