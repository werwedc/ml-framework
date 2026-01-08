using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Profiling;

namespace MLFramework.Tests.Checkpointing.Profiling;

public class CheckpointProfilerTests : IDisposable
{
    private CheckpointManager _checkpointManager;
    private RecomputationEngine _recomputeEngine;
    private CheckpointProfiler _profiler;

    public CheckpointProfilerTests()
    {
        _checkpointManager = new CheckpointManager();
        _recomputeEngine = new RecomputationEngine();
        _profiler = new CheckpointProfiler(_checkpointManager, _recomputeEngine);
    }

    public void Dispose()
    {
        _profiler?.Dispose();
        _checkpointManager?.Dispose();
        _recomputeEngine?.Dispose();
    }

    [Fact]
    public void StartProfiling_StartsProfiling()
    {
        // Arrange & Act
        _profiler.StartProfiling();

        // Assert
        Assert.True(_profiler.IsProfiling);
    }

    [Fact]
    public void StopProfiling_StopsProfiling()
    {
        // Arrange
        _profiler.StartProfiling();

        // Act
        _profiler.StopProfiling();

        // Assert
        Assert.False(_profiler.IsProfiling);
    }

    [Fact]
    public void RecordEvent_RecordsEventsCorrectly()
    {
        // Arrange
        _profiler.StartProfiling();

        // Act
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);

        // Assert
        var profile = _profiler.GetLayerProfile("layer1");
        Assert.NotNull(profile);
        Assert.Equal(1, profile.CheckpointCount);
        Assert.Equal(100, profile.TotalCheckpointTimeMs);
        Assert.Equal(1024, profile.TotalMemorySaved);
    }

    [Fact]
    public void RecordEvent_DoesNotRecordWhenNotProfiling()
    {
        // Arrange
        _profiler.StartProfiling();
        _profiler.StopProfiling();

        // Act
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);

        // Assert
        var profile = _profiler.GetLayerProfile("layer1");
        Assert.Null(profile);
    }

    [Fact]
    public void GetLayerProfile_ReturnsCorrectProfile()
    {
        // Arrange
        _profiler.StartProfiling();
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);

        // Act
        var profile = _profiler.GetLayerProfile("layer1");

        // Assert
        Assert.NotNull(profile);
        Assert.Equal("layer1", profile.LayerId);
    }

    [Fact]
    public void GetLayerProfile_ReturnsNullForNonExistentLayer()
    {
        // Arrange
        _profiler.StartProfiling();

        // Act
        var profile = _profiler.GetLayerProfile("non_existent");

        // Assert
        Assert.Null(profile);
    }

    [Fact]
    public void GetAllLayerProfiles_ReturnsAllProfiles()
    {
        // Arrange
        _profiler.StartProfiling();
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);
        _profiler.RecordEvent("layer2", CheckpointEventType.Checkpoint, 200, 2048);

        // Act
        var profiles = _profiler.GetAllLayerProfiles();

        // Assert
        Assert.Equal(2, profiles.Count);
        Assert.True(profiles.ContainsKey("layer1"));
        Assert.True(profiles.ContainsKey("layer2"));
    }

    [Fact]
    public void GetSummary_ReturnsCorrectSummary()
    {
        // Arrange
        _profiler.StartProfiling();
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);
        _profiler.RecordEvent("layer1", CheckpointEventType.Recompute, 50, 0);
        _profiler.RecordEvent("layer2", CheckpointEventType.Checkpoint, 200, 2048);

        // Act
        var summary = _profiler.GetSummary();

        // Assert
        Assert.Equal(3, summary.TotalEvents);
        Assert.Equal(300, summary.TotalCheckpointTime);
        Assert.Equal(50, summary.TotalRecomputeTime);
        Assert.Equal(3072, summary.TotalMemorySaved);
    }

    [Fact]
    public void GenerateReport_GeneratesCorrectReport()
    {
        // Arrange
        _profiler.StartProfiling();
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);

        // Act
        var report = _profiler.GenerateReport();

        // Assert
        Assert.Contains("Checkpoint Profiling Summary", report);
        Assert.Contains("layer1", report);
    }

    [Fact]
    public void ExportToJson_ExportsCorrectJson()
    {
        // Arrange
        _profiler.StartProfiling();
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);

        // Act
        var json = _profiler.ExportToJson();

        // Assert
        Assert.Contains("TotalEvents", json);
        Assert.Contains("TotalCheckpointTime", json);
        Assert.Contains("layer1", json);
    }

    [Fact]
    public void Dispose_ClearsData()
    {
        // Arrange
        _profiler.StartProfiling();
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);

        // Act
        _profiler.Dispose();
        var profiles = _profiler.GetAllLayerProfiles();

        // Assert
        Assert.Empty(profiles);
    }

    [Fact]
    public void RecordMultipleEvents_AccumulatesCorrectly()
    {
        // Arrange
        _profiler.StartProfiling();

        // Act
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 150, 2048);
        _profiler.RecordEvent("layer1", CheckpointEventType.Recompute, 75, 0);
        _profiler.RecordEvent("layer1", CheckpointEventType.Retrieve, 0, 0);

        // Assert
        var profile = _profiler.GetLayerProfile("layer1");
        Assert.NotNull(profile);
        Assert.Equal(2, profile.CheckpointCount);
        Assert.Equal(250, profile.TotalCheckpointTimeMs);
        Assert.Equal(125.0, profile.AverageCheckpointTimeMs);
        Assert.Equal(1, profile.RecomputeCount);
        Assert.Equal(75, profile.TotalRecomputeTimeMs);
        Assert.Equal(1, profile.CacheHitCount);
        Assert.Equal(3072, profile.TotalMemorySaved);
    }
}
