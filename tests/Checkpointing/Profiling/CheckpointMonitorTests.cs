using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Profiling;

namespace MLFramework.Tests.Checkpointing.Profiling;

public class CheckpointMonitorTests : IDisposable
{
    private CheckpointManager _checkpointManager;
    private RecomputationEngine _recomputeEngine;
    private CheckpointProfiler _profiler;
    private CheckpointMonitor _monitor;

    public CheckpointMonitorTests()
    {
        _checkpointManager = new CheckpointManager();
        _recomputeEngine = new RecomputationEngine();
        _profiler = new CheckpointProfiler(_checkpointManager, _recomputeEngine);
        _monitor = new CheckpointMonitor(_profiler, memoryThresholdBytes: 1024, recomputeThresholdMs: 100);
    }

    public void Dispose()
    {
        _monitor?.Dispose();
        _profiler?.Dispose();
        _checkpointManager?.Dispose();
        _recomputeEngine?.Dispose();
    }

    [Fact]
    public void CheckpointMonitor_RaisesMemoryThresholdExceeded()
    {
        // Arrange
        _profiler.StartProfiling();
        var memoryThresholdExceededRaised = false;
        MemoryThresholdEventArgs? args = null;

        _monitor.MemoryThresholdExceeded += (sender, e) =>
        {
            memoryThresholdExceededRaised = true;
            args = e;
        };

        // Act
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 2048);
        _profiler.RecordEvent("layer2", CheckpointEventType.Checkpoint, 100, 2048); // Total: 4096 > 1024

        // Wait for the monitor timer to trigger
        System.Threading.Thread.Sleep(1500);

        // Assert
        Assert.True(memoryThresholdExceededRaised);
        Assert.NotNull(args);
        Assert.True(args.CurrentMemoryUsage > args.Threshold);
    }

    [Fact]
    public void CheckpointMonitor_RaisesRecomputationThresholdExceeded()
    {
        // Arrange
        _profiler.StartProfiling();
        var recomputationThresholdExceededRaised = false;
        RecomputationThresholdEventArgs? args = null;

        _monitor.RecomputationThresholdExceeded += (sender, e) =>
        {
            recomputationThresholdExceededRaised = true;
            args = e;
        };

        // Act
        _profiler.RecordEvent("layer1", CheckpointEventType.Recompute, 150, 0); // 150 > 100

        // Wait for the monitor timer to trigger
        System.Threading.Thread.Sleep(1500);

        // Assert
        Assert.True(recomputationThresholdExceededRaised);
        Assert.NotNull(args);
        Assert.True(args.CurrentRecomputeTime > args.Threshold);
    }

    [Fact]
    public void CheckpointMonitor_DoesNotRaiseEventWhenBelowThreshold()
    {
        // Arrange
        _profiler.StartProfiling();
        var memoryThresholdExceededRaised = false;
        var recomputationThresholdExceededRaised = false;

        _monitor.MemoryThresholdExceeded += (sender, e) =>
        {
            memoryThresholdExceededRaised = true;
        };

        _monitor.RecomputationThresholdExceeded += (sender, e) =>
        {
            recomputationThresholdExceededRaised = true;
        };

        // Act - Record events below thresholds
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 50, 512); // 512 < 1024
        _profiler.RecordEvent("layer1", CheckpointEventType.Recompute, 50, 0); // 50 < 100

        // Wait for the monitor timer to trigger
        System.Threading.Thread.Sleep(1500);

        // Assert
        Assert.False(memoryThresholdExceededRaised);
        Assert.False(recomputationThresholdExceededRaised);
    }

    [Fact]
    public void CheckpointMonitor_ThrowsWhenConstructedWithNullProfiler()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
        {
            new CheckpointMonitor(null!);
        });
    }

    [Fact]
    public void CheckpointMonitor_DisposesCorrectly()
    {
        // Arrange
        _profiler.StartProfiling();

        // Act
        _monitor.Dispose();

        // Should not throw when trying to record after disposal
        // (The profiler should still work, just the monitor stops)
        _profiler.RecordEvent("layer1", CheckpointEventType.Checkpoint, 100, 1024);
        Assert.True(_profiler.IsProfiling);
    }

    [Fact]
    public void CheckpointMonitor_CanCreateWithCustomThresholds()
    {
        // Arrange
        var customMemoryThreshold = 10 * 1024 * 1024; // 10MB
        var customRecomputeThreshold = 5000; // 5 seconds
        var customInterval = 2000; // 2 seconds

        // Act & Assert - Should not throw
        using var customMonitor = new CheckpointMonitor(
            _profiler,
            memoryThresholdBytes: customMemoryThreshold,
            recomputeThresholdMs: customRecomputeThreshold,
            monitorIntervalMs: customInterval);

        Assert.NotNull(customMonitor);
    }
}
