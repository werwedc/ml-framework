using MachineLearning.Visualization.Memory;
using MachineLearning.Visualization.Collection;
using MachineLearning.Visualization.Storage;

namespace MLFramework.Visualization.Tests.Memory;

/// <summary>
/// Unit tests for MemoryProfiler
/// </summary>
public class MemoryProfilerTests : IDisposable
{
    private TestStorageBackend _storage = null!;
    private TestEventCollector _eventCollector = null!;

    public MemoryProfilerTests()
    {
        _storage = new TestStorageBackend();
        _eventCollector = new TestEventCollector();
    }

    public void Dispose()
    {
        _storage?.Dispose();
        _eventCollector?.Dispose();
    }

    [Fact]
    public void TrackAllocation_ValidParameters_RecordsAllocation()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act
        profiler.TrackAllocation(0x1000, 1024, "CPU");

        // Assert
        var stats = profiler.GetStatistics();
        Assert.Equal(1024, stats.TotalAllocatedBytes);
        Assert.Equal(1, stats.AllocationCount);
        Assert.Equal(1024, stats.CurrentUsageBytes);
    }

    [Fact]
    public void TrackAllocation_InvalidSize_ThrowsException()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            profiler.TrackAllocation(0x1000, -1024, "CPU"));
    }

    [Fact]
    public void TrackAllocation_InvalidType_ThrowsException()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            profiler.TrackAllocation(0x1000, 1024, ""));
    }

    [Fact]
    public void TrackDeallocation_ValidParameters_RecordsDeallocation()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        profiler.TrackAllocation(0x1000, 1024, "CPU");

        // Act
        profiler.TrackDeallocation(0x1000, 1024);

        // Assert
        var stats = profiler.GetStatistics();
        Assert.Equal(1024, stats.TotalFreedBytes);
        Assert.Equal(1, stats.DeallocationCount);
        Assert.Equal(0, stats.CurrentUsageBytes);
    }

    [Fact]
    public void TrackDeallocation_InvalidSize_ThrowsException()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            profiler.TrackDeallocation(0x1000, -1024));
    }

    [Fact]
    public void TrackDeallocation_WithoutMatchingAllocation_RecordsDeallocation()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act
        profiler.TrackDeallocation(0x1000, 1024);

        // Assert
        var stats = profiler.GetStatistics();
        Assert.Equal(1, stats.DeallocationCount);
    }

    [Fact]
    public void TrackSnapshot_RecordsCurrentState()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        profiler.TrackAllocation(0x1000, 1024, "CPU");

        // Act
        profiler.TrackSnapshot();

        // Assert
        var events = profiler.GetAllocationsSince(DateTime.MinValue);
        Assert.True(events.Count() >= 1);
    }

    [Fact]
    public void GetStatistics_ReturnsCorrectStatistics()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        profiler.TrackAllocation(0x1000, 1024, "CPU");
        profiler.TrackAllocation(0x2000, 2048, "GPU");
        profiler.TrackDeallocation(0x1000, 1024);

        // Act
        var stats = profiler.GetStatistics();

        // Assert
        Assert.Equal(3072, stats.TotalAllocatedBytes);
        Assert.Equal(1024, stats.TotalFreedBytes);
        Assert.Equal(2048, stats.CurrentUsageBytes);
        Assert.Equal(2, stats.AllocationCount);
        Assert.Equal(1, stats.DeallocationCount);
    }

    [Fact]
    public void GetStatisticsForType_ReturnsCorrectStatistics()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        profiler.TrackAllocation(0x1000, 1024, "CPU");
        profiler.TrackAllocation(0x2000, 2048, "GPU");

        // Act
        var stats = profiler.GetStatisticsForType("CPU");

        // Assert
        Assert.Equal(1024, stats.CurrentUsageBytes);
    }

    [Fact]
    public void GetEvents_ReturnsEventsInRange()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        var startTime = DateTime.UtcNow;
        profiler.TrackAllocation(0x1000, 1024, "CPU");
        Thread.Sleep(10);
        var endTime = DateTime.UtcNow;

        // Act
        var events = profiler.GetEvents(startTime.Ticks, endTime.Ticks);

        // Assert
        Assert.Single(events);
    }

    [Fact]
    public void GetAllocationsSince_ReturnsAllocationsAfterTime()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        var startTime = DateTime.UtcNow;
        Thread.Sleep(10);
        profiler.TrackAllocation(0x1000, 1024, "CPU");

        // Act
        var events = profiler.GetAllocationsSince(startTime);

        // Assert
        Assert.Single(events);
    }

    [Fact]
    public void DetectPotentialLeaks_DetectsUnfreedAllocations()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        profiler.CaptureStackTraces = true;
        profiler.TrackAllocation(0x1000, 1024, "CPU");

        // Wait for allocation to age past the threshold
        Thread.Sleep(1100);

        // Act
        var leaks = profiler.DetectPotentialLeaks();

        // Assert
        Assert.Single(leaks);
        Assert.Equal(0x1000, leaks[0].address);
        Assert.Equal(1024, leaks[0].size);
    }

    [Fact]
    public void DetectPotentialLeaks_IgnoresRecentAllocations()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        profiler.CaptureStackTraces = true;
        profiler.TrackAllocation(0x1000, 1024, "CPU");

        // Act
        var leaks = profiler.DetectPotentialLeaks();

        // Assert
        Assert.Empty(leaks);
    }

    [Fact]
    public void Enable_Disable_Enable_RespectsState()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        profiler.Disable();

        // Act
        profiler.TrackAllocation(0x1000, 1024, "CPU");
        var stats1 = profiler.GetStatistics();

        profiler.Enable();
        profiler.TrackAllocation(0x2000, 2048, "CPU");
        var stats2 = profiler.GetStatistics();

        profiler.Disable();
        profiler.TrackAllocation(0x3000, 4096, "CPU");
        var stats3 = profiler.GetStatistics();

        // Assert
        Assert.Equal(0, stats1.AllocationCount);
        Assert.Equal(1, stats2.AllocationCount);
        Assert.Equal(1, stats3.AllocationCount);
    }

    [Fact]
    public void IsEnabled_ReflectsCurrentState()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act & Assert
        Assert.True(profiler.IsEnabled);

        profiler.Disable();
        Assert.False(profiler.IsEnabled);

        profiler.Enable();
        Assert.True(profiler.IsEnabled);
    }

    [Fact]
    public void CaptureStackTraces_WhenEnabled_CapturesStackTraces()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        profiler.CaptureStackTraces = true;

        // Act
        profiler.TrackAllocation(0x1000, 1024, "CPU");

        // Wait for allocation to age past the threshold
        Thread.Sleep(1100);

        var leaks = profiler.DetectPotentialLeaks();

        // Assert
        Assert.Single(leaks);
        Assert.NotNull(leaks[0].trace);
    }

    [Fact]
    public void PeakUsage_TracksCorrectly()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act
        profiler.TrackAllocation(0x1000, 1024, "CPU");
        profiler.TrackAllocation(0x2000, 2048, "CPU");
        profiler.TrackDeallocation(0x1000, 1024);

        var stats = profiler.GetStatistics();

        // Assert
        Assert.Equal(3072, stats.PeakUsageBytes);
    }

    [Fact]
    public void AverageAllocationSize_CalculatesCorrectly()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act
        profiler.TrackAllocation(0x1000, 1024, "CPU");
        profiler.TrackAllocation(0x2000, 2048, "CPU");

        var stats = profiler.GetStatistics();

        // Assert
        Assert.Equal(1536, stats.AverageAllocationSizeBytes);
    }
}
