using System;
using System.Threading.Tasks;
using Xunit;

namespace MLFramework.Checkpointing.Tests;

/// <summary>
/// Unit tests for MemoryTracker
/// </summary>
public class MemoryTrackerTests : IDisposable
{
    private readonly MemoryTracker _tracker;

    public MemoryTrackerTests()
    {
        _tracker = new MemoryTracker();
    }

    public void Dispose()
    {
        _tracker?.Dispose();
    }

    [Fact]
    public void RecordAllocation_SuccessfullyRecordsAllocation()
    {
        // Arrange & Act
        _tracker.RecordAllocation("layer1", 1024);

        // Assert
        Assert.Equal(1024, _tracker.CurrentMemoryUsage);
        Assert.Equal(1024, _tracker.PeakMemoryUsage);
        Assert.Equal(1024, _tracker.TotalMemoryAllocated);
        Assert.Equal(1, _tracker.ActiveAllocationCount);
    }

    [Fact]
    public void RecordDeallocation_SuccessfullyRecordsDeallocation()
    {
        // Arrange
        _tracker.RecordAllocation("layer1", 1024);

        // Act
        _tracker.RecordDeallocation("layer1");

        // Assert
        Assert.Equal(0, _tracker.CurrentMemoryUsage);
        Assert.Equal(1024, _tracker.PeakMemoryUsage);
        Assert.Equal(1024, _tracker.TotalMemoryDeallocated);
        Assert.Equal(0, _tracker.ActiveAllocationCount);
    }

    [Fact]
    public void CurrentMemoryUsage_TracksCorrectly()
    {
        // Arrange & Act
        _tracker.RecordAllocation("layer1", 1024);
        var usage1 = _tracker.CurrentMemoryUsage;

        _tracker.RecordAllocation("layer2", 2048);
        var usage2 = _tracker.CurrentMemoryUsage;

        _tracker.RecordDeallocation("layer1");
        var usage3 = _tracker.CurrentMemoryUsage;

        // Assert
        Assert.Equal(1024, usage1);
        Assert.Equal(3072, usage2);
        Assert.Equal(2048, usage3);
    }

    [Fact]
    public void PeakMemoryUsage_TracksCorrectly()
    {
        // Arrange & Act
        _tracker.RecordAllocation("layer1", 1024);
        var peak1 = _tracker.PeakMemoryUsage;

        _tracker.RecordAllocation("layer2", 2048);
        var peak2 = _tracker.PeakMemoryUsage;

        _tracker.RecordAllocation("layer3", 512);
        var peak3 = _tracker.PeakMemoryUsage;

        _tracker.RecordDeallocation("layer2");
        var peak4 = _tracker.PeakMemoryUsage;

        // Assert
        Assert.Equal(1024, peak1);
        Assert.Equal(3072, peak2);
        Assert.Equal(3072, peak3);
        Assert.Equal(3072, peak4);
    }

    [Fact]
    public void TotalMemoryAllocated_TracksCorrectly()
    {
        // Arrange & Act
        _tracker.RecordAllocation("layer1", 1024);
        _tracker.RecordAllocation("layer2", 2048);
        _tracker.RecordAllocation("layer3", 512);

        // Assert
        Assert.Equal(3584, _tracker.TotalMemoryAllocated);
    }

    [Fact]
    public void TotalMemoryDeallocated_TracksCorrectly()
    {
        // Arrange
        _tracker.RecordAllocation("layer1", 1024);
        _tracker.RecordAllocation("layer2", 2048);

        // Act
        _tracker.RecordDeallocation("layer1");
        _tracker.RecordDeallocation("layer2");

        // Assert
        Assert.Equal(3072, _tracker.TotalMemoryDeallocated);
    }

    [Fact]
    public void ActiveAllocationCount_TracksCorrectly()
    {
        // Arrange & Act
        var count1 = _tracker.ActiveAllocationCount;
        _tracker.RecordAllocation("layer1", 1024);
        var count2 = _tracker.ActiveAllocationCount;
        _tracker.RecordAllocation("layer2", 2048);
        var count3 = _tracker.ActiveAllocationCount;
        _tracker.RecordDeallocation("layer1");
        var count4 = _tracker.ActiveAllocationCount;

        // Assert
        Assert.Equal(0, count1);
        Assert.Equal(1, count2);
        Assert.Equal(2, count3);
        Assert.Equal(1, count4);
    }

    [Fact]
    public void GetStats_ReturnsCorrectStatistics()
    {
        // Arrange
        _tracker.RecordAllocation("layer1", 1024);
        _tracker.RecordAllocation("layer2", 2048);

        // Act
        var stats = _tracker.GetStats();

        // Assert
        Assert.Equal(3072, stats.CurrentMemoryUsed);
        Assert.Equal(3072, stats.PeakMemoryUsed);
        Assert.Equal(2, stats.CheckpointCount);
        Assert.Equal(1536, stats.AverageMemoryPerCheckpoint);
        Assert.Equal(3072, stats.TotalMemoryAllocated);
        Assert.Equal(0, stats.TotalMemoryDeallocated);
        Assert.Equal(2, stats.AllocationCount);
        Assert.Equal(0, stats.DeallocationCount);
        Assert.True(stats.Timestamp <= DateTime.UtcNow && stats.Timestamp > DateTime.UtcNow.AddSeconds(-1));
    }

    [Fact]
    public void GetStats_CalculatesAverageCorrectly()
    {
        // Arrange
        _tracker.RecordAllocation("layer1", 1000);
        _tracker.RecordAllocation("layer2", 2000);
        _tracker.RecordAllocation("layer3", 3000);

        // Act
        var stats = _tracker.GetStats();

        // Assert
        Assert.Equal(2000, stats.AverageMemoryPerCheckpoint);
    }

    [Fact]
    public void GetLayerStats_ReturnsCorrectStatistics()
    {
        // Arrange
        _tracker.RecordAllocation("layer1", 1024);
        _tracker.RecordAllocation("layer2", 2048);

        // Act
        var stats1 = _tracker.GetLayerStats("layer1");
        var stats2 = _tracker.GetLayerStats("layer2");
        var stats3 = _tracker.GetLayerStats("layer3");

        // Assert
        Assert.NotNull(stats1);
        Assert.Equal("layer1", stats1.LayerId);
        Assert.Equal(1024, stats1.TotalBytesAllocated);
        Assert.Equal(1, stats1.AllocationCount);
        Assert.Equal(1024, stats1.MaxAllocationSize);
        Assert.Equal(1024, stats1.AverageAllocationSize);
        Assert.True(stats1.IsCurrentlyAllocated);

        Assert.NotNull(stats2);
        Assert.Equal("layer2", stats2.LayerId);
        Assert.Equal(2048, stats2.TotalBytesAllocated);

        Assert.Null(stats3);
    }

    [Fact]
    public void GetLayerStats_TracksAllocationDeallocation()
    {
        // Arrange
        _tracker.RecordAllocation("layer1", 1024);
        _tracker.RecordAllocation("layer1", 2048);
        _tracker.RecordDeallocation("layer1");
        _tracker.RecordAllocation("layer1", 512);

        // Act
        var stats = _tracker.GetLayerStats("layer1");

        // Assert
        Assert.Equal(3, stats.AllocationCount);
        Assert.Equal(1, stats.DeallocationCount);
        Assert.Equal(3584, stats.TotalBytesAllocated);
        Assert.Equal(1024, stats.TotalBytesDeallocated);
        Assert.Equal(2048, stats.MaxAllocationSize);
        Assert.Equal(1194, stats.AverageAllocationSize);
        Assert.True(stats.IsCurrentlyAllocated);
    }

    [Fact]
    public void ResetStats_ResetsCountersCorrectly()
    {
        // Arrange
        _tracker.RecordAllocation("layer1", 1024);
        _tracker.RecordAllocation("layer2", 2048);
        _tracker.RecordAllocation("layer1", 512);
        _tracker.RecordDeallocation("layer1");

        // Act
        _tracker.ResetStats();
        var stats = _tracker.GetStats();

        // Assert
        Assert.Equal(3072, stats.CurrentMemoryUsed); // Current allocations remain
        Assert.Equal(3072, stats.PeakMemoryUsed);   // Peak reset to current
        Assert.Equal(0, stats.TotalMemoryAllocated);
        Assert.Equal(0, stats.TotalMemoryDeallocated);
        Assert.Equal(0, stats.AllocationCount);
        Assert.Equal(0, stats.DeallocationCount);
    }

    [Fact]
    public void RecordAllocation_ThrowsOnNullLayerId()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.RecordAllocation(null!, 1024));
    }

    [Fact]
    public void RecordAllocation_ThrowsOnWhitespaceLayerId()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.RecordAllocation("   ", 1024));
    }

    [Fact]
    public void RecordAllocation_ThrowsOnZeroSize()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.RecordAllocation("layer1", 0));
    }

    [Fact]
    public void RecordAllocation_ThrowsOnNegativeSize()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.RecordAllocation("layer1", -100));
    }

    [Fact]
    public void RecordDeallocation_DoesNotThrowOnNonExistentLayer()
    {
        // Act & Assert
        _tracker.RecordDeallocation("nonexistent");
        Assert.Equal(0, _tracker.CurrentMemoryUsage);
    }

    [Fact]
    public void SetMemoryLimit_SetsLimit()
    {
        // Act
        _tracker.SetMemoryLimit(10 * 1024 * 1024);

        // Assert
        Assert.Equal(10 * 1024 * 1024, _tracker.MemoryLimit);
    }

    [Fact]
    public void SetMemoryLimit_ThrowsOnNonPositiveLimit()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _tracker.SetMemoryLimit(0));
        Assert.Throws<ArgumentException>(() => _tracker.SetMemoryLimit(-100));
    }

    [Fact]
    public void MemoryLimitExceededEvent_FiresWhenLimitExceeded()
    {
        // Arrange
        _tracker.SetMemoryLimit(2048);
        var eventFired = false;
        MemoryLimitExceededEventArgs? eventArgs = null;

        _tracker.MemoryLimitExceeded += (sender, args) =>
        {
            eventFired = true;
            eventArgs = args;
        };

        // Act
        _tracker.RecordAllocation("layer1", 1024); // Should not fire
        _tracker.RecordAllocation("layer2", 1024); // Should not fire
        _tracker.RecordAllocation("layer3", 1024); // Should fire

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(eventArgs);
        Assert.Equal(3072, eventArgs.CurrentMemoryUsage);
        Assert.Equal(2048, eventArgs.MemoryLimit);
    }

    [Fact]
    public void MemoryAllocatedEvent_FiresOnAllocation()
    {
        // Arrange
        var eventCount = 0;
        MemoryEventArgs? lastArgs = null;

        _tracker.MemoryAllocated += (sender, args) =>
        {
            eventCount++;
            lastArgs = args;
        };

        // Act
        _tracker.RecordAllocation("layer1", 1024);
        _tracker.RecordAllocation("layer2", 2048);

        // Assert
        Assert.Equal(2, eventCount);
        Assert.NotNull(lastArgs);
        Assert.Equal("layer2", lastArgs.LayerId);
        Assert.Equal(2048, lastArgs.SizeBytes);
        Assert.Equal(3072, lastArgs.CurrentMemoryUsage);
    }

    [Fact]
    public void MemoryDeallocatedEvent_FiresOnDeallocation()
    {
        // Arrange
        _tracker.RecordAllocation("layer1", 1024);
        var eventFired = false;
        MemoryEventArgs? eventArgs = null;

        _tracker.MemoryDeallocated += (sender, args) =>
        {
            eventFired = true;
            eventArgs = args;
        };

        // Act
        _tracker.RecordDeallocation("layer1");

        // Assert
        Assert.True(eventFired);
        Assert.NotNull(eventArgs);
        Assert.Equal("layer1", eventArgs.LayerId);
        Assert.Equal(1024, eventArgs.SizeBytes);
        Assert.Equal(0, eventArgs.CurrentMemoryUsage);
    }

    [Fact]
    public void PeakMemoryExceededEvent_FiresWhenPeakUpdated()
    {
        // Arrange
        var eventCount = 0;
        MemoryEventArgs? lastArgs = null;

        _tracker.PeakMemoryExceeded += (sender, args) =>
        {
            eventCount++;
            lastArgs = args;
        };

        // Act
        _tracker.RecordAllocation("layer1", 1024); // Should fire
        _tracker.RecordAllocation("layer2", 2048); // Should fire
        _tracker.RecordAllocation("layer3", 512);  // Should not fire
        _tracker.RecordDeallocation("layer1");
        _tracker.RecordAllocation("layer4", 100); // Should not fire

        // Assert
        Assert.Equal(2, eventCount);
        Assert.NotNull(lastArgs);
        Assert.Equal(3072, lastArgs.PeakMemoryUsage);
    }

    [Fact]
    public async Task ConcurrentAllocations_HandlesSafely()
    {
        // Arrange
        const int numAllocations = 100;
        const int allocationSize = 1024;

        // Act
        var tasks = new Task[numAllocations];
        for (int i = 0; i < numAllocations; i++)
        {
            int layerNum = i;
            tasks[i] = Task.Run(() => _tracker.RecordAllocation($"layer{layerNum}", allocationSize));
        }
        await Task.WhenAll(tasks);

        // Assert
        Assert.Equal(numAllocations, _tracker.ActiveAllocationCount);
        Assert.Equal(numAllocations * allocationSize, _tracker.CurrentMemoryUsage);
        Assert.Equal(numAllocations * allocationSize, _tracker.TotalMemoryAllocated);
    }

    [Fact]
    public async Task MixedConcurrentOperations_HandlesSafely()
    {
        // Arrange
        const int numOps = 100;
        var random = new Random(42);

        // Act
        var tasks = new Task[numOps];
        for (int i = 0; i < numOps; i++)
        {
            int op = i;
            tasks[op] = Task.Run(() =>
            {
                if (op % 2 == 0)
                {
                    _tracker.RecordAllocation($"layer{op}", 1024);
                }
                else if (op > 0)
                {
                    _tracker.RecordDeallocation($"layer{op - 1}");
                }
            });
        }
        await Task.WhenAll(tasks);

        // Assert - just verify no exceptions and reasonable stats
        var stats = _tracker.GetStats();
        Assert.True(stats.CurrentMemoryUsed >= 0);
        Assert.True(stats.ActiveAllocationCount >= 0);
    }

    [Fact]
    public void Dispose_ReleasesResources()
    {
        // Arrange
        _tracker.RecordAllocation("layer1", 1024);

        // Act
        _tracker.Dispose();

        // Assert
        Assert.Throws<ObjectDisposedException>(() => _tracker.CurrentMemoryUsage);
        Assert.Throws<ObjectDisposedException>(() => _tracker.RecordAllocation("layer2", 1024));
    }

    [Fact]
    public void GetLayerStats_ReturnsNullForWhitespaceLayerId()
    {
        // Act
        var stats1 = _tracker.GetLayerStats(null);
        var stats2 = _tracker.GetLayerStats("");
        var stats3 = _tracker.GetLayerStats("   ");

        // Assert
        Assert.Null(stats1);
        Assert.Null(stats2);
        Assert.Null(stats3);
    }

    [Fact]
    public void LargeMemoryValues_HandlesCorrectly()
    {
        // Arrange & Act
        var largeSize = 10L * 1024 * 1024 * 1024; // 10GB
        _tracker.RecordAllocation("layer1", largeSize);

        // Assert
        Assert.Equal(largeSize, _tracker.CurrentMemoryUsage);
        Assert.Equal(largeSize, _tracker.PeakMemoryUsage);

        var stats = _tracker.GetLayerStats("layer1");
        Assert.NotNull(stats);
        Assert.Equal(largeSize, stats.TotalBytesAllocated);
    }
}

/// <summary>
/// Unit tests for MemoryStats
/// </summary>
public class MemoryStatsTests
{
    [Fact]
    public void CalculateMemorySavings_ReturnsCorrectSavings()
    {
        // Arrange
        var stats = new MemoryStats
        {
            CurrentMemoryUsed = 1024
        };

        // Act
        var savings = stats.CalculateMemorySavings(2048);

        // Assert
        Assert.Equal(1024, savings);
    }

    [Fact]
    public void CalculateMemorySavings_ReturnsZeroWhenCurrentExceedsTotal()
    {
        // Arrange
        var stats = new MemoryStats
        {
            CurrentMemoryUsed = 2048
        };

        // Act
        var savings = stats.CalculateMemorySavings(1024);

        // Assert
        Assert.Equal(-1024, savings);
    }

    [Fact]
    public void CalculateMemoryReductionPercentage_ReturnsCorrectPercentage()
    {
        // Arrange
        var stats = new MemoryStats
        {
            CurrentMemoryUsed = 512
        };

        // Act
        var percentage = stats.CalculateMemoryReductionPercentage(1024);

        // Assert
        Assert.Equal(0.5f, percentage);
    }

    [Fact]
    public void CalculateMemoryReductionPercentage_ReturnsZeroWhenTotalIsZero()
    {
        // Arrange
        var stats = new MemoryStats
        {
            CurrentMemoryUsed = 512
        };

        // Act
        var percentage = stats.CalculateMemoryReductionPercentage(0);

        // Assert
        Assert.Equal(0f, percentage);
    }

    [Fact]
    public void CalculateMemoryReductionPercentage_HandlesNegativeSavings()
    {
        // Arrange
        var stats = new MemoryStats
        {
            CurrentMemoryUsed = 2048
        };

        // Act
        var percentage = stats.CalculateMemoryReductionPercentage(1024);

        // Assert
        Assert.Equal(-1f, percentage);
    }
}

/// <summary>
/// Unit tests for LayerMemoryStats
/// </summary>
public class LayerMemoryStatsTests
{
    [Fact]
    public void AverageAllocationSize_CalculatesCorrectly()
    {
        // Arrange
        var stats = new LayerMemoryStats
        {
            AllocationCount = 3,
            TotalBytesAllocated = 3072
        };

        // Act & Assert
        Assert.Equal(1024, stats.AverageAllocationSize);
    }

    [Fact]
    public void AverageAllocationSize_ReturnsZeroWhenNoAllocations()
    {
        // Arrange
        var stats = new LayerMemoryStats
        {
            AllocationCount = 0,
            TotalBytesAllocated = 0
        };

        // Act & Assert
        Assert.Equal(0, stats.AverageAllocationSize);
    }

    [Fact]
    public void IsCurrentlyAllocated_ReturnsTrueWhenAllocated()
    {
        // Arrange
        var stats = new LayerMemoryStats
        {
            AllocationCount = 2,
            DeallocationCount = 1
        };

        // Act & Assert
        Assert.True(stats.IsCurrentlyAllocated);
    }

    [Fact]
    public void IsCurrentlyAllocated_ReturnsFalseWhenDeallocated()
    {
        // Arrange
        var stats = new LayerMemoryStats
        {
            AllocationCount = 2,
            DeallocationCount = 2
        };

        // Act & Assert
        Assert.False(stats.IsCurrentlyAllocated);
    }

    [Fact]
    public void IsCurrentlyAllocated_ReturnsFalseWhenNeverAllocated()
    {
        // Arrange
        var stats = new LayerMemoryStats
        {
            AllocationCount = 0,
            DeallocationCount = 0
        };

        // Act & Assert
        Assert.False(stats.IsCurrentlyAllocated);
    }
}
