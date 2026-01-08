using MachineLearning.Visualization.Memory;

namespace MLFramework.Visualization.Tests.Memory;

/// <summary>
/// Unit tests for MemoryStatistics
/// </summary>
public class MemoryStatisticsTests
{
    [Fact]
    public void CreateBuilder_ReturnsNewBuilder()
    {
        // Act
        var builder = MemoryStatistics.CreateBuilder();

        // Assert
        Assert.NotNull(builder);
        Assert.Equal(0, builder.CurrentUsageBytes);
    }

    [Fact]
    public void RecordAllocation_UpdatesStatistics()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordAllocation(1024, "CPU");

        // Assert
        Assert.Equal(1024, builder.CurrentUsageBytes);
    }

    [Fact]
    public void RecordMultipleAllocations_UpdatesStatistics()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordAllocation(1024, "CPU");
        builder.RecordAllocation(2048, "GPU");
        builder.RecordAllocation(4096, "Pinned");

        // Assert
        Assert.Equal(7168, builder.CurrentUsageBytes);
    }

    [Fact]
    public void RecordDeallocation_UpdatesStatistics()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();
        builder.RecordAllocation(1024, "CPU");

        // Act
        builder.RecordDeallocation(1024, "CPU");

        // Assert
        Assert.Equal(0, builder.CurrentUsageBytes);
    }

    [Fact]
    public void RecordDeallocation_WithoutAllocation_CanGoNegative()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordDeallocation(1024, "CPU");

        // Assert
        Assert.Equal(-1024, builder.CurrentUsageBytes);
    }

    [Fact]
    public void PeakUsage_TracksMaximum()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordAllocation(1024, "CPU");
        builder.RecordAllocation(2048, "CPU");
        builder.RecordDeallocation(1024, "CPU");

        var stats = builder.Build();

        // Assert
        Assert.Equal(3072, stats.PeakUsageBytes);
    }

    [Fact]
    public void PeakUsage_RemainsAtMaximum()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordAllocation(4096, "CPU");
        builder.RecordAllocation(1024, "CPU");
        builder.RecordDeallocation(2048, "CPU");
        builder.RecordDeallocation(1024, "CPU");

        var stats = builder.Build();

        // Assert
        Assert.Equal(5120, stats.PeakUsageBytes);
    }

    [Fact]
    public void RecordGCEvent_UpdatesGCStatistics()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordGCEvent(0, TimeSpan.FromMilliseconds(100));

        var stats = builder.Build();

        // Assert
        Assert.Equal(1, stats.GCCount);
        Assert.Equal(TimeSpan.FromMilliseconds(100), stats.TotalGCTime);
    }

    [Fact]
    public void RecordMultipleGCEvents_UpdatesGCStatistics()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordGCEvent(0, TimeSpan.FromMilliseconds(100));
        builder.RecordGCEvent(1, TimeSpan.FromMilliseconds(200));
        builder.RecordGCEvent(0, TimeSpan.FromMilliseconds(50));

        var stats = builder.Build();

        // Assert
        Assert.Equal(3, stats.GCCount);
        Assert.Equal(TimeSpan.FromMilliseconds(350), stats.TotalGCTime);
    }

    [Fact]
    public void GCCountByGeneration_TracksCorrectly()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordGCEvent(0, TimeSpan.FromMilliseconds(100));
        builder.RecordGCEvent(0, TimeSpan.FromMilliseconds(100));
        builder.RecordGCEvent(1, TimeSpan.FromMilliseconds(200));
        builder.RecordGCEvent(2, TimeSpan.FromMilliseconds(300));

        var stats = builder.Build();

        // Assert
        Assert.True(stats.GCCountByGeneration.TryGetValue(0, out var gen0Count));
        Assert.Equal(2, gen0Count);
        Assert.True(stats.GCCountByGeneration.TryGetValue(1, out var gen1Count));
        Assert.Equal(1, gen1Count);
        Assert.True(stats.GCCountByGeneration.TryGetValue(2, out var gen2Count));
        Assert.Equal(1, gen2Count);
    }

    [Fact]
    public void UsageByType_TracksCorrectly()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordAllocation(1024, "CPU");
        builder.RecordAllocation(2048, "GPU");
        builder.RecordAllocation(4096, "CPU");
        builder.RecordDeallocation(1024, "CPU");

        var stats = builder.Build();

        // Assert
        Assert.True(stats.UsageByType.TryGetValue("CPU", out var cpuUsage));
        Assert.Equal(4096, cpuUsage);
        Assert.True(stats.UsageByType.TryGetValue("GPU", out var gpuUsage));
        Assert.Equal(2048, gpuUsage);
    }

    [Fact]
    public void Build_ReturnsCorrectStatistics()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();
        builder.RecordAllocation(1024, "CPU");
        builder.RecordAllocation(2048, "GPU");
        builder.RecordDeallocation(512, "CPU");
        builder.RecordGCEvent(0, TimeSpan.FromMilliseconds(100));

        // Act
        var stats = builder.Build();

        // Assert
        Assert.Equal(3072, stats.TotalAllocatedBytes);
        Assert.Equal(512, stats.TotalFreedBytes);
        Assert.Equal(2560, stats.CurrentUsageBytes);
        Assert.Equal(2, stats.AllocationCount);
        Assert.Equal(1, stats.DeallocationCount);
        Assert.Equal(1536, stats.AverageAllocationSizeBytes);
        Assert.Equal(1, stats.GCCount);
        Assert.Equal(TimeSpan.FromMilliseconds(100), stats.TotalGCTime);
    }

    [Fact]
    public void AverageAllocationSize_CalculatesCorrectly()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        builder.RecordAllocation(1024, "CPU");
        builder.RecordAllocation(2048, "CPU");
        builder.RecordAllocation(4096, "CPU");

        var stats = builder.Build();

        // Assert
        Assert.Equal(2048, stats.AverageAllocationSizeBytes);
    }

    [Fact]
    public void AverageAllocationSize_WithNoAllocations_IsZero()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();

        // Act
        var stats = builder.Build();

        // Assert
        Assert.Equal(0, stats.AverageAllocationSizeBytes);
    }

    [Fact]
    public void Builder_MultipleBuilds_IndependentInstances()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();
        builder.RecordAllocation(1024, "CPU");

        // Act
        var stats1 = builder.Build();
        builder.RecordAllocation(2048, "CPU");
        var stats2 = builder.Build();

        // Assert
        Assert.Equal(1024, stats1.CurrentUsageBytes);
        Assert.Equal(3072, stats2.CurrentUsageBytes);
    }

    [Fact]
    public void CurrentUsageBytes_ThreadSafe()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();
        var tasks = new List<Task>();

        // Act
        for (int i = 0; i < 100; i++)
        {
            var task = Task.Run(() =>
            {
                builder.RecordAllocation(1024, "CPU");
                Thread.Sleep(1);
                builder.RecordDeallocation(512, "CPU");
            });
            tasks.Add(task);
        }

        Task.WaitAll(tasks.ToArray());

        var stats = builder.Build();

        // Assert
        Assert.Equal(100, stats.AllocationCount);
        Assert.Equal(100, stats.DeallocationCount);
        Assert.Equal(51200, stats.CurrentUsageBytes);
    }

    [Fact]
    public void UsageByType_ThreadSafe()
    {
        // Arrange
        var builder = MemoryStatistics.CreateBuilder();
        var tasks = new List<Task>();

        // Act
        for (int i = 0; i < 50; i++)
        {
            tasks.Add(Task.Run(() => builder.RecordAllocation(1024, "CPU")));
            tasks.Add(Task.Run(() => builder.RecordAllocation(2048, "GPU")));
        }

        Task.WaitAll(tasks.ToArray());

        var stats = builder.Build();

        // Assert
        Assert.True(stats.UsageByType.TryGetValue("CPU", out var cpuUsage));
        Assert.Equal(51200, cpuUsage);
        Assert.True(stats.UsageByType.TryGetValue("GPU", out var gpuUsage));
        Assert.Equal(102400, gpuUsage);
    }
}
