using MachineLearning.Visualization.Memory;
using MachineLearning.Visualization.Collection;
using MachineLearning.Visualization.Storage;

namespace MLFramework.Visualization.Tests.Memory;

/// <summary>
/// Unit tests for GCMonitor
/// </summary>
public class GCMonitorTests : IDisposable
{
    private TestStorageBackend _storage = null!;
    private TestEventCollector _eventCollector = null!;

    public GCMonitorTests()
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
    public void Constructor_ValidProfiler_CreatesMonitor()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act & Assert
        Assert.NotNull(new GCMonitor(profiler));
    }

    [Fact]
    public void Constructor_WithStorage_CreatesMonitor()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act & Assert
        Assert.NotNull(new GCMonitor(profiler, _storage));
    }

    [Fact]
    public void Constructor_WithEventCollector_CreatesMonitor()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_eventCollector);

        // Act & Assert
        Assert.NotNull(new GCMonitor(profiler, _eventCollector));
    }

    [Fact]
    public void Constructor_NullProfiler_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new GCMonitor(null!));
    }

    [Fact]
    public void Constructor_NullStorage_ThrowsException()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new GCMonitor(profiler, (IStorageBackend)null!));
    }

    [Fact]
    public void Constructor_NullEventCollector_ThrowsException()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_eventCollector);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new GCMonitor(profiler, (IEventCollector)null!));
    }

    [Fact]
    public void GetStatistics_ReturnsStatistics()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act
        var stats = monitor.GetStatistics();

        // Assert
        Assert.NotNull(stats);
        Assert.True(stats.LastGCTime <= DateTime.UtcNow);
    }

    [Fact]
    public void GetGCCount_ValidGeneration_ReturnsCount()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act
        var count = monitor.GetGCCount(0);

        // Assert
        Assert.True(count >= 0);
    }

    [Fact]
    public void GetGCCount_InvalidGeneration_ThrowsException()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => monitor.GetGCCount(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => monitor.GetGCCount(3));
    }

    [Fact]
    public void GetTotalMemory_ReturnsMemory()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act
        var memory = monitor.GetTotalMemory();

        // Assert
        Assert.True(memory >= 0);
    }

    [Fact]
    public void GetTotalMemoryWithGC_ReturnsMemory()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act
        var memory = monitor.GetTotalMemoryWithGC();

        // Assert
        Assert.True(memory >= 0);
    }

    [Fact]
    public void ForceFullCollection_CollectsMemory()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act & Assert
        Assert.Null(Record.Exception(() => monitor.ForceFullCollection()));
    }

    [Fact]
    public void ForceCollection_ValidGeneration_CollectsMemory()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act & Assert
        Assert.Null(Record.Exception(() => monitor.ForceCollection(0)));
        Assert.Null(Record.Exception(() => monitor.ForceCollection(1)));
        Assert.Null(Record.Exception(() => monitor.ForceCollection(2)));
    }

    [Fact]
    public void ForceCollection_InvalidGeneration_ThrowsException()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => monitor.ForceCollection(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => monitor.ForceCollection(3));
    }

    [Fact]
    public void IsServerGC_ReturnsValue()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act
        var isServerGC = monitor.IsServerGC;

        // Assert
        Assert.IsType<bool>(isServerGC);
    }

    [Fact]
    public void LargeObjectHeapCompaction_CanBeSet()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act & Assert
        Assert.Null(Record.Exception(() => monitor.LargeObjectHeapCompaction = true));
        Assert.Null(Record.Exception(() => monitor.LargeObjectHeapCompaction = false));
    }

    [Fact]
    public void Dispose_DisposesMonitor()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        var monitor = new GCMonitor(profiler);

        // Act & Assert
        Assert.Null(Record.Exception(() => monitor.Dispose()));
    }

    [Fact]
    public void Dispose_MultipleTimes_DoesNotThrow()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        var monitor = new GCMonitor(profiler);

        // Act & Assert
        Assert.Null(Record.Exception(() => monitor.Dispose()));
        Assert.Null(Record.Exception(() => monitor.Dispose()));
    }

    [Fact]
    public void GCStatistics_HasAllProperties()
    {
        // Arrange
        using var profiler = new MemoryProfiler(_storage);
        using var monitor = new GCMonitor(profiler);

        // Act
        var stats = monitor.GetStatistics();

        // Assert
        Assert.True(stats.TotalGCCount >= 0);
        Assert.True(stats.Gen0GCCount >= 0);
        Assert.True(stats.Gen1GCCount >= 0);
        Assert.True(stats.Gen2GCCount >= 0);
        Assert.True(stats.LastGCTime <= DateTime.UtcNow);
    }
}
