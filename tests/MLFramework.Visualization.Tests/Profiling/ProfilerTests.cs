using MachineLearning.Visualization.Events;
using MLFramework.Visualization.Profiling;
using MLFramework.Visualization.Profiling.Statistics;

namespace MLFramework.Visualization.Tests.Profiling;

public class ProfilerTests : IDisposable
{
    private readonly MockEventPublisher _eventPublisher;

    public ProfilerTests()
    {
        _eventPublisher = new MockEventPublisher();
    }

    public void Dispose()
    {
        _eventPublisher?.Dispose();
    }

    [Fact]
    public void Constructor_WithEventPublisher_Succeeds()
    {
        // Act
        var profiler = new Profiler(_eventPublisher);

        // Assert
        Assert.NotNull(profiler);
        Assert.True(profiler.IsEnabled);
    }

    [Fact]
    public void StartProfile_ReturnsProfilingScope()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);

        // Act
        var scope = profiler.StartProfile("test_operation");

        // Assert
        Assert.NotNull(scope);
        Assert.Equal("test_operation", scope.Name);
    }

    [Fact]
    public void StartProfile_WithNameAndMetadata_Succeeds()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        var metadata = new Dictionary<string, string>
        {
            { "layer", "conv1" },
            { "batch_size", "32" }
        };

        // Act
        var scope = profiler.StartProfile("test_operation", metadata);

        // Assert
        Assert.NotNull(scope);
        Assert.Equal("test_operation", scope.Name);
        Assert.Equal(metadata, scope.Metadata);
    }

    [Fact]
    public void StartProfile_WhenDisabled_CreatesScope()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        profiler.Disable();

        // Act
        var scope = profiler.StartProfile("test_operation");

        // Assert
        Assert.NotNull(scope);
        Assert.Equal("test_operation", scope.Name);
    }

    [Fact]
    public void RecordInstant_PublishesEvents()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        profiler.EnableAutomatic = true;

        // Act
        profiler.RecordInstant("checkpoint");

        // Assert
        Assert.True(_eventPublisher.StartEvents.Any(e => e.Name == "checkpoint"));
        Assert.True(_eventPublisher.EndEvents.Any(e => e.Name == "checkpoint"));
    }

    [Fact]
    public void RecordInstant_WithMetadata_PublishesEvents()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        profiler.EnableAutomatic = true;
        var metadata = new Dictionary<string, string>
        {
            { "type", "validation" }
        };

        // Act
        profiler.RecordInstant("checkpoint", metadata);

        // Assert
        var startEvent = _eventPublisher.StartEvents.First(e => e.Name == "checkpoint");
        Assert.Equal(metadata, startEvent.Metadata);
    }

    [Fact]
    public void GetResult_WithNoData_ReturnsNull()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);

        // Act
        var result = profiler.GetResult("nonexistent");

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void GetResult_WithSingleOperation_ReturnsCorrectStatistics()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        profiler.EnableAutomatic = false;

        using (var scope = profiler.StartProfile("test_operation"))
        {
            Thread.Sleep(10); // Small delay
        }

        // Act
        var result = profiler.GetResult("test_operation");

        // Assert
        Assert.NotNull(result);
        Assert.Equal("test_operation", result.Name);
        Assert.Equal(1, result.Count);
        Assert.True(result.TotalDurationNanoseconds > 0);
        Assert.Equal(result.MinDurationNanoseconds, result.MaxDurationNanoseconds);
    }

    [Fact]
    public void GetResult_WithMultipleOperations_ReturnsCorrectStatistics()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        profiler.EnableAutomatic = false;

        for (int i = 0; i < 10; i++)
        {
            using (var scope = profiler.StartProfile("test_operation"))
            {
                Thread.Sleep(5);
            }
        }

        // Act
        var result = profiler.GetResult("test_operation");

        // Assert
        Assert.NotNull(result);
        Assert.Equal("test_operation", result.Name);
        Assert.Equal(10, result.Count);
        Assert.True(result.MinDurationNanoseconds <= result.AverageDurationNanoseconds);
        Assert.True(result.AverageDurationNanoseconds <= result.MaxDurationNanoseconds);
        Assert.True(result.StdDevNanoseconds >= 0);
    }

    [Fact]
    public void GetAllResults_ReturnsAllRecordedOperations()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        profiler.EnableAutomatic = false;

        using (profiler.StartProfile("operation1")) { Thread.Sleep(1); }
        using (profiler.StartProfile("operation2")) { Thread.Sleep(1); }
        using (profiler.StartProfile("operation3")) { Thread.Sleep(1); }

        // Act
        var results = profiler.GetAllResults();

        // Assert
        Assert.Equal(3, results.Count);
        Assert.True(results.ContainsKey("operation1"));
        Assert.True(results.ContainsKey("operation2"));
        Assert.True(results.ContainsKey("operation3"));
    }

    [Fact]
    public void SetParentScope_StoresParentChildRelationship()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);

        // Act
        profiler.SetParentScope("child", "parent");

        // Assert
        Assert.Equal("parent", profiler.GetParentScope("child"));
    }

    [Fact]
    public void SetParentScope_WithNullChildName_ThrowsException()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => profiler.SetParentScope(null!, "parent"));
    }

    [Fact]
    public void SetParentScope_WithNullParentName_ThrowsException()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => profiler.SetParentScope("child", null!));
    }

    [Fact]
    public void Enable_SetsIsEnabledToTrue()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        profiler.Disable();

        // Act
        profiler.Enable();

        // Assert
        Assert.True(profiler.IsEnabled);
    }

    [Fact]
    public void Disable_SetsIsEnabledToFalse()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);

        // Act
        profiler.Disable();

        // Assert
        Assert.False(profiler.IsEnabled);
    }

    [Fact]
    public void Clear_RemovesAllData()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        profiler.EnableAutomatic = false;

        using (profiler.StartProfile("operation1")) { Thread.Sleep(1); }
        profiler.SetParentScope("child", "parent");

        // Act
        profiler.Clear();

        // Assert
        Assert.Null(profiler.GetResult("operation1"));
        Assert.Null(profiler.GetParentScope("child"));
    }

    [Fact]
    public void ProfilingScope_WithLongOperation_AccuratelyTracksDuration()
    {
        // Arrange
        var profiler = new Profiler(_eventPublisher);
        profiler.EnableAutomatic = false;
        long expectedDurationMs = 100;

        // Act
        var stopwatch = Stopwatch.StartNew();
        using (var scope = profiler.StartProfile("long_operation"))
        {
            Thread.Sleep(expectedDurationMs);
        }
        stopwatch.Stop();

        // Assert
        var result = profiler.GetResult("long_operation");
        Assert.NotNull(result);
        var actualDurationMs = result.TotalDurationNanoseconds / 1_000_000;
        // Allow for some tolerance in timing
        Assert.True(Math.Abs(actualDurationMs - expectedDurationMs) < 20,
            $"Expected {expectedDurationMs}ms, got {actualDurationMs}ms");
    }

    // Mock event publisher for testing
    private class MockEventPublisher : IEventPublisher
    {
        public List<ProfilingStartEvent> StartEvents { get; } = new();
        public List<ProfilingEndEvent> EndEvents { get; } = new();

        public void Publish<T>(T eventData) where T : Event
        {
            if (eventData is ProfilingStartEvent startEvent)
            {
                StartEvents.Add(startEvent);
            }
            else if (eventData is ProfilingEndEvent endEvent)
            {
                EndEvents.Add(endEvent);
            }
        }

        public Task PublishAsync<T>(T eventData) where T : Event
        {
            Publish(eventData);
            return Task.CompletedTask;
        }

        public void Dispose()
        {
            // Nothing to dispose
        }
    }
}
