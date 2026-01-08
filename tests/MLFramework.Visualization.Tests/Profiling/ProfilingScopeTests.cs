using MLFramework.Visualization.Profiling;

namespace MLFramework.Visualization.Tests.Profiling;

public class ProfilingScopeTests
{
    [Fact]
    public void Constructor_WithValidParameters_Succeeds()
    {
        // Arrange
        var profiler = new MockProfiler();
        var name = "test_operation";
        var metadata = new Dictionary<string, string>
        {
            { "key", "value" }
        };

        // Act
        var scope = new ProfilingScope(profiler, name, metadata);

        // Assert
        Assert.Equal(name, scope.Name);
        Assert.Equal(metadata, scope.Metadata);
        Assert.True(scope.StartTimestampNanoseconds > 0);
    }

    [Fact]
    public void Constructor_WithNullProfiler_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ProfilingScope(null!, "test_operation"));
    }

    [Fact]
    public void Constructor_WithNullName_ThrowsException()
    {
        // Arrange
        var profiler = new MockProfiler();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ProfilingScope(profiler, null!));
    }

    [Fact]
    public void Constructor_WithNullMetadata_CreatesEmptyMetadata()
    {
        // Arrange
        var profiler = new MockProfiler();

        // Act
        var scope = new ProfilingScope(profiler, "test_operation", null);

        // Assert
        Assert.NotNull(scope.Metadata);
        Assert.Empty(scope.Metadata);
    }

    [Fact]
    public void Dispose_CalculatesDuration()
    {
        // Arrange
        var profiler = new MockProfiler();
        var scope = new ProfilingScope(profiler, "test_operation");
        Thread.Sleep(10);

        // Act
        scope.Dispose();

        // Assert
        Assert.True(scope.DurationNanoseconds > 0);
    }

    [Fact]
    public void Dispose_IsIdempotent()
    {
        // Arrange
        var profiler = new MockProfiler();
        var scope = new ProfilingScope(profiler, "test_operation");
        Thread.Sleep(10);

        // Act
        scope.Dispose();
        var firstDuration = scope.DurationNanoseconds;
        scope.Dispose();
        var secondDuration = scope.DurationNanoseconds;

        // Assert
        Assert.Equal(firstDuration, secondDuration);
    }

    [Fact]
    public void End_CalculatesDuration()
    {
        // Arrange
        var profiler = new MockProfiler();
        var scope = new ProfilingScope(profiler, "test_operation");
        Thread.Sleep(10);

        // Act
        scope.End();

        // Assert
        Assert.True(scope.DurationNanoseconds > 0);
    }

    [Fact]
    public void End_IsIdempotent()
    {
        // Arrange
        var profiler = new MockProfiler();
        var scope = new ProfilingScope(profiler, "test_operation");
        Thread.Sleep(10);

        // Act
        scope.End();
        var firstDuration = scope.DurationNanoseconds;
        scope.End();
        var secondDuration = scope.DurationNanoseconds;

        // Assert
        Assert.Equal(firstDuration, secondDuration);
    }

    [Fact]
    public void UsingStatement_AutomaticallyDisposesScope()
    {
        // Arrange
        var profiler = new MockProfiler();
        long duration = 0;

        // Act
        using (var scope = new ProfilingScope(profiler, "test_operation"))
        {
            Thread.Sleep(10);
        }
        duration = profiler.LastRecordedDuration;

        // Assert
        Assert.True(duration > 0);
    }

    [Fact]
    public void DurationNanoseconds_BeforeEnd_ReturnsZero()
    {
        // Arrange
        var profiler = new MockProfiler();

        // Act
        var scope = new ProfilingScope(profiler, "test_operation");

        // Assert
        Assert.Equal(0, scope.DurationNanoseconds);
    }

    [Fact]
    public void StartTimestampNanoseconds_IsConsistent()
    {
        // Arrange
        var profiler = new MockProfiler();

        // Act
        var scope = new ProfilingScope(profiler, "test_operation");
        var timestamp1 = scope.StartTimestampNanoseconds;
        Thread.Sleep(1);
        var timestamp2 = scope.StartTimestampNanoseconds;

        // Assert
        Assert.Equal(timestamp1, timestamp2);
    }

    // Mock profiler for testing
    private class MockProfiler : IProfiler
    {
        public bool IsEnabled { get; set; } = true;
        public long LastRecordedDuration { get; private set; }
        public string? LastRecordedName { get; private set; }
        public Dictionary<string, string>? LastRecordedMetadata { get; private set; }

        public IProfilingScope StartProfile(string name)
        {
            return StartProfile(name, new Dictionary<string, string>());
        }

        public IProfilingScope StartProfile(string name, Dictionary<string, string> metadata)
        {
            return new ProfilingScope(this, name, metadata);
        }

        public void RecordInstant(string name)
        {
            // Mock implementation
        }

        public void RecordInstant(string name, Dictionary<string, string> metadata)
        {
            // Mock implementation
        }

        public ProfilingResult? GetResult(string name)
        {
            return null;
        }

        public Dictionary<string, ProfilingResult> GetAllResults()
        {
            return new Dictionary<string, ProfilingResult>();
        }

        public void SetParentScope(string childName, string parentName)
        {
            // Mock implementation
        }

        public void Enable()
        {
            IsEnabled = true;
        }

        public void Disable()
        {
            IsEnabled = false;
        }

        internal void RecordDuration(string name, long durationNanoseconds, Dictionary<string, string> metadata)
        {
            LastRecordedName = name;
            LastRecordedDuration = durationNanoseconds;
            LastRecordedMetadata = metadata;
        }
    }
}
