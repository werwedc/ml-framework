using Xunit;
using MachineLearning.Visualization.Storage;
using MachineLearning.Visualization.Events;
using MLFramework.Visualization.Events;

namespace MLFramework.Visualization.Tests.Storage;

/// <summary>
/// Mock storage backend for testing StorageBackendBase
/// </summary>
public class MockStorageBackend : StorageBackendBase
{
    private readonly List<Event> _storedEvents = new();

    public IReadOnlyList<Event> StoredEvents => _storedEvents.AsReadOnly();
    public int FlushCallCount { get; private set; }
    public int InitializeCallCount { get; private set; }
    public int ShutdownCallCount { get; private set; }
    public int ClearCallCount { get; private set; }

    public MockStorageBackend(StorageConfiguration configuration) : base(configuration)
    {
    }

    protected override void InitializeCore(string connectionString)
    {
        InitializeCallCount++;
        _storedEvents.Clear();
    }

    protected override void ShutdownCore()
    {
        ShutdownCallCount++;
    }

    public override IEnumerable<Event> GetEvents(long startStep, long endStep)
    {
        return _storedEvents.Where(e =>
        {
            // Simple filter based on event type
            if (e is ScalarMetricEvent sme)
            {
                return sme.Step >= startStep && sme.Step <= endStep;
            }
            return false;
        });
    }

    protected override void FlushCore(IEnumerable<Event> events)
    {
        FlushCallCount++;
        _storedEvents.AddRange(events);
    }

    protected override void ClearCore()
    {
        ClearCallCount++;
        _storedEvents.Clear();
    }
}

/// <summary>
/// Unit tests for StorageBackendBase
/// </summary>
public class StorageBackendBaseTests : IDisposable
{
    private readonly StorageConfiguration _validConfiguration;

    public StorageBackendBaseTests()
    {
        _validConfiguration = new StorageConfiguration
        {
            BackendType = "mock",
            ConnectionString = "test-connection",
            WriteBufferSize = 5,
            FlushInterval = TimeSpan.FromMilliseconds(100),
            EnableAsyncWrites = true
        };
    }

    [Fact]
    public void Constructor_WithNullConfiguration_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new MockStorageBackend(null!));
    }

    [Fact]
    public void Constructor_WithInvalidConfiguration_ThrowsInvalidOperationException()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "",
            ConnectionString = ""
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => new MockStorageBackend(config));
    }

    [Fact]
    public void Constructor_WithValidConfiguration_CreatesBackend()
    {
        // Act
        var backend = new MockStorageBackend(_validConfiguration);

        // Assert
        Assert.NotNull(backend);
        Assert.False(backend.IsInitialized);
        Assert.Equal(0, backend.EventCount);
        backend.Dispose();
    }

    [Fact]
    public void Initialize_WithValidConnectionString_InitializesBackend()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);

        // Act
        backend.Initialize("test-connection");

        // Assert
        Assert.True(backend.IsInitialized);
        Assert.Equal(1, backend.InitializeCallCount);
        backend.Dispose();
    }

    [Fact]
    public void Initialize_WhenAlreadyInitialized_ThrowsInvalidOperationException()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => backend.Initialize("test-connection"));
        backend.Dispose();
    }

    [Fact]
    public void Initialize_WithNullConnectionString_ThrowsArgumentException()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => backend.Initialize(null!));
        backend.Dispose();
    }

    [Fact]
    public void Initialize_WithEmptyConnectionString_ThrowsArgumentException()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => backend.Initialize(string.Empty));
        backend.Dispose();
    }

    [Fact]
    public void StoreEvent_WhenNotInitialized_ThrowsInvalidOperationException()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        var testEvent = new ScalarMetricEvent("test", 1.0f);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => backend.StoreEvent(testEvent));
        backend.Dispose();
    }

    [Fact]
    public void StoreEvent_WithNullEvent_ThrowsArgumentNullException()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => backend.StoreEvent(null!));
        backend.Dispose();
    }

    [Fact]
    public void StoreEvent_WithValidEvent_IncrementsEventCount()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");
        var testEvent = new ScalarMetricEvent("test", 1.0f);

        // Act
        backend.StoreEvent(testEvent);

        // Assert
        Assert.Equal(1, backend.EventCount);
        backend.Dispose();
    }

    [Fact]
    public void StoreEvent_WithMultipleEvents_IncrementsEventCount()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");

        // Act
        for (int i = 0; i < 10; i++)
        {
            backend.StoreEvent(new ScalarMetricEvent($"test{i}", i));
        }

        // Assert
        Assert.Equal(10, backend.EventCount);
        backend.Dispose();
    }

    [Fact]
    public void StoreEvent_WhenBufferSizeReached_AutoFlushes()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");

        // Act - Write exactly WriteBufferSize events
        for (int i = 0; i < _validConfiguration.WriteBufferSize; i++)
        {
            backend.StoreEvent(new ScalarMetricEvent($"test{i}", i));
        }

        // Assert
        Assert.Equal(1, backend.FlushCallCount);
        Assert.Equal(_validConfiguration.WriteBufferSize, backend.StoredEvents.Count);
        backend.Dispose();
    }

    [Fact]
    public void StoreEvents_WithMultipleEvents_StoresAllEvents()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");
        var events = Enumerable.Range(0, 5)
            .Select(i => new ScalarMetricEvent($"test{i}", i))
            .ToList();

        // Act
        backend.StoreEvents(events);

        // Assert
        Assert.Equal(5, backend.EventCount);
        backend.Dispose();
    }

    [Fact]
    public async Task StoreEventAsync_WithValidEvent_StoresEvent()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");
        var testEvent = new ScalarMetricEvent("test", 1.0f);

        // Act
        await backend.StoreEventAsync(testEvent);

        // Assert
        Assert.Equal(1, backend.EventCount);
        backend.Dispose();
    }

    [Fact]
    public async Task StoreEventsAsync_WithMultipleEvents_StoresAllEvents()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");
        var events = Enumerable.Range(0, 5)
            .Select(i => new ScalarMetricEvent($"test{i}", i))
            .ToList();

        // Act
        await backend.StoreEventsAsync(events);

        // Assert
        Assert.Equal(5, backend.EventCount);
        backend.Dispose();
    }

    [Fact]
    public void Flush_WhenNotInitialized_DoesNothing()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);

        // Act
        backend.Flush();

        // Assert
        Assert.Equal(0, backend.FlushCallCount);
        backend.Dispose();
    }

    [Fact]
    public void Flush_WhenNoEventsInBuffer_DoesNotCallFlushCore()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");

        // Act
        backend.Flush();

        // Assert
        Assert.Equal(0, backend.FlushCallCount);
        backend.Dispose();
    }

    [Fact]
    public void Flush_WithEventsInBuffer_CallsFlushCore()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");
        backend.StoreEvent(new ScalarMetricEvent("test", 1.0f));

        // Act
        backend.Flush();

        // Assert
        Assert.Equal(1, backend.FlushCallCount);
        Assert.Equal(1, backend.StoredEvents.Count);
        backend.Dispose();
    }

    [Fact]
    public async Task FlushAsync_WithEventsInBuffer_CallsFlushCore()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");
        backend.StoreEvent(new ScalarMetricEvent("test", 1.0f));

        // Act
        await backend.FlushAsync();

        // Assert
        Assert.Equal(1, backend.FlushCallCount);
        Assert.Equal(1, backend.StoredEvents.Count);
        backend.Dispose();
    }

    [Fact]
    public void Shutdown_WhenInitialized_ShutsDownBackend()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");
        backend.StoreEvent(new ScalarMetricEvent("test", 1.0f));

        // Act
        backend.Shutdown();

        // Assert
        Assert.False(backend.IsInitialized);
        Assert.Equal(1, backend.ShutdownCallCount);
        Assert.True(backend.FlushCallCount >= 1); // Should flush on shutdown
        backend.Dispose();
    }

    [Fact]
    public void Clear_WhenNotInitialized_ThrowsInvalidOperationException()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => backend.Clear());
        backend.Dispose();
    }

    [Fact]
    public void Clear_WhenInitialized_ClearsEvents()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");
        backend.StoreEvent(new ScalarMetricEvent("test", 1.0f));
        backend.Flush();

        // Act
        backend.Clear();

        // Assert
        Assert.Equal(0, backend.EventCount);
        Assert.Equal(0, backend.StoredEvents.Count);
        Assert.Equal(1, backend.ClearCallCount);
        backend.Dispose();
    }

    [Fact]
    public void Dispose_WhenCalled_FlushesAndShutdown()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");
        backend.StoreEvent(new ScalarMetricEvent("test", 1.0f));

        // Act
        backend.Dispose();

        // Assert
        Assert.Equal(1, backend.ShutdownCallCount);
        Assert.True(backend.FlushCallCount >= 1);
    }

    [Fact]
    public void GetEvents_WithValidRange_ReturnsEvents()
    {
        // Arrange
        var backend = new MockStorageBackend(_validConfiguration);
        backend.Initialize("test-connection");

        // Add events with different steps
        for (int i = 0; i < 10; i++)
        {
            var ev = new ScalarMetricEvent($"test{i}", i, i);
            backend.StoreEvent(ev);
        }
        backend.Flush();

        // Act
        var events = backend.GetEvents(3, 6).ToList();

        // Assert
        Assert.NotNull(events);
        Assert.Equal(4, events.Count); // Should get events with steps 3, 4, 5, 6
        backend.Dispose();
    }

    public void Dispose()
    {
        // Cleanup if needed
    }
}
