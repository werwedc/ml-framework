using Xunit;
using MachineLearning.Visualization.Storage;
using MachineLearning.Visualization.Events;
using MLFramework.Visualization.Events;

namespace MLFramework.Visualization.Tests.Storage;

/// <summary>
/// Simple mock backend for testing factory
/// </summary>
public class SimpleMockBackend : IStorageBackend
{
    private bool _isInitialized;
    private long _eventCount;

    public bool IsInitialized => _isInitialized;
    public long EventCount => _eventCount;

    public void Initialize(string connectionString)
    {
        if (string.IsNullOrWhiteSpace(connectionString))
        {
            throw new ArgumentException("Connection string cannot be null or empty", nameof(connectionString));
        }
        _isInitialized = true;
    }

    public void Shutdown()
    {
        _isInitialized = false;
    }

    public void StoreEvent(Event eventData)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Backend is not initialized");
        }
        _eventCount++;
    }

    public Task StoreEventAsync(Event eventData)
    {
        StoreEvent(eventData);
        return Task.CompletedTask;
    }

    public void StoreEvents(IEnumerable<Event> events)
    {
        if (events == null)
        {
            throw new ArgumentNullException(nameof(events));
        }
        foreach (var ev in events)
        {
            StoreEvent(ev);
        }
    }

    public Task StoreEventsAsync(IEnumerable<Event> events)
    {
        StoreEvents(events);
        return Task.CompletedTask;
    }

    public IEnumerable<Event> GetEvents(long startStep, long endStep)
    {
        return Enumerable.Empty<Event>();
    }

    public Task<IEnumerable<Event>> GetEventsAsync(long startStep, long endStep)
    {
        return Task.FromResult(Enumerable.Empty<Event>());
    }

    public void Flush()
    {
        // No-op for simple mock
    }

    public Task FlushAsync()
    {
        return Task.CompletedTask;
    }

    public void Clear()
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Backend is not initialized");
        }
        _eventCount = 0;
    }

    public void Dispose()
    {
        Shutdown();
    }
}

/// <summary>
/// Unit tests for StorageBackendFactory
/// </summary>
public class StorageBackendFactoryTests
{
    [Fact]
    public void Constructor_CreatesFactory()
    {
        // Act
        var factory = new StorageBackendFactory();

        // Assert
        Assert.NotNull(factory);
    }

    [Fact]
    public void RegisterBackend_WithValidType_RegistersBackend()
    {
        // Arrange
        var factory = new StorageBackendFactory();

        // Act
        factory.RegisterBackend<SimpleMockBackend>("mock");

        // Assert
        Assert.True(factory.IsBackendRegistered("mock"));
    }

    [Fact]
    public void RegisterBackend_WithNullBackendType_ThrowsArgumentException()
    {
        // Arrange
        var factory = new StorageBackendFactory();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => factory.RegisterBackend<SimpleMockBackend>(null!));
    }

    [Fact]
    public void RegisterBackend_WithEmptyBackendType_ThrowsArgumentException()
    {
        // Arrange
        var factory = new StorageBackendFactory();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => factory.RegisterBackend<SimpleMockBackend>(""));
    }

    [Fact]
    public void RegisterBackendFactory_WithValidFactory_RegistersBackend()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        Func<IStorageBackend> backendFactory = () => new SimpleMockBackend();

        // Act
        factory.RegisterBackendFactory("mock", backendFactory);

        // Assert
        Assert.True(factory.IsBackendRegistered("mock"));
    }

    [Fact]
    public void RegisterBackendFactory_WithNullBackendType_ThrowsArgumentException()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        Func<IStorageBackend> backendFactory = () => new SimpleMockBackend();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => factory.RegisterBackendFactory(null!, backendFactory));
    }

    [Fact]
    public void RegisterBackendFactory_WithNullFactory_ThrowsArgumentNullException()
    {
        // Arrange
        var factory = new StorageBackendFactory();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => factory.RegisterBackendFactory("mock", null!));
    }

    [Fact]
    public void CreateBackend_WithRegisteredType_CreatesBackend()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("mock");

        // Act
        var backend = factory.CreateBackend("mock", "test-connection");

        // Assert
        Assert.NotNull(backend);
        Assert.True(backend.IsInitialized);
        backend.Dispose();
    }

    [Fact]
    public void CreateBackend_WithRegisteredFactory_CreatesBackend()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        Func<IStorageBackend> backendFactory = () => new SimpleMockBackend();
        factory.RegisterBackendFactory("mock", backendFactory);

        // Act
        var backend = factory.CreateBackend("mock", "test-connection");

        // Assert
        Assert.NotNull(backend);
        Assert.True(backend.IsInitialized);
        backend.Dispose();
    }

    [Fact]
    public void CreateBackend_WithNullBackendType_ThrowsArgumentException()
    {
        // Arrange
        var factory = new StorageBackendFactory();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => factory.CreateBackend(null!, "test-connection"));
    }

    [Fact]
    public void CreateBackend_WithEmptyBackendType_ThrowsArgumentException()
    {
        // Arrange
        var factory = new StorageBackendFactory();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => factory.CreateBackend("", "test-connection"));
    }

    [Fact]
    public void CreateBackend_WithUnregisteredType_ThrowsInvalidOperationException()
    {
        // Arrange
        var factory = new StorageBackendFactory();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => factory.CreateBackend("unknown", "test-connection"));
    }

    [Fact]
    public void CreateBackend_WithNullConnectionString_ThrowsException()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("mock");

        // Act & Assert
        Assert.Throws<ArgumentException>(() => factory.CreateBackend("mock", null!));
    }

    [Fact]
    public void CreateBackend_CaseInsensitive_MatchesBackend()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("Mock");

        // Act
        var backend = factory.CreateBackend("MOCK", "test-connection");

        // Assert
        Assert.NotNull(backend);
        Assert.True(backend.IsInitialized);
        backend.Dispose();
    }

    [Fact]
    public void GetAvailableBackendTypes_WithNoRegisteredTypes_ReturnsEmpty()
    {
        // Arrange
        var factory = new StorageBackendFactory();

        // Act
        var types = factory.GetAvailableBackendTypes();

        // Assert
        Assert.Empty(types);
    }

    [Fact]
    public void GetAvailableBackendTypes_WithRegisteredTypes_ReturnsTypes()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("mock1");
        factory.RegisterBackend<SimpleMockBackend>("mock2");
        Func<IStorageBackend> backendFactory = () => new SimpleMockBackend();
        factory.RegisterBackendFactory("mock3", backendFactory);

        // Act
        var types = factory.GetAvailableBackendTypes();

        // Assert
        Assert.Equal(3, types.Count());
        Assert.Contains("mock1", types);
        Assert.Contains("mock2", types);
        Assert.Contains("mock3", types);
    }

    [Fact]
    public void GetAvailableBackendTypes_ReturnsCaseInsensitiveDistinct()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("mock");
        factory.RegisterBackend<SimpleMockBackend>("MOCK");

        // Act
        var types = factory.GetAvailableBackendTypes();

        // Assert
        Assert.Single(types);
    }

    [Fact]
    public void IsBackendRegistered_WithUnregisteredType_ReturnsFalse()
    {
        // Arrange
        var factory = new StorageBackendFactory();

        // Act
        var result = factory.IsBackendRegistered("unknown");

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsBackendRegistered_WithRegisteredType_ReturnsTrue()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("mock");

        // Act
        var result = factory.IsBackendRegistered("mock");

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBackendRegistered_CaseInsensitive_ReturnsTrue()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("Mock");

        // Act
        var result = factory.IsBackendRegistered("mock");

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsBackendRegistered_WithNullType_ReturnsFalse()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("mock");

        // Act
        var result = factory.IsBackendRegistered(null);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsBackendRegistered_WithEmptyType_ReturnsFalse()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("mock");

        // Act
        var result = factory.IsBackendRegistered("");

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void FactoryCanOverrideRegisteredType()
    {
        // Arrange
        var factory = new StorageBackendFactory();
        factory.RegisterBackend<SimpleMockBackend>("mock");

        // Create first backend
        var backend1 = factory.CreateBackend("mock", "connection1");
        backend1.Dispose();

        // Override with factory
        Func<IStorageBackend> newFactory = () => new SimpleMockBackend();
        factory.RegisterBackendFactory("mock", newFactory);

        // Act
        var backend2 = factory.CreateBackend("mock", "connection2");

        // Assert
        Assert.NotNull(backend2);
        Assert.True(backend2.IsInitialized);
        backend2.Dispose();
    }
}
