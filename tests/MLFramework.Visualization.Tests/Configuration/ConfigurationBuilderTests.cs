using Xunit;
using MLFramework.Visualization.Configuration;

namespace MLFramework.Visualization.Tests.Configuration;

public class ConfigurationBuilderTests
{
    [Fact]
    public void Build_ReturnsValidConfiguration()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.Build();

        // Assert
        Assert.NotNull(config);
        Assert.True(config.IsEnabled);
        Assert.NotNull(config.Storage);
        Assert.NotNull(config.Logging);
    }

    [Fact]
    public void WithStorageDirectory_SetsLogDirectory()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithStorageDirectory("/custom/logs").Build();

        // Assert
        Assert.Equal("/custom/logs", config.Storage.LogDirectory);
    }

    [Fact]
    public void WithStorageBackend_SetsBackendType()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithStorageBackend("database").Build();

        // Assert
        Assert.Equal("database", config.Storage.BackendType);
    }

    [Fact]
    public void WithStorage_SetsStorageConfiguration()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();
        var storage = new StorageConfiguration
        {
            BackendType = "database",
            LogDirectory = "/db/logs",
            WriteBufferSize = 500
        };

        // Act
        var config = builder.WithStorage(storage).Build();

        // Assert
        Assert.Equal("database", config.Storage.BackendType);
        Assert.Equal("/db/logs", config.Storage.LogDirectory);
        Assert.Equal(500, config.Storage.WriteBufferSize);
    }

    [Fact]
    public void WithStorage_NullInitializesNewStorage()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithStorage(null).Build();

        // Assert
        Assert.NotNull(config.Storage);
    }

    [Fact]
    public void WithLogPrefix_SetsScalarLogPrefix()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithLogPrefix("experiment1/").Build();

        // Assert
        Assert.Equal("experiment1/", config.Logging.ScalarLogPrefix);
    }

    [Fact]
    public void WithScalarLogging_EnablesDisablesScalarLogging()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithScalarLogging(false).Build();

        // Assert
        Assert.False(config.Logging.LogScalars);
    }

    [Fact]
    public void WithHistogramBinCount_SetsBinCount()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithHistogramBinCount(50).Build();

        // Assert
        Assert.Equal(50, config.Logging.HistogramBinCount);
    }

    [Fact]
    public void WithLogging_SetsLoggingConfiguration()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();
        var logging = new LoggingConfiguration
        {
            LogScalars = false,
            HistogramBinCount = 50,
            DefaultSmoothingWindow = 20
        };

        // Act
        var config = builder.WithLogging(logging).Build();

        // Assert
        Assert.False(config.Logging.LogScalars);
        Assert.Equal(50, config.Logging.HistogramBinCount);
        Assert.Equal(20, config.Logging.DefaultSmoothingWindow);
    }

    [Fact]
    public void WithLogging_NullInitializesNewLogging()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithLogging(null).Build();

        // Assert
        Assert.NotNull(config.Logging);
    }

    [Fact]
    public void EnableProfiling_EnablesDisablesProfiling()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.EnableProfiling(false).Build();

        // Assert
        Assert.False(config.Profiling.EnableProfiling);
    }

    [Fact]
    public void EnableCPUProfiling_EnablesDisablesCPUProfiling()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.EnableCPUProfiling(false).Build();

        // Assert
        Assert.False(config.Profiling.ProfileCPU);
    }

    [Fact]
    public void EnableGPUProfiling_EnablesDisablesGPUProfiling()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.EnableGPUProfiling(false).Build();

        // Assert
        Assert.False(config.Profiling.ProfileGPU);
    }

    [Fact]
    public void WithMaxStoredOperations_SetsMaxStoredOperations()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithMaxStoredOperations(50000).Build();

        // Assert
        Assert.Equal(50000, config.Profiling.MaxStoredOperations);
    }

    [Fact]
    public void WithProfiling_SetsProfilingConfiguration()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();
        var profiling = new ProfilingConfiguration
        {
            EnableProfiling = false,
            MaxStoredOperations = 50000,
            ProfileCPU = false
        };

        // Act
        var config = builder.WithProfiling(profiling).Build();

        // Assert
        Assert.False(config.Profiling.EnableProfiling);
        Assert.Equal(50000, config.Profiling.MaxStoredOperations);
        Assert.False(config.Profiling.ProfileCPU);
    }

    [Fact]
    public void WithProfiling_NullInitializesNewProfiling()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithProfiling(null).Build();

        // Assert
        Assert.NotNull(config.Profiling);
    }

    [Fact]
    public void EnableMemoryProfiling_EnablesDisablesMemoryProfiling()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.EnableMemoryProfiling(true).Build();

        // Assert
        Assert.True(config.MemoryProfiling.EnableMemoryProfiling);
    }

    [Fact]
    public void WithMemorySnapshotIntervalMs_SetsSnapshotInterval()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithMemorySnapshotIntervalMs(500).Build();

        // Assert
        Assert.Equal(500, config.MemoryProfiling.SnapshotIntervalMs);
    }

    [Fact]
    public void WithMemoryProfiling_SetsMemoryProfilingConfiguration()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();
        var memoryProfiling = new MemoryProfilingConfiguration
        {
            EnableMemoryProfiling = true,
            SnapshotIntervalMs = 500,
            CaptureStackTraces = true
        };

        // Act
        var config = builder.WithMemoryProfiling(memoryProfiling).Build();

        // Assert
        Assert.True(config.MemoryProfiling.EnableMemoryProfiling);
        Assert.Equal(500, config.MemoryProfiling.SnapshotIntervalMs);
        Assert.True(config.MemoryProfiling.CaptureStackTraces);
    }

    [Fact]
    public void WithMemoryProfiling_NullInitializesNewMemoryProfiling()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithMemoryProfiling(null).Build();

        // Assert
        Assert.NotNull(config.MemoryProfiling);
    }

    [Fact]
    public void EnableGPUTracking_EnablesDisablesGPUTracking()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.EnableGPUTracking(true).Build();

        // Assert
        Assert.True(config.GPUTracking.EnableGPUTracking);
    }

    [Fact]
    public void WithGPUSamplingIntervalMs_SetsSamplingInterval()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithGPUSamplingIntervalMs(500).Build();

        // Assert
        Assert.Equal(500, config.GPUTracking.SamplingIntervalMs);
    }

    [Fact]
    public void WithGPUTracking_SetsGPUTrackingConfiguration()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();
        var gpuTracking = new GPUTrackingConfiguration
        {
            EnableGPUTracking = true,
            SamplingIntervalMs = 500,
            TrackTemperature = false
        };

        // Act
        var config = builder.WithGPUTracking(gpuTracking).Build();

        // Assert
        Assert.True(config.GPUTracking.EnableGPUTracking);
        Assert.Equal(500, config.GPUTracking.SamplingIntervalMs);
        Assert.False(config.GPUTracking.TrackTemperature);
    }

    [Fact]
    public void WithGPUTracking_NullInitializesNewGPUTracking()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithGPUTracking(null).Build();

        // Assert
        Assert.NotNull(config.GPUTracking);
    }

    [Fact]
    public void EnableAsyncEventCollection_EnablesDisablesAsync()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.EnableAsyncEventCollection(false).Build();

        // Assert
        Assert.False(config.EventCollection.EnableAsync);
    }

    [Fact]
    public void WithEventBufferCapacity_SetsBufferCapacity()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithEventBufferCapacity(5000).Build();

        // Assert
        Assert.Equal(5000, config.EventCollection.BufferCapacity);
    }

    [Fact]
    public void WithEventBatchSize_SetsBatchSize()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithEventBatchSize(500).Build();

        // Assert
        Assert.Equal(500, config.EventCollection.BatchSize);
    }

    [Fact]
    public void WithEventCollection_SetsEventCollectionConfiguration()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();
        var eventCollection = new EventCollectionConfiguration
        {
            EnableAsync = false,
            BufferCapacity = 5000,
            BatchSize = 500
        };

        // Act
        var config = builder.WithEventCollection(eventCollection).Build();

        // Assert
        Assert.False(config.EventCollection.EnableAsync);
        Assert.Equal(5000, config.EventCollection.BufferCapacity);
        Assert.Equal(500, config.EventCollection.BatchSize);
    }

    [Fact]
    public void WithEventCollection_NullInitializesNewEventCollection()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithEventCollection(null).Build();

        // Assert
        Assert.NotNull(config.EventCollection);
    }

    [Fact]
    public void Enable_EnablesDisablesVisualization()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.Enable(false).Build();

        // Assert
        Assert.False(config.IsEnabled);
    }

    [Fact]
    public void WithVerboseLogging_EnablesDisablesVerboseLogging()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder.WithVerboseLogging(true).Build();

        // Assert
        Assert.True(config.VerboseLogging);
    }

    [Fact]
    public void FluentBuilder_AllowsChaining()
    {
        // Arrange & Act
        var config = new VisualizationConfigurationBuilder()
            .Enable(false)
            .WithVerboseLogging(true)
            .WithStorageDirectory("/custom/logs")
            .WithLogPrefix("exp1/")
            .EnableProfiling(false)
            .EnableMemoryProfiling(true)
            .EnableGPUTracking(true)
            .Build();

        // Assert
        Assert.False(config.IsEnabled);
        Assert.True(config.VerboseLogging);
        Assert.Equal("/custom/logs", config.Storage.LogDirectory);
        Assert.Equal("exp1/", config.Logging.ScalarLogPrefix);
        Assert.False(config.Profiling.EnableProfiling);
        Assert.True(config.MemoryProfiling.EnableMemoryProfiling);
        Assert.True(config.GPUTracking.EnableGPUTracking);
    }

    [Fact]
    public void BuildWithoutValidation_ReturnsConfigurationWithoutValidating()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var config = builder
            .WithStorageDirectory("") // Invalid
            .BuildWithoutValidation();

        // Assert
        Assert.NotNull(config);
        Assert.Equal("", config.Storage.LogDirectory);
    }

    [Fact]
    public void BuildWithValidation_ReturnsConfigurationAndValidationResult()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var (config, result) = builder
            .WithStorageDirectory("") // Invalid
            .BuildWithValidation();

        // Assert
        Assert.NotNull(config);
        Assert.NotNull(result);
        Assert.False(result.IsValid);
        Assert.NotEmpty(result.Errors);
    }

    [Fact]
    public void BuildWithValidation_ValidConfiguration()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act
        var (config, result) = builder
            .WithStorageDirectory("/logs")
            .BuildWithValidation();

        // Assert
        Assert.NotNull(config);
        Assert.NotNull(result);
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void Build_InvalidConfigurationThrowsException()
    {
        // Arrange
        var builder = new VisualizationConfigurationBuilder();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            builder.WithStorageDirectory("").Build());
    }
}
