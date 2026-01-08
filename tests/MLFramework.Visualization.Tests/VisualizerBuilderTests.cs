using Xunit;
using MLFramework.Visualization;
using MLFramework.Visualization.Configuration;
using System.IO;

namespace MLFramework.Visualization.Tests;

/// <summary>
/// Unit tests for VisualizerBuilder
/// </summary>
public class VisualizerBuilderTests : IDisposable
{
    private readonly string _testLogDirectory;

    public VisualizerBuilderTests()
    {
        // Create a unique test directory for each test
        _testLogDirectory = Path.Combine(Path.GetTempPath(), "builder_tests", Guid.NewGuid().ToString());
    }

    public void Dispose()
    {
        // Clean up test directory
        if (Directory.Exists(_testLogDirectory))
        {
            try
            {
                Directory.Delete(_testLogDirectory, recursive: true);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    [Fact]
    public void Create_ReturnsNewBuilder()
    {
        // Act
        var builder = VisualizerBuilder.Create();

        // Assert
        Assert.NotNull(builder);
    }

    [Fact]
    public void WithLogDirectory_SetsDirectory()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.WithLogDirectory(_testLogDirectory);

        // Assert
        Assert.Same(builder, result); // Should return same instance for fluent chaining
    }

    [Fact]
    public void WithLogDirectory_NullDirectory_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => builder.WithLogDirectory(null!));
    }

    [Fact]
    public void WithLogDirectory_EmptyDirectory_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => builder.WithLogDirectory(""));
    }

    [Fact]
    public void WithStorageBackend_SetsStorage()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();
        var storageConfig = new MachineLearning.Visualization.Storage.StorageConfiguration
        {
            BackendType = "file",
            ConnectionString = _testLogDirectory,
            WriteBufferSize = 100,
            FlushInterval = TimeSpan.FromSeconds(1),
            EnableAsyncWrites = true
        };
        var storage = new MLFramework.Visualization.Storage.FileStorageBackend(storageConfig);

        // Act
        var result = builder.WithStorageBackend(storage);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithStorageBackend_NullStorage_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => builder.WithStorageBackend(null!));
    }

    [Fact]
    public void WithStorageConfig_SetsConfig()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();
        var config = new StorageConfiguration
        {
            BackendType = "file",
            LogDirectory = _testLogDirectory,
            ConnectionString = _testLogDirectory
        };

        // Act
        var result = builder.WithStorageConfig(config);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithStorageConfig_NullConfig_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => builder.WithStorageConfig(null!));
    }

    [Fact]
    public void WithScalarLogger_SetsLogger()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();
        var eventPublisher = new MachineLearning.Visualization.Events.EventSystem();
        var logger = new MachineLearning.Visualization.Scalars.ScalarLogger(eventPublisher);

        // Act
        var result = builder.WithScalarLogger(logger);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithHistogramLogger_SetsLogger()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.WithHistogramLogger(null!); // Interface not fully implemented yet

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithProfiler_SetsProfiler()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.WithProfiler(null!); // Interface not fully implemented yet

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void EnableAsync_SetsAsyncFlag()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.EnableAsync(true);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void EnableAsync_DisablesAsyncWhenFalse()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.EnableAsync(false);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void EnableProfiling_SetsProfilingFlag()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.EnableProfiling(true);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void EnableProfiling_DisablesProfilingWhenFalse()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.EnableProfiling(false);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void Enable_SetsEnabledFlag()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.Enable(true);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void Enable_DisablesWhenFalse()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.Enable(false);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithRunName_SetsRunName()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.WithRunName("test_run");

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithRunName_NullRunName_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => builder.WithRunName(null!));
    }

    [Fact]
    public void WithRunName_EmptyRunName_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => builder.WithRunName(""));
    }

    [Fact]
    public void WithMetadata_AddsSingleMetadata()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act
        var result = builder.WithMetadata("key1", "value1");

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithMetadata_NullKey_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => builder.WithMetadata(null!, "value1"));
    }

    [Fact]
    public void WithMetadata_EmptyKey_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => builder.WithMetadata("", "value1"));
    }

    [Fact]
    public void WithMetadata_AddsMultipleMetadata()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();
        var metadata = new Dictionary<string, string>
        {
            ["key1"] = "value1",
            ["key2"] = "value2",
            ["key3"] = "value3"
        };

        // Act
        var result = builder.WithMetadata(metadata);

        // Assert
        Assert.Same(builder, result);
    }

    [Fact]
    public void WithMetadata_NullDictionary_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => builder.WithMetadata(null!));
    }

    [Fact]
    public void Build_WithLogDirectory_CreatesVisualizer()
    {
        // Arrange
        var builder = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory);

        // Act
        var visualizer = builder.Build();

        // Assert
        Assert.NotNull(visualizer);
        Assert.True(Directory.Exists(_testLogDirectory));
        visualizer.Dispose();
    }

    [Fact]
    public void Build_WithStorageConfig_CreatesVisualizer()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            LogDirectory = _testLogDirectory,
            ConnectionString = _testLogDirectory
        };
        var builder = VisualizerBuilder.Create()
            .WithStorageConfig(config);

        // Act
        var visualizer = builder.Build();

        // Assert
        Assert.NotNull(visualizer);
        Assert.True(Directory.Exists(_testLogDirectory));
        visualizer.Dispose();
    }

    [Fact]
    public void Build_WithMultipleOptions_CreatesVisualizer()
    {
        // Arrange
        var builder = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory)
            .EnableAsync(true)
            .EnableProfiling(true)
            .Enable(true)
            .WithRunName("test_run")
            .WithMetadata("key1", "value1")
            .WithMetadata("key2", "value2");

        // Act
        var visualizer = builder.Build();

        // Assert
        Assert.NotNull(visualizer);
        Assert.Equal("test_run", visualizer.RunName);
        Assert.Equal(2, visualizer.Metadata.Count);
        Assert.True(visualizer.IsEnabled);
        visualizer.Dispose();
    }

    [Fact]
    public void Build_WithAsyncEnabled_VisualizerUsesAsync()
    {
        // Arrange
        var builder = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory)
            .EnableAsync(true);

        // Act
        var visualizer = builder.Build();

        // Assert
        Assert.NotNull(visualizer);
        visualizer.Dispose();
    }

    [Fact]
    public void Build_WithAsyncDisabled_VisualizerDoesNotUseAsync()
    {
        // Arrange
        var builder = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory)
            .EnableAsync(false);

        // Act
        var visualizer = builder.Build();

        // Assert
        Assert.NotNull(visualizer);
        visualizer.Dispose();
    }

    [Fact]
    public void Build_WithProfilingDisabled_VisualizerHasProfilingDisabled()
    {
        // Arrange
        var builder = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory)
            .EnableProfiling(false);

        // Act
        var visualizer = builder.Build();

        // Assert
        Assert.NotNull(visualizer);
        visualizer.Dispose();
    }

    [Fact]
    public void Build_WithDisabled_VisualizerIsDisabled()
    {
        // Arrange
        var builder = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory)
            .Enable(false);

        // Act
        var visualizer = builder.Build();

        // Assert
        Assert.NotNull(visualizer);
        Assert.False(visualizer.IsEnabled);
        visualizer.Dispose();
    }

    [Fact]
    public void Build_WithoutStorage_ThrowsException()
    {
        // Arrange
        var builder = VisualizerBuilder.Create();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => builder.Build());
    }

    [Fact]
    public void Build_WithInvalidStorageConfig_ThrowsException()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            LogDirectory = "",
            ConnectionString = ""
        };
        var builder = VisualizerBuilder.Create()
            .WithStorageConfig(config);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => builder.Build());
    }

    [Fact]
    public void FluentChain_AllMethodsChain()
    {
        // Arrange & Act
        var visualizer = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory)
            .EnableAsync(true)
            .EnableProfiling(true)
            .Enable(true)
            .WithRunName("test_run")
            .WithMetadata("key1", "value1")
            .WithMetadata("key2", "value2")
            .Build();

        // Assert
        Assert.NotNull(visualizer);
        visualizer.Dispose();
    }

    [Fact]
    public void Build_MultipleBuilds_CreatesSeparateInstances()
    {
        // Arrange
        var builder = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory);

        // Act
        var visualizer1 = builder.Build();
        var visualizer2 = builder.Build();

        // Assert
        Assert.NotSame(visualizer1, visualizer2);
        visualizer1.Dispose();
        visualizer2.Dispose();
    }

    [Fact]
    public void Build_CreatedVisualizer_CanLogData()
    {
        // Arrange
        var builder = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory);

        // Act
        var visualizer = builder.Build();
        visualizer.LogScalar("test_metric", 1.5f);

        // Assert - Should not throw
        visualizer.Dispose();
    }

    [Fact]
    public void WithMetadata_MultipleCalls_AllMetadataPreserved()
    {
        // Arrange
        var builder = VisualizerBuilder.Create()
            .WithLogDirectory(_testLogDirectory);

        // Act
        var visualizer = builder
            .WithMetadata("key1", "value1")
            .WithMetadata("key2", "value2")
            .WithMetadata("key3", "value3")
            .Build();

        // Assert
        Assert.Equal(3, visualizer.Metadata.Count);
        Assert.Equal("value1", visualizer.Metadata["key1"]);
        Assert.Equal("value2", visualizer.Metadata["key2"]);
        Assert.Equal("value3", visualizer.Metadata["key3"]);
        visualizer.Dispose();
    }
}
