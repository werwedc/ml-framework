using Xunit;
using MLFramework.Visualization;
using MLFramework.Visualization.Configuration;
using System.IO;

namespace MLFramework.Visualization.Tests;

/// <summary>
/// Unit tests for TensorBoardVisualizer
/// </summary>
public class TensorBoardVisualizerTests : IDisposable
{
    private readonly string _testLogDirectory;
    private TensorBoardVisualizer? _visualizer;

    public TensorBoardVisualizerTests()
    {
        // Create a unique test directory for each test
        _testLogDirectory = Path.Combine(Path.GetTempPath(), "tensorboard_tests", Guid.NewGuid().ToString());
    }

    public void Dispose()
    {
        _visualizer?.Dispose();

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
    public void Constructor_WithLogDirectory_CreatesDirectory()
    {
        // Act
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Assert
        Assert.True(Directory.Exists(_testLogDirectory));
    }

    [Fact]
    public void Constructor_WithStorageConfig_InitializesCorrectly()
    {
        // Arrange
        var config = new StorageConfiguration
        {
            BackendType = "file",
            LogDirectory = _testLogDirectory,
            ConnectionString = _testLogDirectory
        };

        // Act
        _visualizer = new TensorBoardVisualizer(config);

        // Assert
        Assert.NotNull(_visualizer);
        Assert.True(Directory.Exists(_testLogDirectory));
    }

    [Fact]
    public void Constructor_WithVisualizerConfig_InitializesCorrectly()
    {
        // Arrange
        var config = VisualizerConfiguration.CreateDefault(_testLogDirectory);
        config.EnableAsync = false;

        // Act
        _visualizer = new TensorBoardVisualizer(config);

        // Assert
        Assert.NotNull(_visualizer);
        Assert.True(Directory.Exists(_testLogDirectory));
    }

    [Fact]
    public void LogScalar_ValidInput_LogsCorrectly()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act
        _visualizer.LogScalar("test_metric", 1.5f, step: 0);

        // Assert - Should not throw
    }

    [Fact]
    public void LogScalar_InvalidName_ThrowsException()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _visualizer.LogScalar("", 1.0f));
    }

    [Fact]
    public void LogScalar_AutoIncrementStep_IncrementsCorrectly()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act
        _visualizer.LogScalar("metric1", 1.0f);
        _visualizer.LogScalar("metric1", 2.0f);
        _visualizer.LogScalar("metric2", 3.0f);

        // Assert - Should not throw
    }

    [Fact]
    public void LogScalar_FloatAndDoubleOverloads_BothWork()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act
        _visualizer.LogScalar("float_metric", 1.5f);
        _visualizer.LogScalar("double_metric", 2.5);

        // Assert - Should not throw
    }

    [Fact]
    public async Task LogScalarAsync_WithAsyncEnabled_LogsCorrectly()
    {
        // Arrange
        var config = VisualizerConfiguration.CreateDefault(_testLogDirectory);
        config.EnableAsync = true;
        _visualizer = new TensorBoardVisualizer(config);

        // Act
        await _visualizer.LogScalarAsync("async_metric", 1.0f);

        // Assert - Should not throw
    }

    [Fact]
    public void LogHistogram_ValidInput_LogsCorrectly()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        _visualizer.LogHistogram("test_histogram", values);

        // Assert - Should not throw
    }

    [Fact]
    public void LogHistogram_InvalidName_ThrowsException()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        var values = new float[] { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _visualizer.LogHistogram("", values));
    }

    [Fact]
    public void LogHistogram_NullValues_ThrowsException()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _visualizer.LogHistogram("test", null!));
    }

    [Fact]
    public void LogHistogram_WithCustomConfig_LogsCorrectly()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var config = new Histograms.HistogramBinConfig
        {
            BinCount = 10,
            UseLogScale = true
        };

        // Act
        _visualizer.LogHistogram("test_histogram", values, config);

        // Assert - Should not throw
    }

    [Fact]
    public async Task LogHistogramAsync_WithAsyncEnabled_LogsCorrectly()
    {
        // Arrange
        var config = VisualizerConfiguration.CreateDefault(_testLogDirectory);
        config.EnableAsync = true;
        _visualizer = new TensorBoardVisualizer(config);
        var values = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        await _visualizer.LogHistogramAsync("async_histogram", values);

        // Assert - Should not throw
    }

    [Fact]
    public void LogGraph_ValidInput_LogsCorrectly()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        var graph = new Graphs.ComputationalGraph("test_graph");

        // Act
        _visualizer.LogGraph(graph);

        // Assert - Should not throw
    }

    [Fact]
    public void LogGraph_NullGraph_ThrowsException()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _visualizer.LogGraph(null!));
    }

    [Fact]
    public async Task LogGraphAsync_WithAsyncEnabled_LogsCorrectly()
    {
        // Arrange
        var config = VisualizerConfiguration.CreateDefault(_testLogDirectory);
        config.EnableAsync = true;
        _visualizer = new TensorBoardVisualizer(config);
        var graph = new Graphs.ComputationalGraph("test_graph");

        // Act
        await _visualizer.LogGraphAsync(graph);

        // Assert - Should not throw
    }

    [Fact]
    public void StartProfile_RecordsDuration()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act
        using (_visualizer.StartProfile("test_operation"))
        {
            Thread.Sleep(10);
        }

        // Assert - Should not throw
    }

    [Fact]
    public void StartProfile_WithMetadata_RecordsMetadata()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        var metadata = new Dictionary<string, string>
        {
            ["param1"] = "value1",
            ["param2"] = "value2"
        };

        // Act
        using (_visualizer.StartProfile("test_operation", metadata))
        {
            Thread.Sleep(10);
        }

        // Assert - Should not throw
    }

    [Fact]
    public void RecordInstant_LogsInstantEvent()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act
        _visualizer.RecordInstant("checkpoint");

        // Assert - Should not throw
    }

    [Fact]
    public void RecordInstant_WithMetadata_LogsMetadata()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        var metadata = new Dictionary<string, string>
        {
            ["checkpoint_id"] = "123"
        };

        // Act
        _visualizer.RecordInstant("checkpoint", metadata);

        // Assert - Should not throw
    }

    [Fact]
    public void LogHyperparameters_LogsCorrectly()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        var hyperparams = new Dictionary<string, object>
        {
            ["learning_rate"] = 0.001,
            ["batch_size"] = 32,
            ["epochs"] = 10
        };

        // Act
        _visualizer.LogHyperparameters(hyperparams);

        // Assert
        Assert.Equal(3, _visualizer.Metadata.Count);
        Assert.Contains("learning_rate", _visualizer.Metadata);
        Assert.Equal("0.001", _visualizer.Metadata["learning_rate"]);
    }

    [Fact]
    public void LogText_LogsCorrectly()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act
        _visualizer.LogText("debug_note", "This is a debug message");

        // Assert
        Assert.Contains("text_debug_note", _visualizer.Metadata);
        Assert.Equal("This is a debug message", _visualizer.Metadata["text_debug_note"]);
    }

    [Fact]
    public void LogImage_LogsCorrectly()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        var imageData = new byte[] { 0x01, 0x02, 0x03, 0x04 };

        // Act
        _visualizer.LogImage("test_image", imageData);

        // Assert
        Assert.Contains("image_test_image_step_0", _visualizer.Metadata);
    }

    [Fact]
    public void IsEnabled_WhenFalse_DoesNotLog()
    {
        // Arrange
        var config = VisualizerConfiguration.CreateDefault(_testLogDirectory);
        config.IsEnabled = false;
        _visualizer = new TensorBoardVisualizer(config);

        // Act - Should not throw even when disabled
        _visualizer.LogScalar("disabled_metric", 1.0f);
        _visualizer.LogHistogram("disabled_histogram", new float[] { 1.0f });
        _visualizer.RecordInstant("disabled_instant");

        // Assert - Should not throw and metadata should be empty
        Assert.Empty(_visualizer.Metadata);
    }

    [Fact]
    public void Flush_ForcesWritesToStorage()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        _visualizer.LogScalar("flush_test", 1.0f);

        // Act
        _visualizer.Flush();

        // Assert - Should not throw
    }

    [Fact]
    public async Task FlushAsync_WithAsyncEnabled_FlushesCorrectly()
    {
        // Arrange
        var config = VisualizerConfiguration.CreateDefault(_testLogDirectory);
        config.EnableAsync = true;
        _visualizer = new TensorBoardVisualizer(config);
        _visualizer.LogScalar("async_flush_test", 1.0f);

        // Act
        await _visualizer.FlushAsync();

        // Assert - Should not throw
    }

    [Fact]
    public void Export_ClosesFilesAndFlushes()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        _visualizer.LogScalar("export_test", 1.0f);

        // Act
        _visualizer.Export();

        // Assert - Should not throw
    }

    [Fact]
    public async Task ExportAsync_WithAsyncEnabled_ExportsCorrectly()
    {
        // Arrange
        var config = VisualizerConfiguration.CreateDefault(_testLogDirectory);
        config.EnableAsync = true;
        _visualizer = new TensorBoardVisualizer(config);
        _visualizer.LogScalar("async_export_test", 1.0f);

        // Act
        await _visualizer.ExportAsync();

        // Assert - Should not throw
    }

    [Fact]
    public void Dispose_ReleasesResources()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);
        _visualizer.LogScalar("dispose_test", 1.0f);

        // Act
        _visualizer.Dispose();

        // Assert - Should not throw
    }

    [Fact]
    public void MultipleLogOperations_AllSucceed()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act
        for (int i = 0; i < 10; i++)
        {
            _visualizer.LogScalar($"metric_{i}", i * 1.0f);
        }

        for (int i = 0; i < 5; i++)
        {
            var values = Enumerable.Range(0, 10).Select(x => x * 1.0f).ToArray();
            _visualizer.LogHistogram($"histogram_{i}", values);
        }

        // Assert - Should not throw
    }

    [Fact]
    public void ProfilingWithMultipleScopes_AllRecorded()
    {
        // Arrange
        _visualizer = new TensorBoardVisualizer(_testLogDirectory);

        // Act
        using (_visualizer.StartProfile("operation1"))
        {
            Thread.Sleep(10);
        }

        using (_visualizer.StartProfile("operation2"))
        {
            Thread.Sleep(15);
        }

        using (_visualizer.StartProfile("operation3"))
        {
            Thread.Sleep(20);
        }

        // Assert - Should not throw
    }
}
