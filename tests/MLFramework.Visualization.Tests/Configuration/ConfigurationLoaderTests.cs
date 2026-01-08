using Xunit;
using MLFramework.Visualization.Configuration;

namespace MLFramework.Visualization.Tests.Configuration;

public class ConfigurationLoaderTests
{
    private readonly ConfigurationLoader _loader;

    public ConfigurationLoaderTests()
    {
        _loader = new ConfigurationLoader();
    }

    [Fact]
    public void Load_ReturnsDefaultConfiguration()
    {
        // Act
        var config = _loader.Load();

        // Assert
        Assert.NotNull(config);
        Assert.True(config.IsEnabled);
        Assert.NotNull(config.Storage);
        Assert.NotNull(config.Logging);
        Assert.NotNull(config.Profiling);
        Assert.NotNull(config.MemoryProfiling);
        Assert.NotNull(config.GPUTracking);
        Assert.NotNull(config.EventCollection);
    }

    [Fact]
    public void Load_DefaultsAreCorrect()
    {
        // Act
        var config = _loader.Load();

        // Assert
        Assert.Equal("file", config.Storage.BackendType);
        Assert.Equal("./logs", config.Storage.LogDirectory);
        Assert.Equal(100, config.Storage.WriteBufferSize);
        Assert.Equal(TimeSpan.FromSeconds(1), config.Storage.FlushInterval);
        Assert.True(config.Storage.EnableAsyncWrites);

        Assert.True(config.Logging.LogScalars);
        Assert.True(config.Logging.LogHistograms);
        Assert.True(config.Logging.LogGraphs);
        Assert.True(config.Logging.LogHyperparameters);
        Assert.Equal("", config.Logging.ScalarLogPrefix);
        Assert.Equal(30, config.Logging.HistogramBinCount);
        Assert.True(config.Logging.AutoSmoothScalars);
        Assert.Equal(10, config.Logging.DefaultSmoothingWindow);

        Assert.True(config.Profiling.EnableProfiling);
        Assert.True(config.Profiling.ProfileForwardPass);
        Assert.True(config.Profiling.ProfileBackwardPass);
        Assert.False(config.Profiling.ProfileOptimizerStep);
        Assert.True(config.Profiling.ProfileCPU);
        Assert.True(config.Profiling.ProfileGPU);
        Assert.Equal(10000, config.Profiling.MaxStoredOperations);

        Assert.False(config.MemoryProfiling.EnableMemoryProfiling);
        Assert.False(config.MemoryProfiling.CaptureStackTraces);
        Assert.Equal(10, config.MemoryProfiling.MaxStackTraceDepth);
        Assert.Equal(1000, config.MemoryProfiling.SnapshotIntervalMs);
        Assert.True(config.MemoryProfiling.AutoSnapshot);

        Assert.False(config.GPUTracking.EnableGPUTracking);
        Assert.Equal(1000, config.GPUTracking.SamplingIntervalMs);
        Assert.True(config.GPUTracking.TrackTemperature);
        Assert.True(config.GPUTracking.TrackPower);

        Assert.True(config.EventCollection.EnableAsync);
        Assert.Equal(1000, config.EventCollection.BufferCapacity);
        Assert.Equal(100, config.EventCollection.BatchSize);
        Assert.True(config.EventCollection.EnableBackpressure);
        Assert.Equal(10000, config.EventCollection.MaxQueueLength);
    }

    [Fact]
    public void LoadFromJson_ParsesValidJson()
    {
        // Arrange
        var json = @"{
            ""isEnabled"": false,
            ""verboseLogging"": true,
            ""storage"": {
                ""backendType"": ""database"",
                ""logDirectory"": ""/custom/logs"",
                ""writeBufferSize"": 200
            },
            ""logging"": {
                ""logScalars"": false,
                ""histogramBinCount"": 50
            }
        }";

        // Act
        var config = _loader.LoadFromJson(json);

        // Assert
        Assert.False(config.IsEnabled);
        Assert.True(config.VerboseLogging);
        Assert.Equal("database", config.Storage.BackendType);
        Assert.Equal("/custom/logs", config.Storage.LogDirectory);
        Assert.Equal(200, config.Storage.WriteBufferSize);
        Assert.False(config.Logging.LogScalars);
        Assert.Equal(50, config.Logging.HistogramBinCount);
    }

    [Fact]
    public void LoadFromJson_EmptyStringReturnsDefaults()
    {
        // Arrange
        var json = "";

        // Act
        var config = _loader.LoadFromJson(json);

        // Assert
        Assert.NotNull(config);
        Assert.True(config.IsEnabled);
        Assert.Equal("file", config.Storage.BackendType);
    }

    [Fact]
    public void LoadFromJson_NullStringReturnsDefaults()
    {
        // Arrange
        string json = null;

        // Act
        var config = _loader.LoadFromJson(json);

        // Assert
        Assert.NotNull(config);
        Assert.True(config.IsEnabled);
        Assert.Equal("file", config.Storage.BackendType);
    }

    [Fact]
    public void LoadFromJson_InvalidJsonThrowsException()
    {
        // Arrange
        var json = "{ invalid json }";

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _loader.LoadFromJson(json));
    }

    [Fact]
    public void LoadFromJson_PartialConfigurationUsesDefaults()
    {
        // Arrange
        var json = @"{
            ""isEnabled"": false,
            ""storage"": {
                ""logDirectory"": ""/custom""
            }
        }";

        // Act
        var config = _loader.LoadFromJson(json);

        // Assert
        Assert.False(config.IsEnabled);
        Assert.Equal("/custom", config.Storage.LogDirectory);
        Assert.Equal("file", config.Storage.BackendType); // Default
        Assert.Equal(100, config.Storage.WriteBufferSize); // Default
        Assert.True(config.Logging.LogScalars); // Default
    }

    [Fact]
    public void LoadFromJson_MissingNestedObjectsAreInitialized()
    {
        // Arrange
        var json = @"{
            ""isEnabled"": true
        }";

        // Act
        var config = _loader.LoadFromJson(json);

        // Assert
        Assert.NotNull(config.Storage);
        Assert.NotNull(config.Logging);
        Assert.NotNull(config.Profiling);
        Assert.NotNull(config.MemoryProfiling);
        Assert.NotNull(config.GPUTracking);
        Assert.NotNull(config.EventCollection);
    }

    [Fact]
    public void LoadFromFile_ValidFile()
    {
        // Arrange
        var tempFile = Path.GetTempFileName();
        var json = @"{
            ""isEnabled"": false,
            ""storage"": {
                ""logDirectory"": ""/test/logs""
            }
        }";
        File.WriteAllText(tempFile, json);

        try
        {
            // Act
            var config = _loader.LoadFromFile(tempFile);

            // Assert
            Assert.False(config.IsEnabled);
            Assert.Equal("/test/logs", config.Storage.LogDirectory);
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void LoadFromFile_NonExistentFileThrowsException()
    {
        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => _loader.LoadFromFile("/nonexistent/file.json"));
    }

    [Fact]
    public void SaveToFile_CreatesValidJson()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            IsEnabled = false,
            Storage = new StorageConfiguration
            {
                LogDirectory = "/custom/logs",
                WriteBufferSize = 250
            }
        };
        var tempFile = Path.GetTempFileName();

        try
        {
            // Act
            _loader.Save(config, tempFile);

            // Assert
            Assert.True(File.Exists(tempFile));
            var loadedConfig = _loader.LoadFromFile(tempFile);
            Assert.Equal(false, loadedConfig.IsEnabled);
            Assert.Equal("/custom/logs", loadedConfig.Storage.LogDirectory);
            Assert.Equal(250, loadedConfig.Storage.WriteBufferSize);
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void SaveToFile_CreatesDirectoryIfNotExists()
    {
        // Arrange
        var config = new VisualizationConfiguration();
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        var tempFile = Path.Combine(tempDir, "config.json");

        try
        {
            // Act
            _loader.Save(config, tempFile);

            // Assert
            Assert.True(File.Exists(tempFile));
        }
        finally
        {
            if (Directory.Exists(tempDir))
            {
                Directory.Delete(tempDir, true);
            }
        }
    }

    [Fact]
    public void SaveToJson_ReturnsValidJson()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            IsEnabled = false,
            Storage = new StorageConfiguration
            {
                LogDirectory = "/test"
            }
        };

        // Act
        var json = _loader.SaveToJson(config);

        // Assert
        Assert.Contains("IsEnabled", json);
        Assert.Contains("false", json);
        Assert.Contains("LogDirectory", json);
        Assert.Contains("/test", json);
    }

    [Fact]
    public void LoadFromEnvironment_NoEnvironmentVariablesReturnsDefaults()
    {
        // Act
        var config = _loader.LoadFromEnvironment();

        // Assert
        Assert.True(config.IsEnabled);
        Assert.Equal("file", config.Storage.BackendType);
        Assert.Equal(100, config.Storage.WriteBufferSize);
    }
}
