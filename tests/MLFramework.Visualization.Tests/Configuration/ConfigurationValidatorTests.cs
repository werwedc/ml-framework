using Xunit;
using MLFramework.Visualization.Configuration;

namespace MLFramework.Visualization.Tests.Configuration;

public class ConfigurationValidatorTests
{
    private readonly ConfigurationValidator _validator;

    public ConfigurationValidatorTests()
    {
        _validator = new ConfigurationValidator();
    }

    [Fact]
    public void Validate_ValidConfigurationReturnsValid()
    {
        // Arrange
        var config = new VisualizationConfiguration();

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void Validate_NullConfigurationReturnsInvalid()
    {
        // Arrange
        VisualizationConfiguration config = null;

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Configuration cannot be null", result.Errors);
    }

    [Fact]
    public void Validate_NullStorageReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Storage = null
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Storage configuration cannot be null", result.Errors);
    }

    [Fact]
    public void Validate_EmptyLogDirectoryReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Storage = new StorageConfiguration
            {
                LogDirectory = ""
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Storage.LogDirectory cannot be null or empty", result.Errors);
    }

    [Fact]
    public void Validate_NegativeWriteBufferSizeReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Storage = new StorageConfiguration
            {
                LogDirectory = "/logs",
                WriteBufferSize = -1
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Storage.WriteBufferSize must be positive", result.Errors);
    }

    [Fact]
    public void Validate_ZeroWriteBufferSizeReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Storage = new StorageConfiguration
            {
                LogDirectory = "/logs",
                WriteBufferSize = 0
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Storage.WriteBufferSize must be positive", result.Errors);
    }

    [Fact]
    public void Validate_LargeWriteBufferSizeReturnsWarning()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Storage = new StorageConfiguration
            {
                LogDirectory = "/logs",
                WriteBufferSize = 200000
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.True(result.IsValid);
        Assert.Contains("very large", result.Warnings[0]);
    }

    [Fact]
    public void Validate_NegativeFlushIntervalReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Storage = new StorageConfiguration
            {
                LogDirectory = "/logs",
                FlushInterval = TimeSpan.FromMilliseconds(-1)
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Storage.FlushInterval must be positive", result.Errors);
    }

    [Fact]
    public void Validate_LongFlushIntervalReturnsWarning()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Storage = new StorageConfiguration
            {
                LogDirectory = "/logs",
                FlushInterval = TimeSpan.FromMinutes(6)
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.True(result.IsValid);
        Assert.Contains("very long", result.Warnings[0]);
    }

    [Fact]
    public void Validate_NullLoggingReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Logging = null
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Logging configuration cannot be null", result.Errors);
    }

    [Fact]
    public void Validate_NegativeHistogramBinCountReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Logging = new LoggingConfiguration
            {
                HistogramBinCount = -1
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Logging.HistogramBinCount must be positive", result.Errors);
    }

    [Fact]
    public void Validate_LargeHistogramBinCountReturnsWarning()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Logging = new LoggingConfiguration
            {
                HistogramBinCount = 2000
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.True(result.IsValid);
        Assert.Contains("very large", result.Warnings[0]);
    }

    [Fact]
    public void Validate_NegativeSmoothingWindowReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Logging = new LoggingConfiguration
            {
                DefaultSmoothingWindow = -1
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Logging.DefaultSmoothingWindow must be positive", result.Errors);
    }

    [Fact]
    public void Validate_LargeSmoothingWindowReturnsWarning()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Logging = new LoggingConfiguration
            {
                DefaultSmoothingWindow = 2000
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.True(result.IsValid);
        Assert.Contains("very large", result.Warnings[0]);
    }

    [Fact]
    public void Validate_NullProfilingReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Profiling = null
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Profiling configuration cannot be null", result.Errors);
    }

    [Fact]
    public void Validate_NegativeMaxStoredOperationsReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Profiling = new ProfilingConfiguration
            {
                MaxStoredOperations = -1
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Profiling.MaxStoredOperations must be positive", result.Errors);
    }

    [Fact]
    public void Validate_NoProfilingTargetsReturnsWarning()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Profiling = new ProfilingConfiguration
            {
                EnableProfiling = true,
                ProfileCPU = false,
                ProfileGPU = false
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.True(result.IsValid);
        Assert.Contains("Profiling is enabled but neither CPU nor GPU profiling is selected", result.Warnings);
    }

    [Fact]
    public void Validate_NullMemoryProfilingReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            MemoryProfiling = null
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("MemoryProfiling configuration cannot be null", result.Errors);
    }

    [Fact]
    public void Validate_NegativeStackTraceDepthReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            MemoryProfiling = new MemoryProfilingConfiguration
            {
                MaxStackTraceDepth = -1
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("MemoryProfiling.MaxStackTraceDepth must be positive", result.Errors);
    }

    [Fact]
    public void Validate_StackTraceCaptureDisabledWhenMemoryProfilingDisabledReturnsWarning()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            MemoryProfiling = new MemoryProfilingConfiguration
            {
                EnableMemoryProfiling = false,
                CaptureStackTraces = true
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.True(result.IsValid);
        Assert.Contains("MemoryProfiling.CaptureStackTraces is enabled but memory profiling is disabled", result.Warnings);
    }

    [Fact]
    public void Validate_VerySmallMemorySnapshotIntervalReturnsWarning()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            MemoryProfiling = new MemoryProfilingConfiguration
            {
                SnapshotIntervalMs = 50,
                AutoSnapshot = true
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.True(result.IsValid);
        Assert.Contains("very small", result.Warnings[0]);
    }

    [Fact]
    public void Validate_NullGPUTrackingReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            GPUTracking = null
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("GPUTracking configuration cannot be null", result.Errors);
    }

    [Fact]
    public void Validate_NegativeSamplingIntervalMsReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            GPUTracking = new GPUTrackingConfiguration
            {
                SamplingIntervalMs = -1
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("GPUTracking.SamplingIntervalMs must be positive", result.Errors);
    }

    [Fact]
    public void Validate_NoGPUTrackingMetricsReturnsWarning()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            GPUTracking = new GPUTrackingConfiguration
            {
                EnableGPUTracking = true,
                TrackTemperature = false,
                TrackPower = false
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.True(result.IsValid);
        Assert.Contains("GPUTracking is enabled but neither temperature nor power tracking is selected", result.Warnings);
    }

    [Fact]
    public void Validate_NullEventCollectionReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            EventCollection = null
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("EventCollection configuration cannot be null", result.Errors);
    }

    [Fact]
    public void Validate_NegativeBufferCapacityReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            EventCollection = new EventCollectionConfiguration
            {
                BufferCapacity = -1
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("EventCollection.BufferCapacity must be positive", result.Errors);
    }

    [Fact]
    public void Validate_BatchSizeLargerThanBufferCapacityReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            EventCollection = new EventCollectionConfiguration
            {
                BufferCapacity = 100,
                BatchSize = 200
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("EventCollection.BatchSize cannot be larger than BufferCapacity", result.Errors);
    }

    [Fact]
    public void Validate_BatchSizeLargerThanMaxQueueLengthReturnsInvalid()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            EventCollection = new EventCollectionConfiguration
            {
                BufferCapacity = 1000,
                BatchSize = 100,
                MaxQueueLength = 50
            }
        };

        // Act
        var result = _validator.Validate(config);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("EventCollection.BatchSize cannot be larger than MaxQueueLength", result.Errors);
    }

    [Fact]
    public void ValidateAndThrow_ValidConfigurationSucceeds()
    {
        // Arrange
        var config = new VisualizationConfiguration();

        // Act & Assert
        _validator.ValidateAndThrow(config);
    }

    [Fact]
    public void ValidateAndThrow_InvalidConfigurationThrowsException()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Storage = new StorageConfiguration
            {
                LogDirectory = ""
            }
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _validator.ValidateAndThrow(config));
    }

    [Fact]
    public void GetSummary_ReturnsFormattedErrorsAndWarnings()
    {
        // Arrange
        var config = new VisualizationConfiguration
        {
            Storage = new StorageConfiguration
            {
                LogDirectory = "",
                WriteBufferSize = 200000
            }
        };

        // Act
        var result = _validator.Validate(config);
        var summary = result.GetSummary();

        // Assert
        Assert.Contains("Errors:", summary);
        Assert.Contains("Warnings:", summary);
        Assert.Contains("Storage.LogDirectory cannot be null or empty", summary);
        Assert.Contains("Storage.WriteBufferSize is very large", summary);
    }

    [Fact]
    public void GetSummary_ValidConfigurationReturnsSuccessMessage()
    {
        // Arrange
        var config = new VisualizationConfiguration();

        // Act
        var result = _validator.Validate(config);
        var summary = result.GetSummary();

        // Assert
        Assert.Contains("Configuration is valid", summary);
    }
}
