# Spec: Visualization Configuration System

## Overview
Implement a configuration system that allows flexible and user-friendly configuration of visualization and profiling behavior across the entire ML framework.

## Objectives
- Provide centralized configuration management
- Support multiple configuration sources (code, JSON files, environment variables)
- Enable default values with easy overrides
- Validate configurations at startup

## API Design

```csharp
// Main configuration class
public class VisualizationConfiguration
{
    // Storage configuration
    public StorageConfiguration Storage { get; set; }

    // Logging configuration
    public LoggingConfiguration Logging { get; set; }

    // Profiling configuration
    public ProfilingConfiguration Profiling { get; set; }

    // Memory profiling configuration
    public MemoryProfilingConfiguration MemoryProfiling { get; set; }

    // GPU tracking configuration
    public GPUTrackingConfiguration GPUTracking { get; set; }

    // Event collection configuration
    public EventCollectionConfiguration EventCollection { get; set; }

    // Global settings
    public bool IsEnabled { get; set; } = true;
    public bool VerboseLogging { get; set; } = false;
}

// Storage configuration
public class StorageConfiguration
{
    public string BackendType { get; set; } = "file";
    public string LogDirectory { get; set; } = "./logs";
    public string ConnectionString { get; set; }

    public int WriteBufferSize { get; set; } = 100;
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(1);
    public bool EnableAsyncWrites { get; set; } = true;
}

// Logging configuration
public class LoggingConfiguration
{
    public bool LogScalars { get; set; } = true;
    public bool LogHistograms { get; set; } = true;
    public bool LogGraphs { get; set; } = true;
    public bool LogHyperparameters { get; set; } = true;

    public string ScalarLogPrefix { get; set; } = "";
    public int HistogramBinCount { get; set; } = 30;
    public bool AutoSmoothScalars { get; set; } = true;
    public int DefaultSmoothingWindow { get; set; } = 10;
}

// Profiling configuration
public class ProfilingConfiguration
{
    public bool EnableProfiling { get; set; } = true;
    public bool ProfileForwardPass { get; set; } = true;
    public bool ProfileBackwardPass { get; set; } = true;
    public bool ProfileOptimizerStep { get; set; } = false;

    public bool ProfileCPU { get; set; } = true;
    public bool ProfileGPU { get; set; } = true;
    public int MaxStoredOperations { get; set; } = 10000;
}

// Memory profiling configuration
public class MemoryProfilingConfiguration
{
    public bool EnableMemoryProfiling { get; set; } = false;
    public bool CaptureStackTraces { get; set; } = false;
    public int MaxStackTraceDepth { get; set; } = 10;
    public int SnapshotIntervalMs { get; set; } = 1000;
    public bool AutoSnapshot { get; set; } = true;
}

// GPU tracking configuration
public class GPUTrackingConfiguration
{
    public bool EnableGPUTracking { get; set; } = false;
    public int SamplingIntervalMs { get; set; } = 1000;
    public bool TrackTemperature { get; set; } = true;
    public bool TrackPower { get; set; } = true;
}

// Event collection configuration
public class EventCollectionConfiguration
{
    public bool EnableAsync { get; set; } = true;
    public int BufferCapacity { get; set; } = 1000;
    public int BatchSize { get; set; } = 100;
    public bool EnableBackpressure { get; set; } = true;
    public int MaxQueueLength { get; set; } = 10000;
}

// Configuration loader
public interface IConfigurationLoader
{
    VisualizationConfiguration Load();
    VisualizationConfiguration LoadFromFile(string filePath);
    VisualizationConfiguration LoadFromJson(string json);
    VisualizationConfiguration LoadFromEnvironment();
    void Save(VisualizationConfiguration config, string filePath);
    void SaveToJson(VisualizationConfiguration config, string filePath);
}

public class ConfigurationLoader : IConfigurationLoader
{
    public ConfigurationLoader();
    public VisualizationConfiguration Load();
    public VisualizationConfiguration LoadFromFile(string filePath);
    public VisualizationConfiguration LoadFromJson(string json);
    public VisualizationConfiguration LoadFromEnvironment();
}

// Configuration validator
public interface IConfigurationValidator
{
    ValidationResult Validate(VisualizationConfiguration config);
    void ValidateAndThrow(VisualizationConfiguration config);
}

public class ValidationResult
{
    public bool IsValid { get; }
    public List<string> Errors { get; }
    public List<string> Warnings { get; }
}

// Usage examples
// Load from file
var config = ConfigurationLoader.LoadFromFile("viz_config.json");

// Load with overrides
var config = ConfigurationLoader.Load();
config.Storage.LogDirectory = "./custom_logs";
config.Profiling.EnableProfiling = true;

// Use with visualizer
var visualizer = new TensorBoardVisualizer(config);
```

## Implementation Requirements

### 1. Configuration Classes (30-45 min)
- Implement all configuration classes:
  - `VisualizationConfiguration`
  - `StorageConfiguration`
  - `LoggingConfiguration`
  - `ProfilingConfiguration`
  - `MemoryProfilingConfiguration`
  - `GPUTrackingConfiguration`
  - `EventCollectionConfiguration`
- Set sensible defaults for all properties
- Add validation attributes where appropriate
- Support nested configuration objects

### 2. ConfigurationLoader (45-60 min)
- Implement `IConfigurationLoader` interface
- Load from JSON files:
  - Parse JSON configuration
  - Map to configuration objects
  - Handle missing fields (use defaults)
  - Handle invalid values (throw or use defaults)
- Load from JSON strings:
  - Same as file but from string input
- Load from environment variables:
  - Use naming convention (e.g., `MLFRAMEWORK_VISUALIZATION_STORAGE_LOGDIR`)
  - Support nested values with underscores
  - Type conversion (string to int, bool, etc.)
- Load default configuration:
  - Use hard-coded defaults
  - Override with environment variables if available
- Save configuration:
  - Serialize to JSON
  - Write to file
- Handle file I/O errors gracefully

### 3. ConfigurationValidator (30-45 min)
- Implement `IConfigurationValidator` interface
- Validate all configuration sections:
  - Storage: Check directory paths, connection strings
  - Logging: Check positive values for counts/windows
  - Profiling: Check reasonable limits
  - Memory: Check valid stack trace depths
  - GPU: Check valid sampling intervals
  - Event collection: Check positive buffer sizes
- Collect errors and warnings:
  - Errors: Critical issues that must be fixed
  - Warnings: Non-critical issues that should be reviewed
- Implement `ValidateAndThrow`:
  - Throw exception if validation fails
  - Include all errors in exception message

### 4. Configuration Merging (20-30 min)
- Implement configuration merging:
  - Merge multiple configuration sources
  - Priority: Code overrides > Environment > File > Defaults
  - Handle partial configurations (only override specified values)
- Support profiles:
  - Define multiple named profiles (e.g., "debug", "production")
  - Load profile by name
  - Override profile values after loading

### 5. Configuration Builder (20-30 min)
- Implement fluent builder pattern:
  - Chain configuration methods
  - Build final configuration object
- Example:
  ```csharp
  var config = new VisualizationConfigurationBuilder()
      .WithStorageDirectory("./logs")
      .EnableProfiling(true)
      .WithLogPrefix("experiment1/")
      .Build();
  ```
- Validate configuration before returning
- Provide helpful error messages for invalid configurations

## File Structure
```
src/
  MLFramework.Visualization/
    Configuration/
      VisualizationConfiguration.cs
      StorageConfiguration.cs
      LoggingConfiguration.cs
      ProfilingConfiguration.cs
      MemoryProfilingConfiguration.cs
      GPUTrackingConfiguration.cs
      EventCollectionConfiguration.cs
      IConfigurationLoader.cs
      ConfigurationLoader.cs
      IConfigurationValidator.cs
      ConfigurationValidator.cs
      VisualizationConfigurationBuilder.cs

tests/
  MLFramework.Visualization.Tests/
    Configuration/
      ConfigurationLoaderTests.cs
      ConfigurationValidatorTests.cs
      ConfigurationBuilderTests.cs
```

## Dependencies
- `MLFramework.Visualization` (Main visualizer)
- System.Text.Json (for JSON serialization)

## Integration Points
- Used by TensorBoardVisualizer to configure behavior
- Used by all visualization components for their settings
- Enables external configuration without code changes

## Success Criteria
- Loading configuration from JSON file works correctly
- Environment variables override default values
- Validation catches all invalid configurations
- Builder pattern enables easy configuration
- Default configuration works out of the box
- Unit tests verify all loading, merging, and validation scenarios
