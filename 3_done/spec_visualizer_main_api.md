# Spec: Visualizer Main API

## Overview
Implement the main TensorBoardVisualizer API that provides a unified interface for all visualization and profiling functionality.

## Objectives
- Provide a simple, high-level API for common visualization tasks
- Integrate all visualization components (scalars, histograms, profiling, etc.)
- Support easy configuration and setup
- Maintain backward compatibility while being extensible

## API Design

```csharp
// Main visualizer class
public class TensorBoardVisualizer : IDisposable
{
    // Constructors
    public TensorBoardVisualizer(string logDirectory);
    public TensorBoardVisualizer(StorageConfiguration config);
    public TensorBoardVisualizer(IStorageBackend storage);

    // Scalar metrics
    public void LogScalar(string name, float value, long step = -1);
    public void LogScalar(string name, double value, long step = -1);
    public Task LogScalarAsync(string name, float value, long step = -1);

    // Histograms
    public void LogHistogram(string name, float[] values, long step = -1);
    public void LogHistogram(string name, float[] values, HistogramBinConfig config, long step = -1);
    public Task LogHistogramAsync(string name, float[] values, long step = -1);

    // Computational graph
    public void LogGraph(IModel model);
    public void LogGraph(ComputationalGraph graph);
    public Task LogGraphAsync(IModel model);

    // Profiling
    public IProfilingScope StartProfile(string name);
    public IProfilingScope StartProfile(string name, Dictionary<string, string> metadata);
    public void RecordInstant(string name);

    // Advanced features
    public void LogHyperparameters(Dictionary<string, object> hyperparams);
    public void LogText(string name, string text);
    public void LogImage(string name, byte[] imageData, long step = -1);

    // Export and cleanup
    public void Export();
    public Task ExportAsync();
    public void Flush();
    public Task FlushAsync();

    // Configuration
    public StorageConfiguration StorageConfig { get; set; }
    public bool IsEnabled { get; set; } = true;
}

// Fluent builder pattern
public class VisualizerBuilder
{
    public static VisualizerBuilder Create() => new VisualizerBuilder();

    public VisualizerBuilder WithLogDirectory(string directory);
    public VisualizerBuilder WithStorageBackend(IStorageBackend backend);
    public VisualizerBuilder WithStorageConfig(StorageConfiguration config);
    public VisualizerBuilder WithScalarLogger(IScalarLogger logger);
    public VisualizerBuilder WithHistogramLogger(IHistogramLogger logger);
    public VisualizerBuilder WithProfiler(IProfiler profiler);
    public VisualizerBuilder EnableAsync(bool enable = true);
    public VisualizerBuilder EnableProfiling(bool enable = true);

    public TensorBoardVisualizer Build();
}

// Example usage
// Basic
var visualizer = new TensorBoardVisualizer("./logs");
visualizer.LogScalar("loss", 0.5f, step: 1);

// Advanced configuration
var visualizer = VisualizerBuilder.Create()
    .WithLogDirectory("./logs")
    .EnableAsync(true)
    .Build();

// Profiling
using (visualizer.StartProfile("forward_pass")) {
    var output = model.Forward(input);
}
```

## Implementation Requirements

### 1. TensorBoardVisualizer Core (45-60 min)
- Implement main class with constructor overloads
- Initialize internal components:
  - Event system
  - Storage backend (file-based by default)
  - Scalar logger
  - Histogram logger
  - Profiler
- Implement delegate methods to internal components:
  - `LogScalar` -> `IScalarLogger.LogScalar`
  - `LogHistogram` -> `IHistogramLogger.LogHistogram`
  - `StartProfile` -> `IProfiler.StartProfile`
- Add `IsEnabled` flag to skip all operations when disabled
- Implement `Dispose` pattern for cleanup

### 2. Logging Methods (30-45 min)
- Implement logging methods for all data types:
  - Scalars (float, double)
  - Histograms
  - Computational graphs (placeholder for now)
  - Hyperparameters (as metadata)
  - Text (for debugging notes)
  - Images (as byte arrays)
- Support both sync and async versions
- Handle step numbers (auto-increment if not provided)
- Add validation for inputs

### 3. Export and Flush (20-30 min)
- Implement `Flush()` to force all data to be written
- Implement `Export()` to finalize and export data:
  - Close all open files
  - Flush all buffers
  - Write summary metadata
- Support async versions
- Ensure thread-safe flush operations

### 4. VisualizerBuilder (30-45 min)
- Implement fluent builder pattern
- Support all configuration options:
  - Log directory (creates file storage)
  - Custom storage backend
  - Custom loggers
  - Async enable/disable
  - Profiling enable/disable
- Validate configuration before building
- Provide helpful error messages for invalid configuration
- Support partial configuration (use defaults for unspecified)

## File Structure
```
src/
  MLFramework.Visualization/
    TensorBoardVisualizer.cs
    VisualizerBuilder.cs
    Configuration/
      VisualizerConfiguration.cs

tests/
  MLFramework.Visualization.Tests/
    TensorBoardVisualizerTests.cs
    VisualizerBuilderTests.cs
    IntegrationTests/
      EndToEndTests.cs
```

## Dependencies
- `MLFramework.Visualization.Events` (Event system)
- `MLFramework.Visualization.Storage` (Storage backends)
- `MLFramework.Visualization.Scalars` (Scalar logger)
- `MLFramework.Visualization.Histograms` (Histogram logger)
- `MLFramework.Visualization.Profiling` (Profiler)

## Integration Points
- Main entry point for users
- Integrates all visualization components
- Will be integrated with training loops

## Success Criteria
- Basic usage requires <5 lines of code
- All logging methods work correctly with minimal overhead
- Builder pattern enables easy configuration
- Unit tests verify all functionality
- Integration tests verify end-to-end workflow
