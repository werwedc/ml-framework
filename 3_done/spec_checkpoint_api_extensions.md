# Spec: Checkpoint API Extensions

## Overview
Implement convenient extension methods and high-level APIs for easy integration of checkpointing into models and training loops. These extensions provide a user-friendly interface that works out-of-the-box for most use cases.

## Classes and Extensions

### Location
`src/MLFramework/Checkpointing/Extensions/`

### Class: ModelCheckpointExtensions

```csharp
namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Extension methods for checkpointing models
/// </summary>
public static class ModelCheckpointExtensions
{
    /// <summary>
    /// Enables checkpointing for all layers in the model
    /// </summary>
    /// <typeparam name="T">Type of the model</typeparam>
    /// <param name="model">The model to enable checkpointing for</param>
    /// <returns>The model for method chaining</returns>
    public static T CheckpointAll<T>(this T model) where T : class
    {
        var config = CheckpointConfig.Default;
        return CheckpointModel(model, config);
    }

    /// <summary>
    /// Enables checkpointing for specific layers in the model
    /// </summary>
    /// <typeparam name="T">Type of the model</typeparam>
    /// <param name="model">The model to enable checkpointing for</param>
    /// <param name="layerIds">List of layer IDs to checkpoint</param>
    /// <returns>The model for method chaining</returns>
    public static T CheckpointLayers<T>(this T model, IEnumerable<string> layerIds) where T : class
    {
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Selective,
            CheckpointLayers = layerIds.ToArray()
        };
        return CheckpointModel(model, config);
    }

    /// <summary>
    /// Enables checkpointing with a custom configuration
    /// </summary>
    /// <typeparam name="T">Type of the model</typeparam>
    /// <param name="model">The model to enable checkpointing for</param>
    /// <param name="config">Checkpoint configuration</param>
    /// <returns>The model for method chaining</returns>
    public static T Checkpoint<T>(this T model, CheckpointConfig config) where T : class
    {
        return CheckpointModel(model, config);
    }

    /// <summary>
    /// Enables interval-based checkpointing
    /// </summary>
    /// <typeparam name="T">Type of the model</typeparam>
    /// <param name="model">The model to enable checkpointing for</param>
    /// <param name="interval">Interval between checkpoints</param>
    /// <returns>The model for method chaining</returns>
    public static T CheckpointEvery<T>(this T model, int interval = 2) where T : class
    {
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = interval
        };
        return CheckpointModel(model, config);
    }

    /// <summary>
    /// Disables checkpointing for the model
    /// </summary>
    /// <typeparam name="T">Type of the model</typeparam>
    /// <param name="model">The model to disable checkpointing for</param>
    /// <returns>The model for method chaining</returns>
    public static T DisableCheckpointing<T>(this T model) where T : class
    {
        // Implementation will remove checkpointing from the model
        throw new NotImplementedException();
    }

    /// <summary>
    /// Gets checkpointing statistics for the model
    /// </summary>
    /// <typeparam name="T">Type of the model</typeparam>
    /// <param name="model">The model to get statistics for</param>
    /// <returns>Checkpointing statistics</returns>
    public static CheckpointStatistics GetCheckpointStatistics<T>(this T model) where T : class
    {
        // Implementation will return checkpointing statistics
        throw new NotImplementedException();
    }

    private static T CheckpointModel<T>(T model, CheckpointConfig config) where T : class
    {
        // Implementation will integrate checkpointing into the model
        // This will involve wrapping layers with checkpoint functions
        // and setting up the checkpoint manager
        throw new NotImplementedException();
    }
}
```

### Class: ModuleCheckpointExtensions

```csharp
namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Extension methods for checkpointing individual modules/layers
/// </summary>
public static class ModuleCheckpointExtensions
{
    /// <summary>
    /// Wraps a module with checkpointing
    /// </summary>
    /// <typeparam name="T">Type of the module</typeparam>
    /// <param name="module">The module to wrap</param>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>Checkpointed module wrapper</returns>
    public static ICheckpointedModule<T> AsCheckpointed<T>(this T module, string layerId)
        where T : class
    {
        return new CheckpointedModuleWrapper<T>(module, layerId);
    }

    /// <summary>
    /// Wraps a module with checkpointing and custom configuration
    /// </summary>
    /// <typeparam name="T">Type of the module</typeparam>
    /// <param name="module">The module to wrap</param>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="config">Checkpoint configuration</param>
    /// <returns>Checkpointed module wrapper</returns>
    public static ICheckpointedModule<T> AsCheckpointed<T>(
        this T module,
        string layerId,
        CheckpointConfig config)
        where T : class
    {
        return new CheckpointedModuleWrapper<T>(module, layerId, config);
    }
}
```

## Checkpointed Module Interface

### Interface: ICheckpointedModule<T>

```csharp
namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Interface for checkpointed modules
/// </summary>
/// <typeparam name="T">Type of the underlying module</typeparam>
public interface ICheckpointedModule<T> : IDisposable
{
    /// <summary>
    /// Gets the underlying module
    /// </summary>
    T Module { get; }

    /// <summary>
    /// Gets the layer ID
    /// </summary>
    string LayerId { get; }

    /// <summary>
    /// Gets the checkpoint configuration
    /// </summary>
    CheckpointConfig Config { get; }

    /// <summary>
    /// Enables checkpointing
    /// </summary>
    void EnableCheckpointing();

    /// <summary>
    /// Disables checkpointing
    /// </summary>
    void DisableCheckpointing();

    /// <summary>
    /// Gets checkpointing statistics
    /// </summary>
    /// <returns>Checkpointing statistics</returns>
    CheckpointStatistics GetStatistics();
}
```

### Class: CheckpointedModuleWrapper<T>

```csharp
namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Wrapper for checkpointed modules
/// </summary>
/// <typeparam name="T">Type of the underlying module</typeparam>
public class CheckpointedModuleWrapper<T> : ICheckpointedModule<T>
    where T : class
{
    private readonly T _module;
    private readonly string _layerId;
    private readonly CheckpointConfig _config;
    private readonly CheckpointManager _checkpointManager;
    private readonly RecomputationEngine _recomputeEngine;
    private bool _checkpointingEnabled;

    /// <summary>
    /// Initializes a new instance of CheckpointedModuleWrapper
    /// </summary>
    /// <param name="module">The module to wrap</param>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="config">Checkpoint configuration (optional, uses default if null)</param>
    public CheckpointedModuleWrapper(
        T module,
        string layerId,
        CheckpointConfig? config = null)
    {
        _module = module ?? throw new ArgumentNullException(nameof(module));
        _layerId = layerId ?? throw new ArgumentNullException(nameof(layerId));
        _config = config ?? CheckpointConfig.Default;
        _checkpointManager = new CheckpointManager();
        _recomputeEngine = new RecomputationEngine();
        _checkpointingEnabled = true;
    }

    /// <summary>
    /// Gets the underlying module
    /// </summary>
    public T Module => _module;

    /// <summary>
    /// Gets the layer ID
    /// </summary>
    public string LayerId => _layerId;

    /// <summary>
    /// Gets the checkpoint configuration
    /// </summary>
    public CheckpointConfig Config => _config;

    /// <summary>
    /// Enables checkpointing
    /// </summary>
    public void EnableCheckpointing()
    {
        _checkpointingEnabled = true;
    }

    /// <summary>
    /// Disables checkpointing
    /// </summary>
    public void DisableCheckpointing()
    {
        _checkpointingEnabled = false;
    }

    /// <summary>
    /// Gets checkpointing statistics
    /// </summary>
    public CheckpointStatistics GetStatistics()
    {
        var memoryStats = _checkpointManager.GetMemoryStats();
        var recomputeStats = _recomputeEngine.GetStats();

        return new CheckpointStatistics
        {
            LayerId = _layerId,
            MemoryUsed = memoryStats.CurrentMemoryUsed,
            PeakMemoryUsed = memoryStats.PeakMemoryUsed,
            RecomputationCount = recomputeStats.TotalRecomputations,
            RecomputationTimeMs = recomputeStats.TotalRecomputationTimeMs,
            IsCheckpointingEnabled = _checkpointingEnabled
        };
    }

    /// <summary>
    /// Disposes the wrapper and releases resources
    /// </summary>
    public void Dispose()
    {
        _checkpointManager.Dispose();
        _recomputeEngine.Dispose();
    }
}
```

## Training Loop Extensions

### Class: TrainingLoopExtensions

```csharp
namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Extension methods for training loop integration
/// </summary>
public static class TrainingLoopExtensions
{
    /// <summary>
    /// Creates a checkpoint-aware training context
    /// </summary>
    /// <typeparam name="TModel">Type of the model</typeparam>
    /// <param name="model">The model to train</param>
    /// <param name="config">Checkpoint configuration</param>
    /// <returns>Checkpointed training context</returns>
    public static CheckpointedTrainingContext<TModel> WithCheckpointing<TModel>(
        this TModel model,
        CheckpointConfig config)
        where TModel : class
    {
        return new CheckpointedTrainingContext<TModel>(model, config);
    }

    /// <summary>
    /// Creates a checkpoint-aware training context with default configuration
    /// </summary>
    /// <typeparam name="TModel">Type of the model</typeparam>
    /// <param name="model">The model to train</param>
    /// <returns>Checkpointed training context</returns>
    public static CheckpointedTrainingContext<TModel> WithCheckpointing<TModel>(
        this TModel model)
        where TModel : class
    {
        return new CheckpointedTrainingContext<TModel>(model, CheckpointConfig.Default);
    }
}
```

### Class: CheckpointedTrainingContext<TModel>

```csharp
namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Context for checkpointed training
/// </summary>
/// <typeparam name="TModel">Type of the model</typeparam>
public class CheckpointedTrainingContext<TModel> : IDisposable
    where TModel : class
{
    private readonly TModel _model;
    private readonly CheckpointConfig _config;
    private readonly CheckpointContext _checkpointContext;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of CheckpointedTrainingContext
    /// </summary>
    /// <param name="model">The model to train</param>
    /// <param name="config">Checkpoint configuration</param>
    public CheckpointedTrainingContext(TModel model, CheckpointConfig config)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _checkpointContext = new CheckpointContext(config);
        _checkpointContext.Enter();
    }

    /// <summary>
    /// Gets the model
    /// </summary>
    public TModel Model => _model;

    /// <summary>
    /// Gets the checkpoint configuration
    /// </summary>
    public CheckpointConfig Config => _config;

    /// <summary>
    /// Gets checkpointing statistics
    /// </summary>
    public CheckpointStatistics GetStatistics()
    {
        // Implementation will return statistics from the context
        throw new NotImplementedException();
    }

    /// <summary>
    /// Disposes the context
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _checkpointContext.Exit();
            _checkpointContext.Dispose();
            _disposed = true;
        }
    }
}
```

## Statistics Classes

### Class: CheckpointStatistics

```csharp
namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Statistics for checkpointing
/// </summary>
public class CheckpointStatistics
{
    /// <summary>
    /// Layer ID (if applicable)
    /// </summary>
    public string LayerId { get; set; } = string.Empty;

    /// <summary>
    /// Current memory used by checkpoints (in bytes)
    /// </summary>
    public long MemoryUsed { get; set; }

    /// <summary>
    /// Peak memory used by checkpoints (in bytes)
    /// </summary>
    public long PeakMemoryUsed { get; set; }

    /// <summary>
    /// Total number of recomputations
    /// </summary>
    public int RecomputationCount { get; set; }

    /// <summary>
    /// Total time spent on recomputation (in milliseconds)
    /// </summary>
    public long RecomputationTimeMs { get; set; }

    /// <summary>
    /// Whether checkpointing is enabled
    /// </summary>
    public bool IsCheckpointingEnabled { get; set; }

    /// <summary>
    /// Number of checkpoints stored
    /// </summary>
    public int CheckpointCount { get; set; }

    /// <summary>
    /// Memory savings compared to full storage (in bytes)
    /// </summary>
    public long MemorySavings { get; set; }

    /// <summary>
    /// Memory reduction percentage (0.0 to 1.0)
    /// </summary>
    public float MemoryReductionPercentage { get; set; }

    /// <summary>
    /// Timestamp when statistics were collected
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Creates a string summary of the statistics
    /// </summary>
    /// <returns>Summary string</returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("Checkpoint Statistics:");
        sb.AppendLine($"  Memory Used: {FormatBytes(MemoryUsed)}");
        sb.AppendLine($"  Peak Memory: {FormatBytes(PeakMemoryUsed)}");
        sb.AppendLine($"  Recomputations: {RecomputationCount}");
        sb.AppendLine($"  Recomputation Time: {RecomputationTimeMs}ms");
        sb.AppendLine($"  Checkpoint Count: {CheckpointCount}");
        sb.AppendLine($"  Memory Savings: {FormatBytes(MemorySavings)}");
        sb.AppendLine($"  Memory Reduction: {MemoryReductionPercentage:P0}");
        sb.AppendLine($"  Enabled: {IsCheckpointingEnabled}");
        return sb.ToString();
    }

    private string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes}B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F2}KB";
        if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F2}MB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2}GB";
    }
}
```

## Fluent API Example

```csharp
// Example usage of the fluent API
public class Model
{
    public Layer1 Layer1 { get; set; }
    public Layer2 Layer2 { get; set; }
    public Layer3 Layer3 { get; set; }
    // ... more layers
}

// Usage
var model = new Model
{
    Layer1 = new Layer1().AsCheckpointed("layer1"),
    Layer2 = new Layer2().AsCheckpointed("layer2"),
    Layer3 = new Layer3().AsCheckpointed("layer3")
};

// Or use model-level checkpointing
model.CheckpointEvery(3);

// Or selective checkpointing
model.CheckpointLayers(new[] { "layer5", "layer10", "layer15" });

// Or custom configuration
var config = new CheckpointConfig
{
    Strategy = CheckpointStrategy.MemoryAware,
    MaxMemoryPercentage = 0.75f
};
model.Checkpoint(config);

// Training loop with checkpointing
using var context = model.WithCheckpointing(config);
foreach (var batch in dataLoader)
{
    var loss = context.Model.Forward(batch);
    loss.Backward();
    optimizer.Step();

    // Get statistics periodically
    if (batch.Index % 100 == 0)
    {
        var stats = context.GetStatistics();
        Console.WriteLine(stats);
    }
}
```

## Testing Requirements

### Unit Tests

1. **ModelCheckpointExtensions Tests**
   - [ ] CheckpointAll enables checkpointing for all layers
   - [ ] CheckpointLayers enables checkpointing for specified layers
   - [ ] CheckpointEvery enables interval-based checkpointing
   - [ ] Checkpoint applies custom configuration
   - [ ] DisableCheckpointing removes checkpointing
   - [ ] GetCheckpointStatistics returns correct statistics

2. **ModuleCheckpointExtensions Tests**
   - [ ] AsCheckpointed creates correct wrapper
   - [ ] AsCheckpointed with config uses custom config
   - [ ] Wrapper returns correct module
   - [ ] Wrapper returns correct layer ID
   - [ ] Wrapper returns correct config

3. **CheckpointedModuleWrapper Tests**
   - [ ] EnableCheckpointing enables checkpointing
   - [ ] DisableCheckpointing disables checkpointing
   - [ ] GetStatistics returns correct statistics
   - [ ] Dispose cleans up resources correctly

4. **TrainingLoopExtensions Tests**
   - [ ] WithCheckpointing creates correct context
   - [ ] WithCheckpointing with default config works
   - [ ] Context returns correct model
   - [ ] Context returns correct config

5. **CheckpointedTrainingContext Tests**
   - [ ] GetStatistics returns correct statistics
   - [ ] Dispose cleans up resources correctly
   - [ ] Multiple contexts can be created

6. **CheckpointStatistics Tests**
   - [ ] ToString generates correct string
   - [ ] FormatBytes formats correctly for various sizes
   - [ ] All properties are set correctly

7. **Integration Tests**
   - [ ] End-to-end checkpointing with extensions
   - [ ] Training loop with context works correctly
   - [ ] Statistics are accurate

8. **Edge Cases**
   - [ ] Handle null model
   - [ ] Handle null config
   - [ ] Handle empty layer lists
   - [ ] Handle invalid intervals

## Implementation Notes

1. **API Design**:
   - Provide intuitive, easy-to-use APIs
   - Support both simple and advanced use cases
   - Enable method chaining for fluent API

2. **Integration**:
   - Seamlessly integrate with existing model code
   - Minimal changes required to existing code
   - Backward compatibility where possible

3. **Performance**:
   - Minimize overhead of extension methods
   - Avoid unnecessary object creation
   - Cache frequently used objects

4. **Documentation**:
   - Provide clear examples for all APIs
   - Document trade-offs and limitations
   - Include best practices

## Dependencies on Other Specs

This spec depends on:
- **Checkpoint Manager Core** (spec_1) for CheckpointManager
- **Checkpoint Configuration** (spec_2) for CheckpointConfig
- **Recomputation Engine** (spec_4) for RecomputationEngine
- **Autograd Integration** (spec_5) for CheckpointContext

## Estimated Implementation Time
45-60 minutes
