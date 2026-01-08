# Spec: Checkpoint Strategies

## Overview
Implement various checkpointing strategies that determine which layers to checkpoint based on different criteria (interval, size-based, selective, memory-aware, smart).

## Classes

### Location
`src/MLFramework/Checkpointing/Strategies/`

### Interface: ICheckpointStrategy

```csharp
namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Interface for checkpointing strategies
/// </summary>
public interface ICheckpointStrategy
{
    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor</param>
    /// <param name="layerIndex">Index of the layer in the network</param>
    /// <returns>True if should checkpoint, false otherwise</returns>
    bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex);

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    void Reset();
}
```

## Strategy Implementations

### Class: IntervalCheckpointStrategy

```csharp
namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Checkpoints layers at fixed intervals (every N layers)
/// </summary>
public class IntervalCheckpointStrategy : ICheckpointStrategy
{
    private readonly int _interval;
    private int _checkpointedCount;

    /// <summary>
    /// Initializes a new instance of IntervalCheckpointStrategy
    /// </summary>
    /// <param name="interval">Number of layers between checkpoints</param>
    public IntervalCheckpointStrategy(int interval = 2)
    {
        if (interval <= 0)
            throw new ArgumentException("Interval must be greater than 0", nameof(interval));

        _interval = interval;
        _checkpointedCount = 0;
    }

    /// <summary>
    /// Gets the interval value
    /// </summary>
    public int Interval => _interval;

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => $"Interval({_interval})";

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    public bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex)
    {
        // Checkpoint every N layers
        var shouldCheckpoint = (layerIndex % _interval) == 0;

        if (shouldCheckpoint)
        {
            _checkpointedCount++;
        }

        return shouldCheckpoint;
    }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    public void Reset()
    {
        _checkpointedCount = 0;
    }
}
```

### Class: SelectiveCheckpointStrategy

```csharp
namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Checkpoints only specific layers defined by the user
/// </summary>
public class SelectiveCheckpointStrategy : ICheckpointStrategy
{
    private readonly HashSet<string> _checkpointLayers;
    private readonly HashSet<string> _excludeLayers;

    /// <summary>
    /// Initializes a new instance of SelectiveCheckpointStrategy
    /// </summary>
    /// <param name="checkpointLayers">List of layer IDs to checkpoint</param>
    /// <param name="excludeLayers">List of layer IDs to exclude from checkpointing</param>
    public SelectiveCheckpointStrategy(
        IEnumerable<string>? checkpointLayers = null,
        IEnumerable<string>? excludeLayers = null)
    {
        _checkpointLayers = checkpointLayers?.ToHashSet() ?? new HashSet<string>();
        _excludeLayers = excludeLayers?.ToHashSet() ?? new HashSet<string>();

        // Check for overlaps
        var overlap = _checkpointLayers.Intersect(_excludeLayers).ToList();
        if (overlap.Count > 0)
        {
            throw new ArgumentException(
                $"Layers cannot be both checkpointed and excluded: {string.Join(", ", overlap)}");
        }
    }

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => "Selective";

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    public bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex)
    {
        // If in exclude list, don't checkpoint
        if (_excludeLayers.Contains(layerId))
        {
            return false;
        }

        // If in checkpoint list, do checkpoint
        if (_checkpointLayers.Contains(layerId))
        {
            return true;
        }

        // Default: don't checkpoint
        return false;
    }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    public void Reset()
    {
        // No state to reset for this strategy
    }
}
```

### Class: SizeBasedCheckpointStrategy

```csharp
namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Checkpoints layers based on activation size
/// </summary>
public class SizeBasedCheckpointStrategy : ICheckpointStrategy
{
    private readonly long _minActivationSizeBytes;
    private readonly HashSet<string> _excludeLayers;
    private int _checkpointedCount;

    /// <summary>
    /// Initializes a new instance of SizeBasedCheckpointStrategy
    /// </summary>
    /// <param name="minActivationSizeBytes">Minimum activation size to trigger checkpointing</param>
    /// <param name="excludeLayers">List of layer IDs to exclude</param>
    public SizeBasedCheckpointStrategy(
        long minActivationSizeBytes = 1024 * 1024, // 1MB default
        IEnumerable<string>? excludeLayers = null)
    {
        if (minActivationSizeBytes <= 0)
            throw new ArgumentException("MinActivationSizeBytes must be greater than 0");

        _minActivationSizeBytes = minActivationSizeBytes;
        _excludeLayers = excludeLayers?.ToHashSet() ?? new HashSet<string>();
        _checkpointedCount = 0;
    }

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => $"SizeBased({FormatBytes(_minActivationSizeBytes)})";

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    public bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex)
    {
        // If in exclude list, don't checkpoint
        if (_excludeLayers.Contains(layerId))
        {
            return false;
        }

        // Calculate activation size
        var activationSize = CalculateActivationSize(activation);

        // Checkpoint if size exceeds threshold
        var shouldCheckpoint = activationSize >= _minActivationSizeBytes;

        if (shouldCheckpoint)
        {
            _checkpointedCount++;
        }

        return shouldCheckpoint;
    }

    private long CalculateActivationSize(Tensor activation)
    {
        return activation.ElementCount * activation.DataTypeSize;
    }

    private string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes}B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024}KB";
        if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024 * 1024)}MB";
        return $"{bytes / (1024 * 1024 * 1024)}GB";
    }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    public void Reset()
    {
        _checkpointedCount = 0;
    }
}
```

### Class: MemoryAwareCheckpointStrategy

```csharp
namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Dynamically adjusts checkpointing based on available memory
/// </summary>
public class MemoryAwareCheckpointStrategy : ICheckpointStrategy
{
    private readonly float _maxMemoryPercentage;
    private readonly MemoryTracker _memoryTracker;
    private readonly long _totalSystemMemory;
    private readonly int _initialInterval;
    private int _currentInterval;
    private long _lastAdjustmentTime;
    private int _consecutiveHighMemoryPressure;

    /// <summary>
    /// Initializes a new instance of MemoryAwareCheckpointStrategy
    /// </summary>
    /// <param name="maxMemoryPercentage">Maximum memory percentage to use (0.0 to 1.0)</param>
    /// <param name="memoryTracker">Memory tracker instance</param>
    /// <param name="totalSystemMemory">Total system memory available</param>
    public MemoryAwareCheckpointStrategy(
        float maxMemoryPercentage = 0.8f,
        MemoryTracker? memoryTracker = null,
        long? totalSystemMemory = null)
    {
        if (maxMemoryPercentage <= 0 || maxMemoryPercentage > 1.0f)
            throw new ArgumentException("MaxMemoryPercentage must be between 0 and 1");

        _maxMemoryPercentage = maxMemoryPercentage;
        _memoryTracker = memoryTracker ?? new MemoryTracker();
        _totalSystemMemory = totalSystemMemory ?? EstimateTotalSystemMemory();
        _initialInterval = 2;
        _currentInterval = _initialInterval;
        _lastAdjustmentTime = DateTime.UtcNow.Ticks;
        _consecutiveHighMemoryPressure = 0;
    }

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => $"MemoryAware({maxPercentage:P0})";

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    public bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex)
    {
        // Check current memory pressure
        var memoryPressure = CalculateMemoryPressure();

        // Adjust interval based on memory pressure
        AdjustInterval(memoryPressure);

        // Checkpoint based on current interval
        var shouldCheckpoint = (layerIndex % _currentInterval) == 0;

        return shouldCheckpoint;
    }

    private float CalculateMemoryPressure()
    {
        var stats = _memoryTracker.GetStats();
        return (float)stats.CurrentMemoryUsed / _totalSystemMemory;
    }

    private void AdjustInterval(float memoryPressure)
    {
        var now = DateTime.UtcNow.Ticks;
        var timeSinceLastAdjustment = TimeSpan.FromTicks(now - _lastAdjustmentTime).TotalSeconds;

        // Only adjust every 10 seconds to avoid oscillation
        if (timeSinceLastAdjustment < 10)
            return;

        var threshold = _maxMemoryPercentage;

        if (memoryPressure > threshold)
        {
            // High memory pressure - increase checkpoint frequency (decrease interval)
            _currentInterval = Math.Max(1, _currentInterval - 1);
            _consecutiveHighMemoryPressure++;
        }
        else if (memoryPressure < threshold * 0.8f)
        {
            // Low memory pressure - decrease checkpoint frequency (increase interval)
            if (_consecutiveHighMemoryPressure == 0 || timeSinceLastAdjustment > 30)
            {
                _currentInterval = Math.Min(10, _currentInterval + 1);
            }
        }
        else
        {
            _consecutiveHighMemoryPressure = 0;
        }

        _lastAdjustmentTime = now;
    }

    private long EstimateTotalSystemMemory()
    {
        // This would use system APIs to get available memory
        // For now, return a reasonable default (16GB)
        return 16L * 1024 * 1024 * 1024;
    }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    public void Reset()
    {
        _currentInterval = _initialInterval;
        _lastAdjustmentTime = DateTime.UtcNow.Ticks;
        _consecutiveHighMemoryPressure = 0;
    }
}
```

### Class: SmartCheckpointStrategy

```csharp
namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Uses heuristics to determine optimal checkpointing strategy
/// </summary>
public class SmartCheckpointStrategy : ICheckpointStrategy
{
    private readonly List<LayerInfo> _layerInfo = new List<LayerInfo>();
    private readonly Dictionary<string, long> _layerActivationSizes = new Dictionary<string, long>();
    private readonly Dictionary<string, int> _layerAccessCounts = new Dictionary<string, int>();
    private readonly HashSet<string> _excludeLayers;
    private bool _initialized;

    /// <summary>
    /// Initializes a new instance of SmartCheckpointStrategy
    /// </summary>
    /// <param name="excludeLayers">List of layer IDs to exclude</param>
    public SmartCheckpointStrategy(IEnumerable<string>? excludeLayers = null)
    {
        _excludeLayers = excludeLayers?.ToHashSet() ?? new HashSet<string>();
        _initialized = false;
    }

    /// <summary>
    /// Gets the strategy name
    /// </summary>
    public string Name => "Smart";

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    public bool ShouldCheckpoint(string layerId, Tensor activation, int layerIndex)
    {
        // Update layer statistics
        UpdateLayerStatistics(layerId, activation);

        // If not initialized, collect data first
        if (!_initialized)
        {
            // Collect data for first pass, don't checkpoint yet
            if (layerIndex > 10) // After collecting data from first 10 layers
            {
                _initialized = true;
            }
            return false;
        }

        // If in exclude list, don't checkpoint
        if (_excludeLayers.Contains(layerId))
        {
            return false;
        }

        // Use heuristics to determine if should checkpoint
        return ShouldCheckpointHeuristic(layerId, layerIndex);
    }

    private void UpdateLayerStatistics(string layerId, Tensor activation)
    {
        var activationSize = activation.ElementCount * activation.DataTypeSize;

        if (!_layerActivationSizes.ContainsKey(layerId))
        {
            _layerActivationSizes[layerId] = activationSize;
            _layerAccessCounts[layerId] = 0;
            _layerInfo.Add(new LayerInfo
            {
                LayerId = layerId,
                ActivationSize = activationSize,
                AccessCount = 0
            });
        }

        _layerAccessCounts[layerId]++;
    }

    private bool ShouldCheckpointHeuristic(string layerId, int layerIndex)
    {
        if (!_layerActivationSizes.TryGetValue(layerId, out var activationSize))
        {
            return false;
        }

        var accessCount = _layerAccessCounts[layerId];
        var avgSize = _layerInfo.Average(l => l.ActivationSize);

        // Heuristic 1: Checkpoint large activations
        if (activationSize > avgSize * 1.5)
        {
            return true;
        }

        // Heuristic 2: Checkpoint infrequently accessed layers
        if (accessCount < 2)
        {
            return true;
        }

        // Heuristic 3: Checkpoint layers later in the network
        if (layerIndex > _layerInfo.Count * 0.7)
        {
            return true;
        }

        // Default: don't checkpoint
        return false;
    }

    /// <summary>
    /// Resets the strategy state
    /// </summary>
    public void Reset()
    {
        _layerInfo.Clear();
        _layerActivationSizes.Clear();
        _layerAccessCounts.Clear();
        _initialized = false;
    }

    private class LayerInfo
    {
        public string LayerId { get; set; } = string.Empty;
        public long ActivationSize { get; set; }
        public int AccessCount { get; set; }
    }
}
```

## Strategy Factory

### Class: CheckpointStrategyFactory

```csharp
namespace MLFramework.Checkpointing.Strategies;

/// <summary>
/// Factory for creating checkpoint strategies
/// </summary>
public static class CheckpointStrategyFactory
{
    /// <summary>
    /// Creates a strategy from a configuration
    /// </summary>
    /// <param name="config">Checkpoint configuration</param>
    /// <returns>Checkpoint strategy instance</returns>
    public static ICheckpointStrategy CreateStrategy(CheckpointConfig config)
    {
        return config.Strategy switch
        {
            CheckpointStrategy.Interval => new IntervalCheckpointStrategy(config.Interval),
            CheckpointStrategy.Selective => new SelectiveCheckpointStrategy(
                config.CheckpointLayers,
                config.ExcludeLayers),
            CheckpointStrategy.SizeBased => new SizeBasedCheckpointStrategy(
                config.MinActivationSizeBytes,
                config.ExcludeLayers),
            CheckpointStrategy.MemoryAware => new MemoryAwareCheckpointStrategy(
                config.MaxMemoryPercentage),
            CheckpointStrategy.Smart => new SmartCheckpointStrategy(
                config.ExcludeLayers),
            _ => throw new ArgumentException($"Unknown strategy: {config.Strategy}")
        };
    }

    /// <summary>
    /// Creates an interval strategy
    /// </summary>
    public static ICheckpointStrategy CreateInterval(int interval = 2)
    {
        return new IntervalCheckpointStrategy(interval);
    }

    /// <summary>
    /// Creates a selective strategy
    /// </summary>
    public static ICheckpointStrategy CreateSelective(
        IEnumerable<string>? checkpointLayers = null,
        IEnumerable<string>? excludeLayers = null)
    {
        return new SelectiveCheckpointStrategy(checkpointLayers, excludeLayers);
    }

    /// <summary>
    /// Creates a size-based strategy
    /// </summary>
    public static ICheckpointStrategy CreateSizeBased(
        long minActivationSizeBytes = 1024 * 1024,
        IEnumerable<string>? excludeLayers = null)
    {
        return new SizeBasedCheckpointStrategy(minActivationSizeBytes, excludeLayers);
    }

    /// <summary>
    /// Creates a memory-aware strategy
    /// </summary>
    public static ICheckpointStrategy CreateMemoryAware(
        float maxMemoryPercentage = 0.8f,
        MemoryTracker? memoryTracker = null)
    {
        return new MemoryAwareCheckpointStrategy(maxMemoryPercentage, memoryTracker);
    }

    /// <summary>
    /// Creates a smart strategy
    /// </summary>
    public static ICheckpointStrategy CreateSmart(
        IEnumerable<string>? excludeLayers = null)
    {
        return new SmartCheckpointStrategy(excludeLayers);
    }
}
```

## Testing Requirements

### Unit Tests

1. **IntervalCheckpointStrategy Tests**
   - [ ] Checkpoints at correct intervals
   - [ ] Handles invalid interval (throws exception)
   - [ ] Reset clears state correctly
   - [ ] Name returns correct string

2. **SelectiveCheckpointStrategy Tests**
   - [ ] Checkpoints only specified layers
   - [ ] Excludes specified layers
   - [ ] Throws exception on overlap
   - [ ] Handles empty lists correctly

3. **SizeBasedCheckpointStrategy Tests**
   - [ ] Checkpoints large activations
   - [ ] Skips small activations
   - [ ] Handles invalid size threshold (throws exception)
   - [ ] Excludes specified layers

4. **MemoryAwareCheckpointStrategy Tests**
   - [ ] Adjusts interval based on memory pressure
   - [ ] Respects max memory percentage
   - [ ] Doesn't oscillate too frequently
   - [ ] Reset clears state correctly

5. **SmartCheckpointStrategy Tests**
   - [ ] Collects statistics during initialization
   - [ ] Checkpoints large activations
   - [ ] Checkpoints infrequently accessed layers
   - [ ] Checkpoints layers later in network
   - [ ] Reset clears state correctly

6. **CheckpointStrategyFactory Tests**
   - [ ] Creates correct strategy from config
   - [ ] Creates correct strategy from helper methods
   - [ ] Throws exception for unknown strategy
   - [ ] All factory methods work correctly

7. **Edge Cases**
   - [ ] Handle layer index 0
   - [ ] Handle very large layer indices
   - [ ] Handle very small/large activations
   - [ ] Handle concurrent strategy usage

## Implementation Notes

1. **Strategy Selection**:
   - Provide sensible defaults for each strategy
   - Document trade-offs for each strategy
   - Allow strategy composition for advanced use cases

2. **Performance**:
   - Minimize overhead in ShouldCheckpoint
   - Cache computed values where appropriate
   - Avoid unnecessary allocations

3. **Extensibility**:
   - Design interface to allow custom strategies
   - Make factory easy to extend
   - Support strategy parameters

4. **Testing**:
   - Test each strategy independently
   - Test integration with CheckpointManager
   - Test behavior under various conditions

## Dependencies on Other Specs

This spec depends on:
- **Checkpoint Configuration** (spec_2) for CheckpointConfig and CheckpointStrategy enum
- **Memory Tracking System** (spec_3) for MemoryTracker (used by MemoryAwareCheckpointStrategy)

## Estimated Implementation Time
45-60 minutes
