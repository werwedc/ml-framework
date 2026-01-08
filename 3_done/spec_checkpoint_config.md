# Spec: Checkpoint Configuration

## Overview
Implement configuration classes that define checkpointing strategies, including enum types for strategies, configuration data structures, and validation logic.

## Classes and Enums

### Location
`src/MLFramework/Checkpointing/CheckpointConfig.cs`

### Enum: CheckpointStrategy

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Defines different checkpointing strategies for memory optimization
/// </summary>
public enum CheckpointStrategy
{
    /// <summary>
    /// Store activations at fixed intervals (e.g., every N layers)
    /// </summary>
    Interval,

    /// <summary>
    /// Manually select specific layers to checkpoint
    /// </summary>
    Selective,

    /// <summary>
    /// Automatically checkpoint layers based on activation size
    /// </summary>
    SizeBased,

    /// <summary>
    /// Dynamically adjust checkpointing based on available memory
    /// </summary>
    MemoryAware,

    /// <summary>
    /// Use smart heuristics to determine optimal checkpointing
    /// </summary>
    Smart
}
```

### Class: CheckpointConfig

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Configuration for activation checkpointing behavior
/// </summary>
public class CheckpointConfig
{
    /// <summary>
    /// The checkpointing strategy to use
    /// </summary>
    public CheckpointStrategy Strategy { get; set; } = CheckpointStrategy.Interval;

    /// <summary>
    /// Interval for Interval strategy (checkpoint every N layers)
    /// </summary>
    public int Interval { get; set; } = 2;

    /// <summary>
    /// List of specific layer IDs to checkpoint for Selective strategy
    /// </summary>
    public string[] CheckpointLayers { get; set; } = Array.Empty<string>();

    /// <summary>
    /// List of layer IDs to exclude from checkpointing
    /// </summary>
    public string[] ExcludeLayers { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Minimum activation size in bytes to trigger checkpointing for SizeBased strategy
    /// </summary>
    public long MinActivationSizeBytes { get; set; } = 1024 * 1024; // 1MB default

    /// <summary>
    /// Maximum memory percentage to use for checkpoints (0.0 to 1.0) for MemoryAware strategy
    /// </summary>
    public float MaxMemoryPercentage { get; set; } = 0.8f; // 80% default

    /// <summary>
    /// Whether to enable recomputation caching
    /// </summary>
    public bool EnableRecomputationCache { get; set; } = true;

    /// <summary>
    /// Maximum cache size for recomputed activations (in bytes)
    /// </summary>
    public long MaxRecomputationCacheSize { get; set; } = 100 * 1024 * 1024; // 100MB default

    /// <summary>
    /// Whether to use asynchronous recomputation where possible
    /// </summary>
    public bool UseAsyncRecomputation { get; set; } = false;

    /// <summary>
    /// Whether to track detailed statistics
    /// </summary>
    public bool TrackStatistics { get; set; } = true;

    /// <summary>
    /// Validates the configuration and throws if invalid
    /// </summary>
    /// <exception cref="ArgumentException">Thrown if configuration is invalid</exception>
    public void Validate();
}
```

## Implementation Details

### CheckpointConfig.Validate()

```csharp
public void Validate()
{
    // Validate Strategy
    if (!Enum.IsDefined(typeof(CheckpointStrategy), Strategy))
    {
        throw new ArgumentException($"Invalid checkpoint strategy: {Strategy}");
    }

    // Validate Interval
    if (Interval <= 0)
    {
        throw new ArgumentException("Interval must be greater than 0");
    }

    // Validate MaxMemoryPercentage
    if (MaxMemoryPercentage <= 0 || MaxMemoryPercentage > 1.0f)
    {
        throw new ArgumentException("MaxMemoryPercentage must be between 0 and 1");
    }

    // Validate MinActivationSizeBytes
    if (MinActivationSizeBytes <= 0)
    {
        throw new ArgumentException("MinActivationSizeBytes must be greater than 0");
    }

    // Validate MaxRecomputationCacheSize
    if (MaxRecomputationCacheSize < 0)
    {
        throw new ArgumentException("MaxRecomputationCacheSize cannot be negative");
    }

    // Validate CheckpointLayers (if provided)
    foreach (var layer in CheckpointLayers)
    {
        if (string.IsNullOrWhiteSpace(layer))
        {
            throw new ArgumentException("Checkpoint layer IDs cannot be null or whitespace");
        }
    }

    // Validate ExcludeLayers (if provided)
    foreach (var layer in ExcludeLayers)
    {
        if (string.IsNullOrWhiteSpace(layer))
        {
            throw new ArgumentException("Exclude layer IDs cannot be null or whitespace");
        }
    }

    // Check for overlaps between CheckpointLayers and ExcludeLayers
    var overlap = CheckpointLayers.Intersect(ExcludeLayers).ToList();
    if (overlap.Count > 0)
    {
        throw new ArgumentException(
            $"Layers cannot be both checkpointed and excluded: {string.Join(", ", overlap)}");
    }
}
```

### Default Configuration

```csharp
/// <summary>
/// Creates a default checkpoint configuration (interval-based, every 2 layers)
/// </summary>
public static CheckpointConfig Default => new CheckpointConfig();

/// <summary>
/// Creates an aggressive configuration (checkpoint every 4 layers, ~80% memory savings)
/// </summary>
public static CheckpointConfig Aggressive => new CheckpointConfig
{
    Strategy = CheckpointStrategy.Interval,
    Interval = 4,
    EnableRecomputationCache = true,
    TrackStatistics = true
};

/// <summary>
/// Creates a conservative configuration (checkpoint every 2 layers, ~50% memory savings)
/// </summary>
public static CheckpointConfig Conservative => new CheckpointConfig
{
    Strategy = CheckpointStrategy.Interval,
    Interval = 2,
    EnableRecomputationCache = true,
    TrackStatistics = true
};

/// <summary>
/// Creates a memory-aware configuration (dynamically adjusts based on available memory)
/// </summary>
public static CheckpointConfig MemoryAware => new CheckpointConfig
{
    Strategy = CheckpointStrategy.MemoryAware,
    MaxMemoryPercentage = 0.75f,
    EnableRecomputationCache = true,
    TrackStatistics = true
};
```

## Configuration Builders

### Class: CheckpointConfigBuilder

```csharp
namespace MLFramework.Checkpointing;

/// <summary>
/// Builder pattern for creating CheckpointConfig instances
/// </summary>
public class CheckpointConfigBuilder
{
    private readonly CheckpointConfig _config = new CheckpointConfig();

    public CheckpointConfigBuilder WithStrategy(CheckpointStrategy strategy)
    {
        _config.Strategy = strategy;
        return this;
    }

    public CheckpointConfigBuilder WithInterval(int interval)
    {
        _config.Interval = interval;
        return this;
    }

    public CheckpointConfigBuilder WithCheckpointLayers(string[] layers)
    {
        _config.CheckpointLayers = layers;
        return this;
    }

    public CheckpointConfigBuilder WithExcludeLayers(string[] layers)
    {
        _config.ExcludeLayers = layers;
        return this;
    }

    public CheckpointConfigBuilder WithMinActivationSizeBytes(long size)
    {
        _config.MinActivationSizeBytes = size;
        return this;
    }

    public CheckpointConfigBuilder WithMaxMemoryPercentage(float percentage)
    {
        _config.MaxMemoryPercentage = percentage;
        return this;
    }

    public CheckpointConfigBuilder EnableRecomputationCache(bool enable)
    {
        _config.EnableRecomputationCache = enable;
        return this;
    }

    public CheckpointConfigBuilder WithMaxRecomputationCacheSize(long size)
    {
        _config.MaxRecomputationCacheSize = size;
        return this;
    }

    public CheckpointConfigBuilder UseAsyncRecomputation(bool enable)
    {
        _config.UseAsyncRecomputation = enable;
        return this;
    }

    public CheckpointConfigBuilder TrackStatistics(bool track)
    {
        _config.TrackStatistics = track;
        return this;
    }

    public CheckpointConfig Build()
    {
        _config.Validate();
        return _config;
    }
}
```

## Extension Methods

### CheckpointConfigExtensions

```csharp
namespace MLFramework.Checkpointing;

public static class CheckpointConfigExtensions
{
    /// <summary>
    /// Creates a copy of the configuration with the specified strategy
    /// </summary>
    public static CheckpointConfig WithStrategy(this CheckpointConfig config, CheckpointStrategy strategy)
    {
        var newConfig = config.Clone();
        newConfig.Strategy = strategy;
        return newConfig;
    }

    /// <summary>
    /// Creates a copy of the configuration with the specified interval
    /// </summary>
    public static CheckpointConfig WithInterval(this CheckpointConfig config, int interval)
    {
        var newConfig = config.Clone();
        newConfig.Interval = interval;
        return newConfig;
    }

    /// <summary>
    /// Creates a deep copy of the configuration
    /// </summary>
    public static CheckpointConfig Clone(this CheckpointConfig config)
    {
        return new CheckpointConfig
        {
            Strategy = config.Strategy,
            Interval = config.Interval,
            CheckpointLayers = (string[])config.CheckpointLayers.Clone(),
            ExcludeLayers = (string[])config.ExcludeLayers.Clone(),
            MinActivationSizeBytes = config.MinActivationSizeBytes,
            MaxMemoryPercentage = config.MaxMemoryPercentage,
            EnableRecomputationCache = config.EnableRecomputationCache,
            MaxRecomputationCacheSize = config.MaxRecomputationCacheSize,
            UseAsyncRecomputation = config.UseAsyncRecomputation,
            TrackStatistics = config.TrackStatistics
        };
    }
}
```

## Testing Requirements

### Unit Tests

1. **CheckpointStrategy Enum Tests**
   - [ ] Verify all enum values are defined correctly
   - [ ] Test enum serialization/deserialization (if applicable)

2. **CheckpointConfig Validation Tests**
   - [ ] Valid configuration passes validation
   - [ ] Invalid interval throws exception
   - [ ] Invalid MaxMemoryPercentage throws exception
   - [ ] Negative MinActivationSizeBytes throws exception
   - [ ] Negative MaxRecomputationCacheSize throws exception
   - [ ] Empty layer IDs throw exception
   - [ ] Overlapping CheckpointLayers and ExcludeLayers throw exception
   - [ ] Invalid strategy enum throws exception

3. **Default Configuration Tests**
   - [ ] Default configuration is valid
   - [ ] Default uses Interval strategy with interval 2
   - [ ] Aggressive configuration is valid with interval 4
   - [ ] Conservative configuration is valid with interval 2
   - [ ] MemoryAware configuration uses MemoryAware strategy

4. **Builder Pattern Tests**
   - [ ] Builder creates valid configuration
   - [ ] Fluent API works correctly
   - [ ] Builder can set all properties
   - [ ] Build() validates final configuration

5. **Extension Method Tests**
   - [ ] WithStrategy() creates correct copy
   - [ ] WithInterval() creates correct copy
   - [ ] Clone() creates deep copy
   - [ ] Clone() does not affect original

6. **Edge Cases**
   - [ ] Handle empty arrays for layer lists
   - [ ] Handle very large interval values
   - [ ] Handle boundary values for MaxMemoryPercentage (0, 1)
   - [ ] Handle MinActivationSizeBytes of 1 byte

## Implementation Notes

1. **Immutability Considerations**:
   - Consider making CheckpointConfig immutable for thread safety
   - Use builder pattern for constructing instances

2. **Validation Strategy**:
   - Validate eagerly in Build() or lazily before use
   - Provide clear error messages

3. **Serialization**:
   - Consider adding serialization support for saving/loading configs
   - Support JSON or YAML formats

4. **Extensibility**:
   - Design to allow adding new strategies without breaking existing code
   - Consider using strategy pattern for complex validation logic

## Dependencies on Other Specs

This spec is independent and can be implemented first. Other specs will depend on it:
- **Checkpoint Strategies** (spec_6) will use these configuration classes
- **Checkpoint API Extensions** (spec_7) will use these configuration classes

## Estimated Implementation Time
30-45 minutes
