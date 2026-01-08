using System;
using System.Linq;

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
}

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

/// <summary>
/// Extension methods for CheckpointConfig
/// </summary>
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
