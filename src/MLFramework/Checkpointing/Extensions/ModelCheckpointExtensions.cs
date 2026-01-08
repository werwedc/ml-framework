using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Extension methods for checkpointing models
/// </summary>
public static class ModelCheckpointExtensions
{
    private static readonly Dictionary<object, CheckpointModelInfo> _checkpointedModels = new Dictionary<object, CheckpointModelInfo>();

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
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (_checkpointedModels.TryGetValue(model, out var info))
        {
            info.IsEnabled = false;
        }

        return model;
    }

    /// <summary>
    /// Gets checkpointing statistics for the model
    /// </summary>
    /// <typeparam name="T">Type of the model</typeparam>
    /// <param name="model">The model to get statistics for</param>
    /// <returns>Checkpointing statistics</returns>
    public static CheckpointStatistics GetCheckpointStatistics<T>(this T model) where T : class
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        if (_checkpointedModels.TryGetValue(model, out var info))
        {
            var memoryStats = info.CheckpointManager.GetMemoryStats();
            var recomputeStats = info.RecomputeEngine.GetStats();

            return new CheckpointStatistics
            {
                MemoryUsed = memoryStats.CurrentMemoryUsed,
                PeakMemoryUsed = memoryStats.PeakMemoryUsed,
                RecomputationCount = recomputeStats.TotalRecomputations,
                RecomputationTimeMs = recomputeStats.TotalRecomputationTimeMs,
                IsCheckpointingEnabled = info.IsEnabled,
                CheckpointCount = memoryStats.CheckpointCount,
                Timestamp = DateTime.UtcNow
            };
        }

        return new CheckpointStatistics
        {
            IsCheckpointingEnabled = false,
            Timestamp = DateTime.UtcNow
        };
    }

    private static T CheckpointModel<T>(T model, CheckpointConfig config) where T : class
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));
        if (config == null)
            throw new ArgumentNullException(nameof(config));

        var checkpointManager = new CheckpointManager();
        var recomputeEngine = new RecomputationEngine();

        var info = new CheckpointModelInfo
        {
            Config = config,
            CheckpointManager = checkpointManager,
            RecomputeEngine = recomputeEngine,
            IsEnabled = true
        };

        _checkpointedModels[model] = info;

        // In a real implementation, this would wrap the model's forward pass
        // to automatically checkpoint activations based on the strategy
        // For now, we just store the checkpoint info
        WrapModelProperties(model, config, checkpointManager, recomputeEngine);

        return model;
    }

    private static void WrapModelProperties<T>(T model, CheckpointConfig config, CheckpointManager manager, RecomputationEngine engine)
    {
        // This is a placeholder for a more sophisticated implementation
        // that would inspect the model's structure and wrap layers
        // In practice, this would require reflection or a custom model interface
    }

    private class CheckpointModelInfo
    {
        public CheckpointConfig Config { get; set; } = null!;
        public CheckpointManager CheckpointManager { get; set; } = null!;
        public RecomputationEngine RecomputeEngine { get; set; } = null!;
        public bool IsEnabled { get; set; }
    }
}
