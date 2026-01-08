using System;
using System.Collections.Generic;
using RitterTensor = RitterFramework.Core.Tensor;

namespace MLFramework.Checkpointing;

/// <summary>
/// Static class providing convenient methods for checkpointing functions
/// </summary>
public static class Checkpoint
{
    private static ICheckpointAdapter? _defaultCheckpointAdapter;
    private static IRecomputeAdapter? _defaultRecomputeAdapter;
    private static readonly object _lock = new object();

    /// <summary>
    /// Checkpoints a function during the forward pass
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="func">Function to checkpoint</param>
    /// <param name="checkpointAdapter">Checkpoint adapter (optional, uses default if null)</param>
    /// <param name="recomputeAdapter">Recompute adapter (optional, uses default if null)</param>
    /// <returns>Result of the function</returns>
    public static RitterTensor.Tensor CheckpointFunction(
        string layerId,
        Func<RitterTensor.Tensor> func,
        ICheckpointAdapter? checkpointAdapter = null,
        IRecomputeAdapter? recomputeAdapter = null)
    {
        if (string.IsNullOrEmpty(layerId))
            throw new ArgumentException("Layer ID cannot be null or empty", nameof(layerId));
        if (func == null)
            throw new ArgumentNullException(nameof(func));

        var adapter = checkpointAdapter ?? GetDefaultCheckpointAdapter();
        var engine = recomputeAdapter ?? GetDefaultRecomputeEngine();

        var checkpointFunc = new CheckpointFunction(
            layerId,
            func,
            null,
            adapter,
            engine);

        return checkpointFunc.Apply();
    }

    /// <summary>
    /// Creates a checkpointed version of a function
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="func">Function to checkpoint</param>
    /// <param name="backwardHook">Optional backward hook</param>
    /// <param name="checkpointAdapter">Checkpoint adapter (optional, uses default if null)</param>
    /// <param name="recomputeAdapter">Recompute adapter (optional, uses default if null)</param>
    /// <returns>CheckpointFunction instance</returns>
    public static CheckpointFunction CreateCheckpointFunction(
        string layerId,
        Func<RitterTensor.Tensor> func,
        Action<RitterTensor.Tensor>? backwardHook = null,
        ICheckpointAdapter? checkpointAdapter = null,
        IRecomputeAdapter? recomputeAdapter = null)
    {
        if (string.IsNullOrEmpty(layerId))
            throw new ArgumentException("Layer ID cannot be null or empty", nameof(layerId));
        if (func == null)
            throw new ArgumentNullException(nameof(func));

        var adapter = checkpointAdapter ?? GetDefaultCheckpointAdapter();
        var engine = recomputeAdapter ?? GetDefaultRecomputeEngine();

        return new CheckpointFunction(
            layerId,
            func,
            backwardHook,
            adapter,
            engine);
    }

    /// <summary>
    /// Sets the default checkpoint adapter
    /// </summary>
    /// <param name="adapter">Checkpoint adapter to use as default</param>
    public static void SetDefaultCheckpointAdapter(ICheckpointAdapter adapter)
    {
        if (adapter == null)
            throw new ArgumentNullException(nameof(adapter));

        lock (_lock)
        {
            _defaultCheckpointAdapter = adapter;
        }
    }

    /// <summary>
    /// Sets the default recompute adapter
    /// </summary>
    /// <param name="adapter">Recompute adapter to use as default</param>
    public static void SetDefaultRecomputeAdapter(IRecomputeAdapter adapter)
    {
        if (adapter == null)
            throw new ArgumentNullException(nameof(adapter));

        lock (_lock)
        {
            _defaultRecomputeAdapter = adapter;
        }
    }

    private static ICheckpointAdapter GetDefaultCheckpointAdapter()
    {
        lock (_lock)
        {
            if (_defaultCheckpointAdapter == null)
            {
                var manager = new CheckpointManager();
                _defaultCheckpointAdapter = new CheckpointAdapter(manager);
            }
            return _defaultCheckpointAdapter;
        }
    }

    private static IRecomputeAdapter GetDefaultRecomputeEngine()
    {
        lock (_lock)
        {
            if (_defaultRecomputeAdapter == null)
            {
                var engine = new RecomputationEngine();
                _defaultRecomputeAdapter = new RecomputeAdapter(engine);
            }
            return _defaultRecomputeAdapter;
        }
    }

    /// <summary>
    /// Resets the default adapters
    /// </summary>
    public static void ResetDefaults()
    {
        lock (_lock)
        {
            if (_defaultCheckpointAdapter != null)
            {
                if (_defaultCheckpointAdapter is CheckpointAdapter checkpointAdapter)
                {
                    checkpointAdapter.Dispose();
                }
                _defaultCheckpointAdapter = null;
            }

            if (_defaultRecomputeAdapter != null)
            {
                if (_defaultRecomputeAdapter is RecomputeAdapter recomputeAdapter)
                {
                    recomputeAdapter.Dispose();
                }
                _defaultRecomputeAdapter = null;
            }
        }
    }
}
