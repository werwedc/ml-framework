namespace MLFramework.Distributed.TensorParallel;

using MLFramework.Distributed.Communication;
using System;
using System.Collections.Generic;
using System.Threading;

/// <summary>
/// Context manager for Tensor Parallelism.
/// Manages global state (world size, rank, communicator) with RAII-style initialization and cleanup.
/// </summary>
public class TensorParallelContext : IDisposable
{
    private static TensorParallelContext? _current;
    private static readonly object _lock = new object();

    private readonly ICommunicator _communicator;
    private readonly int _worldSize;
    private readonly int _rank;
    private readonly bool _ownsCommunicator;
    private readonly List<TensorParallelGroup> _processGroups;

    /// <summary>
    /// Gets the world size (total number of processes/ranks) for TP.
    /// </summary>
    public int WorldSize => _worldSize;

    /// <summary>
    /// Gets the rank of this process (0 to WorldSize-1).
    /// </summary>
    public int Rank => _rank;

    /// <summary>
    /// Gets the communicator used for collective operations.
    /// </summary>
    public ICommunicator Communicator => _communicator;

    /// <summary>
    /// Gets the currently active TensorParallel context (null if not initialized).
    /// </summary>
    public static TensorParallelContext? Current => _current;

    private TensorParallelContext(ICommunicator communicator, bool ownsCommunicator)
    {
        _communicator = communicator;
        _worldSize = communicator.WorldSize;
        _rank = communicator.Rank;
        _ownsCommunicator = ownsCommunicator;
        _processGroups = new List<TensorParallelGroup>();

        // Set as current context
        lock (_lock)
        {
            _current = this;
        }
    }

    /// <summary>
    /// Initializes tensor parallelism with a new communicator.
    /// </summary>
    /// <param name="worldSize">Total number of processes/ranks</param>
    /// <param name="rank">Rank of this process (0 to worldSize-1)</param>
    /// <param name="backend">Backend type ("mock", "nccl", etc.)</param>
    /// <returns>New TensorParallelContext instance</returns>
    public static TensorParallelContext Initialize(int worldSize, int rank, string backend = "mock")
    {
        var config = new Dictionary<string, object>
        {
            ["world_size"] = worldSize,
            ["rank"] = rank
        };
        var communicator = CommunicatorFactory.Create(backend, config);
        return new TensorParallelContext(communicator, ownsCommunicator: true);
    }

    /// <summary>
    /// Initializes tensor parallelism with an existing communicator.
    /// </summary>
    /// <param name="communicator">Existing communicator instance</param>
    /// <returns>New TensorParallelContext instance</returns>
    public static TensorParallelContext Initialize(ICommunicator communicator)
    {
        return new TensorParallelContext(communicator, ownsCommunicator: false);
    }

    /// <summary>
    /// Creates a process group for a subset of ranks.
    /// </summary>
    /// <param name="ranks">List of ranks in this process group</param>
    /// <returns>TensorParallelGroup instance</returns>
    public TensorParallelGroup CreateProcessGroup(List<int> ranks)
    {
        var group = new TensorParallelGroup(_communicator, ranks, _rank);
        _processGroups.Add(group);
        return group;
    }

    /// <summary>
    /// Gets the default TP group (all ranks).
    /// </summary>
    public TensorParallelGroup DefaultGroup { get; } = new TensorParallelGroup(null, null, -1);

    /// <summary>
    /// Disposes the context and cleans up resources.
    /// </summary>
    public void Dispose()
    {
        // Cleanup process groups
        foreach (var group in _processGroups)
        {
            group.Dispose();
        }
        _processGroups.Clear();

        // Dispose communicator if we own it
        if (_ownsCommunicator)
        {
            _communicator.Dispose();
        }

        // Clear current context
        lock (_lock)
        {
            if (_current == this)
            {
                _current = null;
            }
        }
    }
}
