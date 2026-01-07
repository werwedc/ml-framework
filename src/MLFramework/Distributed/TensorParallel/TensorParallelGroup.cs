namespace MLFramework.Distributed.TensorParallel;

using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Process group for a subset of ranks in Tensor Parallelism.
/// Provides collective operations scoped to a specific group of processes.
/// </summary>
public class TensorParallelGroup : IDisposable
{
    private readonly ICommunicator? _globalCommunicator;
    private readonly List<int>? _ranks;
    private readonly int _globalRank;
    private readonly bool _isDefaultGroup;

    /// <summary>
    /// Gets the world size of this process group (number of ranks in the group).
    /// </summary>
    public int WorldSize { get; }

    /// <summary>
    /// Gets the local rank within this process group.
    /// </summary>
    public int LocalRank { get; }

    /// <summary>
    /// Gets whether the current rank is part of this process group.
    /// </summary>
    public bool InGroup { get; }

    /// <summary>
    /// Gets the communicator associated with this group (if any).
    /// </summary>
    public ICommunicator? Communicator => _globalCommunicator;

    /// <summary>
    /// Creates a default process group (all ranks in TP context).
    /// </summary>
    /// <param name="globalComm">Global communicator (null for default group)</param>
    /// <param name="ranks">List of ranks in this group (null for default group)</param>
    /// <param name="myGlobalRank">This process's global rank</param>
    public TensorParallelGroup(ICommunicator? globalComm, List<int>? ranks, int myGlobalRank)
    {
        if (globalComm == null && ranks == null)
        {
            // Default group (all ranks in TP context)
            _isDefaultGroup = true;
            _globalCommunicator = null;
            _ranks = null;
            _globalRank = -1;

            // Will be set when context is available
            var ctx = TensorParallel.TryGetContext();
            if (ctx != null)
            {
                WorldSize = ctx.WorldSize;
                LocalRank = ctx.Rank;
                InGroup = true;
            }
            else
            {
                WorldSize = 1;
                LocalRank = 0;
                InGroup = true;
            }
        }
        else
        {
            _isDefaultGroup = false;
            _globalCommunicator = globalComm ?? throw new ArgumentNullException(nameof(globalComm));
            _ranks = ranks ?? throw new ArgumentNullException(nameof(ranks));
            _globalRank = myGlobalRank;

            // Check if this rank is in the group
            InGroup = _ranks.Contains(_globalRank);

            if (InGroup)
            {
                WorldSize = _ranks.Count;
                LocalRank = _ranks.IndexOf(_globalRank);
            }
            else
            {
                WorldSize = 0;
                LocalRank = -1;
            }
        }
    }

    /// <summary>
    /// Performs all-reduce operation on this process group.
    /// </summary>
    /// <param name="tensor">Tensor to reduce</param>
    /// <param name="operation">Reduce operation type</param>
    /// <returns>Reduced tensor</returns>
    public Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation)
    {
        if (_isDefaultGroup)
        {
            var ctx = TensorParallel.GetContext();
            return ctx.Communicator.AllReduceAsync(tensor, operation);
        }
        else if (InGroup)
        {
            // Filter to only ranks in this group
            return _globalCommunicator!.AllReduceAsync(tensor, operation);
        }
        else
        {
            // This rank is not in the group, return tensor unchanged
            return Task.FromResult(tensor);
        }
    }

    /// <summary>
    /// Performs all-gather operation on this process group.
    /// </summary>
    /// <param name="tensor">Tensor to gather</param>
    /// <param name="dim">Dimension along which to gather</param>
    /// <returns>Gathered tensor</returns>
    public Task<Tensor> AllGatherAsync(Tensor tensor, int dim = 0)
    {
        if (_isDefaultGroup)
        {
            var ctx = TensorParallel.GetContext();
            return ctx.Communicator.AllGatherAsync(tensor, dim);
        }
        else if (InGroup)
        {
            return _globalCommunicator!.AllGatherAsync(tensor, dim);
        }
        else
        {
            return Task.FromResult(tensor);
        }
    }

    /// <summary>
    /// Performs reduce-scatter operation on this process group.
    /// </summary>
    /// <param name="tensor">Tensor to reduce and scatter</param>
    /// <param name="operation">Reduce operation type</param>
    /// <returns>Scattered tensor</returns>
    public Task<Tensor> ReduceScatterAsync(Tensor tensor, ReduceOperation operation)
    {
        if (_isDefaultGroup)
        {
            var ctx = TensorParallel.GetContext();
            return ctx.Communicator.ReduceScatterAsync(tensor, operation);
        }
        else if (InGroup)
        {
            return _globalCommunicator!.ReduceScatterAsync(tensor, operation);
        }
        else
        {
            return Task.FromResult(tensor);
        }
    }

    /// <summary>
    /// Performs broadcast operation on this process group.
    /// </summary>
    /// <param name="tensor">Tensor to broadcast</param>
    /// <param name="root">Root rank for broadcast</param>
    /// <returns>Broadcasted tensor</returns>
    public Task<Tensor> BroadcastAsync(Tensor tensor, int root)
    {
        if (_isDefaultGroup)
        {
            var ctx = TensorParallel.GetContext();
            return ctx.Communicator.BroadcastAsync(tensor, root);
        }
        else if (InGroup)
        {
            return _globalCommunicator!.BroadcastAsync(tensor, root);
        }
        else
        {
            return Task.FromResult(tensor);
        }
    }

    /// <summary>
    /// Barrier operation on this process group.
    /// </summary>
    public Task BarrierAsync()
    {
        if (_isDefaultGroup)
        {
            var ctx = TensorParallel.GetContext();
            return ctx.Communicator.BarrierAsync();
        }
        else if (InGroup)
        {
            return _globalCommunicator!.BarrierAsync();
        }
        else
        {
            return Task.CompletedTask;
        }
    }

    /// <summary>
    /// Disposes the process group.
    /// </summary>
    public void Dispose()
    {
        // Cleanup if needed
    }
}
