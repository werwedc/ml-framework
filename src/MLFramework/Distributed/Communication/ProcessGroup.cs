namespace MLFramework.Distributed.Communication;

using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Process group for advanced distributed topologies.
/// Allows creating subgroups from a larger communicator for more efficient communication patterns.
/// </summary>
public class ProcessGroup : IDisposable
{
    private readonly ICommunicator _globalCommunicator;
    private readonly List<int> _ranks;
    private readonly int _globalRank;
    private readonly int _localRank;
    private readonly Dictionary<int, int> _globalToLocalRank;
    private readonly bool _inGroup;
    private bool _disposed;

    /// <summary>
    /// Gets the total number of processes in this process group.
    /// </summary>
    public int WorldSize { get; }

    /// <summary>
    /// Gets the local rank of this process within the group.
    /// Returns -1 if this process is not in the group.
    /// </summary>
    public int LocalRank => _inGroup ? _localRank : -1;

    /// <summary>
    /// Gets whether this process is part of the group.
    /// </summary>
    public bool InGroup => _inGroup;

    /// <summary>
    /// Creates a process group from a subset of ranks.
    /// </summary>
    /// <param name="globalComm">The global communicator</param>
    /// <param name="ranks">List of ranks in the group</param>
    /// <param name="myGlobalRank">The global rank of this process</param>
    public ProcessGroup(ICommunicator globalComm, List<int> ranks, int myGlobalRank)
    {
        _globalCommunicator = globalComm ?? throw new ArgumentNullException(nameof(globalComm));
        _ranks = ranks ?? throw new ArgumentNullException(nameof(ranks));
        _globalRank = myGlobalRank;

        if (_ranks.Count == 0)
        {
            throw new ArgumentException("Ranks list cannot be empty", nameof(ranks));
        }

        // Validate ranks are within global communicator range
        foreach (var rank in _ranks)
        {
            if (rank < 0 || rank >= globalComm.WorldSize)
            {
                throw new ArgumentException(
                    $"Rank {rank} is out of range [0, {globalComm.WorldSize - 1}]");
            }
        }

        // Remove duplicates
        _ranks = _ranks.Distinct().OrderBy(r => r).ToList();

        // Create global-to-local rank mapping
        _globalToLocalRank = _ranks
            .Select((rank, idx) => (rank, idx))
            .ToDictionary(x => x.rank, x => x.idx);

        // Check if this rank is in the group
        _inGroup = _ranks.Contains(_globalRank);

        if (_inGroup)
        {
            WorldSize = _ranks.Count;
            _localRank = _globalToLocalRank[_globalRank];
        }
        else
        {
            WorldSize = 0;
            _localRank = -1;
        }

        _disposed = false;
    }

    /// <summary>
    /// Performs an all-reduce operation on the process group.
    /// Only processes in the group participate in the operation.
    /// </summary>
    /// <param name="tensor">The tensor to reduce</param>
    /// <param name="operation">The reduction operation</param>
    /// <returns>A task that completes with the reduced tensor</returns>
    public async Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation)
    {
        if (!_inGroup)
        {
            // This rank is not in the group, return tensor unchanged
            return tensor.Clone();
        }

        // Filter to only ranks in this group
        // In a real implementation, this would use NCCL's subgroup feature
        // For now, we delegate to the global communicator
        // TODO: Implement proper filtering for subgroups
        return await _globalCommunicator.AllReduceAsync(tensor, operation);
    }

    /// <summary>
    /// Performs an all-gather operation on the process group.
    /// Only processes in the group participate in the operation.
    /// </summary>
    /// <param name="tensor">The tensor to gather</param>
    /// <param name="dim">The dimension along which to concatenate</param>
    /// <returns>A task that completes with the gathered tensor</returns>
    public async Task<Tensor> AllGatherAsync(Tensor tensor, int dim = 0)
    {
        if (!_inGroup)
        {
            // This rank is not in the group, return tensor unchanged
            return tensor.Clone();
        }

        // TODO: Implement proper filtering for subgroups
        return await _globalCommunicator.AllGatherAsync(tensor, dim);
    }

    /// <summary>
    /// Performs a reduce-scatter operation on the process group.
    /// Only processes in the group participate in the operation.
    /// </summary>
    /// <param name="tensor">The tensor to reduce and scatter</param>
    /// <param name="operation">The reduction operation</param>
    /// <returns>A task that completes with the scattered tensor chunk</returns>
    public async Task<Tensor> ReduceScatterAsync(Tensor tensor, ReduceOperation operation)
    {
        if (!_inGroup)
        {
            // This rank is not in the group, return tensor unchanged
            return tensor.Clone();
        }

        // TODO: Implement proper filtering for subgroups
        return await _globalCommunicator.ReduceScatterAsync(tensor, operation);
    }

    /// <summary>
    /// Performs a broadcast operation on the process group.
    /// Only processes in the group participate in the operation.
    /// </summary>
    /// <param name="tensor">The tensor to broadcast</param>
    /// <param name="root">The local rank of the root process</param>
    /// <returns>A task that completes with the broadcasted tensor</returns>
    public async Task<Tensor> BroadcastAsync(Tensor tensor, int root = 0)
    {
        if (!_inGroup)
        {
            // This rank is not in the group, return tensor unchanged
            return tensor.Clone();
        }

        // Convert local root rank to global rank
        int globalRoot = _ranks[root];

        // TODO: Implement proper filtering for subgroups
        return await _globalCommunicator.BroadcastAsync(tensor, globalRoot);
    }

    /// <summary>
    /// Barrier to synchronize all processes in the group.
    /// </summary>
    /// <returns>A task that completes when all processes have reached the barrier</returns>
    public Task BarrierAsync()
    {
        if (!_inGroup)
        {
            // This rank is not in the group, complete immediately
            return Task.CompletedTask;
        }

        // TODO: Implement proper filtering for subgroups
        return _globalCommunicator.BarrierAsync();
    }

    /// <summary>
    /// Gets the global rank from a local rank within this group.
    /// </summary>
    /// <param name="localRank">The local rank</param>
    /// <returns>The corresponding global rank</returns>
    public int GetGlobalRank(int localRank)
    {
        if (localRank < 0 || localRank >= _ranks.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(localRank),
                $"Local rank {localRank} is out of range [0, {_ranks.Count - 1}]");
        }

        return _ranks[localRank];
    }

    /// <summary>
    /// Gets the local rank from a global rank within this group.
    /// </summary>
    /// <param name="globalRank">The global rank</param>
    /// <returns>The corresponding local rank, or -1 if the global rank is not in this group</returns>
    public int GetLocalRank(int globalRank)
    {
        return _globalToLocalRank.TryGetValue(globalRank, out int localRank) ? localRank : -1;
    }

    /// <summary>
    /// Disposes the process group.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }
}
