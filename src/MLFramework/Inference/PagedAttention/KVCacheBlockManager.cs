using System.Collections.Concurrent;
using MlFramework.Inference.PagedAttention.Models;
using MLFramework.Core;

namespace MlFramework.Inference.PagedAttention;

/// <summary>
/// Manages allocation and deallocation of memory blocks for KV cache storage.
/// Implements a free-list based allocator with fixed-size blocks.
/// </summary>
public class KVCacheBlockManager
{
    private readonly ConcurrentQueue<int> _freeBlockIds;
    private readonly ConcurrentDictionary<int, List<int>> _sequenceBlocks;
    private readonly ConcurrentDictionary<int, MemoryBlock> _blocks;
    private readonly int _blockSize;
    private readonly int _headDim;
    private readonly int _numLayers;
    private readonly int _numAttentionHeads;
    private readonly DeviceId _deviceId;
    private long _allocationCount;
    private long _deallocationCount;

    /// <summary>
    /// Number of tokens each block can store.
    /// </summary>
    public int BlockSize => _blockSize;

    /// <summary>
    /// Total number of blocks in the pool.
    /// </summary>
    public int TotalBlocks { get; }

    /// <summary>
    /// Number of blocks currently allocated.
    /// </summary>
    public int AllocatedBlocks => TotalBlocks - _freeBlockIds.Count;

    public KVCacheBlockManager(
        int totalBlocks,
        int blockSize,
        int headDim,
        int numLayers,
        int numAttentionHeads,
        DeviceId deviceId)
    {
        TotalBlocks = totalBlocks;
        _blockSize = blockSize;
        _headDim = headDim;
        _numLayers = numLayers;
        _numAttentionHeads = numAttentionHeads;
        _deviceId = deviceId;

        _freeBlockIds = new ConcurrentQueue<int>();
        _sequenceBlocks = new ConcurrentDictionary<int, List<int>>();
        _blocks = new ConcurrentDictionary<int, MemoryBlock>();

        // Initialize free block pool
        for (int i = 0; i < totalBlocks; i++)
        {
            _freeBlockIds.Enqueue(i);
        }
    }

    /// <summary>
    /// Allocate a block for a sequence.
    /// </summary>
    /// <param name="sequenceId">ID of the requesting sequence</param>
    /// <returns>Allocation result with block ID or error</returns>
    public BlockAllocationResult AllocateBlock(int sequenceId)
    {
        if (!_freeBlockIds.TryDequeue(out int blockId))
        {
            return BlockAllocationResult.Failed("No free blocks available");
        }

        var block = new MemoryBlock(blockId, _blockSize)
        {
            SequenceId = sequenceId
        };

        // Allocate GPU tensors for keys and values
        // Shape: [numLayers, numAttentionHeads, blockSize, headDim]
        // Note: For now, we'll skip actual tensor allocation as it requires the full Tensor API
        // block.KeyTensor = Tensor.Zeros(
        //     new[] { _numLayers, _numAttentionHeads, _blockSize, _headDim },
        //     _deviceId
        // );
        // block.ValueTensor = Tensor.Zeros(
        //     new[] { _numLayers, _numAttentionHeads, _blockSize, _headDim },
        //     _deviceId
        // );

        _blocks[blockId] = block;
        Interlocked.Increment(ref _allocationCount);

        // Track block for sequence
        _sequenceBlocks.AddOrUpdate(
            sequenceId,
            _ => new List<int> { blockId },
            (_, existing) =>
            {
                lock (existing)
                {
                    existing.Add(blockId);
                }
                return existing;
            }
        );

        return BlockAllocationResult.Successful(blockId);
    }

    /// <summary>
    /// Allocate multiple blocks for a sequence (for prefill phase).
    /// </summary>
    /// <param name="sequenceId">ID of the requesting sequence</param>
    /// <param name="count">Number of blocks to allocate</param>
    /// <returns>List of allocated block IDs or empty if insufficient blocks</returns>
    public List<int> AllocateBlocks(int sequenceId, int count)
    {
        var allocated = new List<int>();
        var locks = new List<int>();

        try
        {
            for (int i = 0; i < count; i++)
            {
                var result = AllocateBlock(sequenceId);
                if (!result.Success)
                {
                    // Rollback: free any blocks we allocated
                    foreach (var blockId in allocated)
                    {
                        FreeBlock(blockId);
                    }
                    return new List<int>();
                }
                allocated.Add(result.BlockId);
            }
            return allocated;
        }
        catch
        {
            // Rollback on exception
            foreach (var blockId in allocated)
            {
                FreeBlock(blockId);
            }
            throw;
        }
    }

    /// <summary>
    /// Free a single block back to the pool.
    /// </summary>
    public void FreeBlock(int blockId)
    {
        if (_blocks.TryGetValue(blockId, out var block))
        {
            block.Reset();
            // Note: Dispose calls will be added when Tensor API is available
            // block.KeyTensor?.Dispose();
            // block.ValueTensor?.Dispose();

            if (block.SequenceId.HasValue)
            {
                RemoveBlockFromSequence(block.SequenceId.Value, blockId);
            }

            _freeBlockIds.Enqueue(blockId);
            Interlocked.Increment(ref _deallocationCount);
        }
    }

    /// <summary>
    /// Free all blocks associated with a sequence.
    /// </summary>
    public void FreeSequenceBlocks(int sequenceId)
    {
        if (_sequenceBlocks.TryRemove(sequenceId, out var blockIds))
        {
            foreach (var blockId in blockIds)
            {
                FreeBlock(blockId);
            }
        }
    }

    /// <summary>
    /// Get all blocks allocated for a sequence.
    /// </summary>
    public List<int> GetSequenceBlocks(int sequenceId)
    {
        return _sequenceBlocks.TryGetValue(sequenceId, out var blocks)
            ? new List<int>(blocks)
            : new List<int>();
    }

    /// <summary>
    /// Get a specific block by ID.
    /// </summary>
    public MemoryBlock? GetBlock(int blockId)
    {
        return _blocks.TryGetValue(blockId, out var block) ? block : null;
    }

    /// <summary>
    /// Get current memory usage statistics.
    /// </summary>
    public BlockManagerStats GetStats()
    {
        var allocatedBlocks = AllocatedBlocks;
        int totalTokens = 0;
        int activeSequences = 0;

        foreach (var kvp in _blocks)
        {
            var block = kvp.Value;
            if (block.IsAllocated)
            {
                totalTokens += block.TokenCount;
            }
        }

        activeSequences = _sequenceBlocks.Count;

        double utilization = TotalBlocks > 0
            ? (allocatedBlocks * 100.0 / TotalBlocks)
            : 0.0;

        return new BlockManagerStats
        {
            TotalBlocks = TotalBlocks,
            AllocatedBlocks = allocatedBlocks,
            ActiveSequences = activeSequences,
            TotalTokens = totalTokens,
            MemoryUtilizationPercentage = utilization,
            AllocationCount = _allocationCount,
            DeallocationCount = _deallocationCount
        };
    }

    /// <summary>
    /// Check if sufficient blocks are available.
    /// </summary>
    public bool HasAvailableBlocks(int count = 1)
    {
        return _freeBlockIds.Count >= count;
    }

    private void RemoveBlockFromSequence(int sequenceId, int blockId)
    {
        if (_sequenceBlocks.TryGetValue(sequenceId, out var blockIds))
        {
            lock (blockIds)
            {
                blockIds.Remove(blockId);
                if (blockIds.Count == 0)
                {
                    _sequenceBlocks.TryRemove(sequenceId, out _);
                }
            }
        }
    }

    /// <summary>
    /// Dispose of all allocated tensors and resources.
    /// </summary>
    public void Dispose()
    {
        foreach (var kvp in _blocks)
        {
            // Note: Dispose calls will be added when Tensor API is available
            // kvp.Value.KeyTensor?.Dispose();
            // kvp.Value.ValueTensor?.Dispose();
        }
        _blocks.Clear();
    }
}
