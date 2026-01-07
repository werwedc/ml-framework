using System.Collections.Concurrent;
using MlFramework.Inference.PagedAttention.Models;

namespace MlFramework.Inference.PagedAttention;

/// <summary>
/// Maintains logical-to-physical mapping for paged KV cache.
/// Maps (sequenceId, tokenIndex) to physical block locations.
/// </summary>
public class BlockTable
{
    // Map: sequenceId -> ordered list of block IDs for that sequence
    private readonly ConcurrentDictionary<int, List<int>> _sequenceBlockLists;

    // Map: (sequenceId, tokenIndex) -> block ID (for fast lookup)
    private readonly ConcurrentDictionary<(int seqId, int tokenIdx), int> _blockMapping;

    private readonly KVCacheBlockManager _blockManager;
    private readonly int _blockSize;

    /// <summary>
    /// Initialize block table with a reference to the block manager.
    /// </summary>
    public BlockTable(KVCacheBlockManager blockManager)
    {
        _blockManager = blockManager;
        _blockSize = blockManager.BlockSize;
        _sequenceBlockLists = new ConcurrentDictionary<int, List<int>>();
        _blockMapping = new ConcurrentDictionary<(int, int), int>();
    }

    /// <summary>
    /// Get the physical block ID for a specific token position.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="tokenIndex">Position of the token within the sequence</param>
    /// <returns>Block ID or -1 if not found</returns>
    public int GetBlock(int sequenceId, int tokenIndex)
    {
        return _blockMapping.TryGetValue((sequenceId, tokenIndex), out int blockId)
            ? blockId
            : -1;
    }

    /// <summary>
    /// Get all block IDs for a sequence, in order.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <returns>Ordered list of block IDs</returns>
    public List<int> GetSequenceBlocks(int sequenceId)
    {
        return _sequenceBlockLists.TryGetValue(sequenceId, out var blocks)
            ? new List<int>(blocks)
            : new List<int>();
    }

    /// <summary>
    /// Get the number of blocks allocated for a sequence.
    /// </summary>
    public int GetSequenceBlockCount(int sequenceId)
    {
        return _sequenceBlockLists.TryGetValue(sequenceId, out var blocks)
            ? blocks.Count
            : 0;
    }

    /// <summary>
    /// Get the number of tokens in a sequence.
    /// </summary>
    public int GetSequenceLength(int sequenceId)
    {
        if (_sequenceBlockLists.TryGetValue(sequenceId, out var blocks) && blocks.Count > 0)
        {
            var lastBlockId = blocks[^1];
            var block = _blockManager.GetBlock(lastBlockId);
            if (block != null)
            {
                return (blocks.Count - 1) * _blockSize + block.TokenCount;
            }
        }
        return 0;
    }

    /// <summary>
    /// Append a new block to a sequence's block list.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="blockId">Block ID to append</param>
    public void AppendBlock(int sequenceId, int blockId)
    {
        var currentLength = GetSequenceLength(sequenceId);
        var startToken = currentLength;

        _sequenceBlockLists.AddOrUpdate(
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

        // Map tokens in this block to the block ID
        // Tokens at positions [startToken, startToken + blockSize) map to this block
        for (int i = 0; i < _blockSize; i++)
        {
            var tokenIndex = startToken + i;
            _blockMapping[(sequenceId, tokenIndex)] = blockId;
        }

        // Update block metadata
        if (_blockManager.GetBlock(blockId) is { } block)
        {
            block.StartTokenIndex = startToken;
            block.SequenceId = sequenceId;
        }
    }

    /// <summary>
    /// Allocate and append a new block for a sequence.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <returns>Block ID of allocated block, or -1 if allocation failed</returns>
    public int AllocateAndAppendBlock(int sequenceId)
    {
        var allocationResult = _blockManager.AllocateBlock(sequenceId);
        if (!allocationResult.Success)
        {
            return -1;
        }

        AppendBlock(sequenceId, allocationResult.BlockId);
        return allocationResult.BlockId;
    }

    /// <summary>
    /// Remove all blocks for a sequence from the table.
    /// Does NOT deallocate blocks - that's the block manager's job.
    /// </summary>
    public void RemoveSequence(int sequenceId)
    {
        if (_sequenceBlockLists.TryRemove(sequenceId, out var blocks))
        {
            // Remove all mappings for this sequence
            foreach (var blockId in blocks)
            {
                if (_blockManager.GetBlock(blockId) is { } block)
                {
                    var startToken = block.StartTokenIndex;
                    for (int i = 0; i < _blockSize; i++)
                    {
                        var tokenIndex = startToken + i;
                        _blockMapping.TryRemove((sequenceId, tokenIndex), out _);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Clear all mappings for all sequences.
    /// </summary>
    public void Clear()
    {
        _sequenceBlockLists.Clear();
        _blockMapping.Clear();
    }

    /// <summary>
    /// Get all sequence IDs currently tracked.
    /// </summary>
    public IEnumerable<int> GetActiveSequenceIds()
    {
        return _sequenceBlockLists.Keys;
    }

    /// <summary>
    /// Check if a sequence exists in the table.
    /// </summary>
    public bool ContainsSequence(int sequenceId)
    {
        return _sequenceBlockLists.ContainsKey(sequenceId);
    }

    /// <summary>
    /// Get statistics about the block table.
    /// </summary>
    public BlockTableStats GetStats()
    {
        int totalSequences = _sequenceBlockLists.Count;
        int totalBlocks = 0;
        int totalTokens = 0;

        foreach (var kvp in _sequenceBlockLists)
        {
            var blocks = kvp.Value;
            totalBlocks += blocks.Count;
            totalTokens += GetSequenceLength(kvp.Key);
        }

        return new BlockTableStats
        {
            TotalSequences = totalSequences,
            TotalBlocks = totalBlocks,
            TotalTokens = totalTokens,
            AverageBlocksPerSequence = totalSequences > 0
                ? (double)totalBlocks / totalSequences
                : 0.0
        };
    }
}

/// <summary>
/// Statistics about the block table state.
/// </summary>
public class BlockTableStats
{
    public int TotalSequences { get; set; }
    public int TotalBlocks { get; set; }
    public int TotalTokens { get; set; }
    public double AverageBlocksPerSequence { get; set; }

    public override string ToString()
    {
        return $"BlockTableStats: Sequences={TotalSequences}, " +
               $"Blocks={TotalBlocks}, " +
               $"Tokens={TotalTokens}, " +
               $"AvgBlocksPerSeq={AverageBlocksPerSequence:F2}";
    }
}
