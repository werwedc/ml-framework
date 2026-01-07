using MlFramework.Inference.PagedAttention.Models;

namespace MlFramework.Inference.PagedAttention.Sharing;

/// <summary>
/// Manages block sharing between sequences for parallel decoding.
/// Multiple sequences can share the same blocks for common prefixes.
/// </summary>
public class BlockShareManager
{
    // Reference counting: blockId -> number of sequences using it
    private readonly Dictionary<int, int> _blockReferences;

    // Reverse mapping: (sequenceId, tokenIndex) -> blockId
    private readonly Dictionary<(int seqId, int tokenIdx), int> _sequenceBlockMap;

    // Shared blocks mapping: token content hash -> blockId
    private readonly Dictionary<string, int> _sharedBlocks;

    private readonly object _lock = new object();

    public BlockShareManager()
    {
        _blockReferences = new Dictionary<int, int>();
        _sequenceBlockMap = new Dictionary<(int, int), int>();
        _sharedBlocks = new Dictionary<string, int>();
    }

    /// <summary>
    /// Share a block between multiple sequences.
    /// </summary>
    /// <param name="blockId">ID of the block to share</param>
    /// <param name="sequenceIds">IDs of sequences that will share this block</param>
    public void ShareBlock(int blockId, List<int> sequenceIds)
    {
        lock (_lock)
        {
            // Initialize reference count if not exists
            if (!_blockReferences.ContainsKey(blockId))
            {
                _blockReferences[blockId] = 0;
            }

            // Increment reference count for each new sequence
            foreach (var seqId in sequenceIds)
            {
                _blockReferences[blockId]++;

                // Map sequence positions to the shared block
                // Note: This is simplified - in practice, you'd track token indices
                _sequenceBlockMap[(seqId, -1)] = blockId;
            }
        }
    }

    /// <summary>
    /// Release a sequence's reference to shared blocks.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence to release</param>
    /// <returns>List of blockIds that are now free (reference count = 0)</returns>
    public List<int> ReleaseSequence(int sequenceId)
    {
        lock (_lock)
        {
            var freedBlocks = new List<int>();

            // Find all blocks this sequence references
            var blocksToRelease = _sequenceBlockMap
                .Where(kvp => kvp.Key.Item1 == sequenceId)
                .Select(kvp => kvp.Value)
                .Distinct()
                .ToList();

            foreach (var blockId in blocksToRelease)
            {
                // Decrement reference count
                if (_blockReferences.ContainsKey(blockId))
                {
                    _blockReferences[blockId]--;

                    // If reference count reaches zero, block can be freed
                    if (_blockReferences[blockId] <= 0)
                    {
                        freedBlocks.Add(blockId);
                        _blockReferences.Remove(blockId);
                    }
                }
            }

            // Remove sequence mappings
            var toRemove = _sequenceBlockMap
                .Where(kvp => kvp.Key.Item1 == sequenceId)
                .ToList();

            foreach (var kvp in toRemove)
            {
                _sequenceBlockMap.Remove(kvp.Key);
            }

            return freedBlocks;
        }
    }

    /// <summary>
    /// Check if a block is shared by multiple sequences.
    /// </summary>
    public bool IsBlockShared(int blockId)
    {
        lock (_lock)
        {
            return _blockReferences.TryGetValue(blockId, out int count) && count > 1;
        }
    }

    /// <summary>
    /// Get the reference count for a block.
    /// </summary>
    public int GetReferenceCount(int blockId)
    {
        lock (_lock)
        {
            return _blockReferences.TryGetValue(blockId, out int count) ? count : 0;
        }
    }

    /// <summary>
    /// Get all blocks used by a sequence.
    /// </summary>
    public List<int> GetSequenceBlocks(int sequenceId)
    {
        lock (_lock)
        {
            return _sequenceBlockMap
                .Where(kvp => kvp.Key.Item1 == sequenceId)
                .Select(kvp => kvp.Value)
                .Distinct()
                .ToList();
        }
    }

    /// <summary>
    /// Get all sequences using a specific block.
    /// </summary>
    public List<int> GetBlockUsers(int blockId)
    {
        lock (_lock)
        {
            return _sequenceBlockMap
                .Where(kvp => kvp.Value == blockId)
                .Select(kvp => kvp.Key.Item1)
                .Distinct()
                .ToList();
        }
    }

    /// <summary>
    /// Get sharing statistics.
    /// </summary>
    public ShareStats GetStats()
    {
        lock (_lock)
        {
            int sharedBlocks = _blockReferences.Count(kvp => kvp.Value > 1);
            int totalReferences = _blockReferences.Values.Sum();
            double avgRefCount = _blockReferences.Count > 0
                ? (double)totalReferences / _blockReferences.Count
                : 0.0;

            return new ShareStats
            {
                TotalSharedBlocks = sharedBlocks,
                TotalBlocksReferenced = _blockReferences.Count,
                TotalReferences = totalReferences,
                AverageReferenceCount = avgRefCount
            };
        }
    }
}
