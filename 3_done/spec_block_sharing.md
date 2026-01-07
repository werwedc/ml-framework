# Spec: Block Sharing for Parallel Decoding

## Overview
Implement block sharing mechanisms for parallel decoding techniques such as beam search and speculative decoding. Multiple candidate sequences can share prefix blocks, significantly reducing memory usage.

## Target Directory
`src/MlFramework/Inference/PagedAttention/Sharing/`

## Classes to Implement

### BlockShareManager
```csharp
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
            // Increment reference count for each new sequence
            foreach (var seqId in sequenceIds)
            {
                if (!_sequenceBlockMap.ContainsValue(blockId))
                {
                    _blockReferences[blockId] = 1;
                }
                else
                {
                    _blockReferences[blockId]++;
                }

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

/// <summary>
/// Statistics about block sharing.
/// </summary>
public class ShareStats
{
    public int TotalSharedBlocks { get; set; }
    public int TotalBlocksReferenced { get; set; }
    public int TotalReferences { get; set; }
    public double AverageReferenceCount { get; set; }

    public override string ToString()
    {
        return $"ShareStats: " +
               $"Shared={TotalSharedBlocks}, " +
               $"Referenced={TotalBlocksReferenced}, " +
               $"AvgRefCount={AverageReferenceCount:F2}";
    }
}
```

### BeamSearchBlockSharing
```csharp
namespace MlFramework.Inference.PagedAttention.Sharing;

/// <summary>
/// Implements block sharing for beam search decoding.
/// All beams share prefix blocks until they diverge.
/// </summary>
public class BeamSearchBlockSharing
{
    private readonly BlockShareManager _shareManager;
    private readonly int _beamWidth;
    private readonly Dictionary<int, BeamInfo> _beamInfos;

    public BeamSearchBlockSharing(BlockShareManager shareManager, int beamWidth)
    {
        _shareManager = shareManager;
        _beamWidth = beamWidth;
        _beamInfos = new Dictionary<int, BeamInfo>();
    }

    /// <summary>
    /// Initialize beams for beam search.
    /// </summary>
    /// <param name="baseSequenceId">ID of the base sequence</param>
    /// <param name="prefixLength">Length of the shared prefix</param>
    /// <returns>List of beam sequence IDs</returns>
    public List<int> InitializeBeams(int baseSequenceId, int prefixLength)
    {
        var beamIds = new List<int>();

        // Create beam IDs
        for (int i = 0; i < _beamWidth; i++)
        {
            int beamId = baseSequenceId * 1000 + i; // Simple ID generation
            beamIds.Add(beamId);

            _beamInfos[beamId] = new BeamInfo
            {
                BeamIndex = i,
                BaseSequenceId = baseSequenceId,
                DivergencePoint = prefixLength
            };
        }

        // Share prefix blocks among all beams
        SharePrefixBlocks(baseSequenceId, beamIds, prefixLength);

        return beamIds;
    }

    /// <summary>
    /// Share prefix blocks among beams.
    /// </summary>
    private void SharePrefixBlocks(int baseSequenceId, List<int> beamIds, int prefixLength)
    {
        // In a real implementation, this would:
        // 1. Identify all blocks covering the prefix [0, prefixLength)
        // 2. Share these blocks among all beams
        // 3. Update reference counts

        // Simplified implementation:
        // Assume we have block IDs for the prefix
        var prefixBlockIds = GetPrefixBlockIds(baseSequenceId, prefixLength);

        foreach (var blockId in prefixBlockIds)
        {
            _shareManager.ShareBlock(blockId, beamIds);
        }
    }

    /// <summary>
    /// Handle beam divergence (when beams generate different tokens).
    /// </summary>
    /// <param name="beamId">ID of the beam that diverged</param>
    /// <param name="divergencePoint">Token index where divergence occurred</param>
    public void OnBeamDivergence(int beamId, int divergencePoint)
    {
        if (_beamInfos.TryGetValue(beamId, out var beamInfo))
        {
            beamInfo.DivergencePoint = divergencePoint;

            // At divergence, we need to allocate new blocks
            // Blocks after the divergence point are no longer shared
            // Implementation would allocate unique blocks for this beam
        }
    }

    /// <summary>
    /// Clean up beams after beam search completes.
    /// </summary>
    /// <param name="beamIds">List of beam IDs to clean up</param>
    /// <returns>List of blocks that can be freed</returns>
    public List<int> CleanupBeams(List<int> beamIds)
    {
        var freedBlocks = new List<int>();

        foreach (var beamId in beamIds)
        {
            var blocks = _shareManager.ReleaseSequence(beamId);
            freedBlocks.AddRange(blocks);
            _beamInfos.Remove(beamId);
        }

        return freedBlocks;
    }

    /// <summary>
    /// Get information about a beam.
    /// </summary>
    public BeamInfo? GetBeamInfo(int beamId)
    {
        return _beamInfos.TryGetValue(beamId, out var info) ? info : null;
    }

    private List<int> GetPrefixBlockIds(int baseSequenceId, int prefixLength)
    {
        // In a real implementation, this would query the block table
        // to get all blocks covering the prefix
        return new List<int>(); // Placeholder
    }
}

/// <summary>
/// Information about a beam in beam search.
/// </summary>
public class BeamInfo
{
    public int BeamIndex { get; set; }
    public int BaseSequenceId { get; set; }
    public int DivergencePoint { get; set; }
}
```

### SpeculativeDecodingSharing
```csharp
namespace MlFramework.Inference.PagedAttention.Sharing;

/// <summary>
/// Implements block sharing for speculative decoding.
/// Speculated tokens share blocks until verification.
/// </summary>
public class SpeculativeDecodingSharing
{
    private readonly BlockShareManager _shareManager;
    private readonly int _speculationLength;

    public SpeculativeDecodingSharing(
        BlockShareManager shareManager,
        int speculationLength = 4)
    {
        _shareManager = shareManager;
        _speculationLength = speculationLength;
    }

    /// <summary>
    /// Allocate speculative blocks for a sequence.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="baseSequenceLength">Length before speculation</param>
    /// <param name="speculativeBlockIds">List of speculative block IDs</param>
    public void AllocateSpeculativeBlocks(
        int sequenceId,
        int baseSequenceLength,
        List<int> speculativeBlockIds)
    {
        // Mark these blocks as speculative (not yet verified)
        // They will be shared between main sequence and speculator
        // This is a simplified implementation
    }

    /// <summary>
    /// Verify speculated tokens and update block sharing.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="verifiedTokens">Number of tokens that were verified</param>
    /// <returns>List of blocks to keep (verified) and free (rejected)</returns>
    public (List<int> keptBlocks, List<int> freedBlocks) VerifySpeculation(
        int sequenceId,
        int verifiedTokens)
    {
        var keptBlocks = new List<int>();
        var freedBlocks = new List<int>();

        // Keep blocks for verified tokens
        // Free blocks for rejected tokens
        // This is a simplified implementation

        return (keptBlocks, freedBlocks);
    }

    /// <summary>
    /// Reject all speculative blocks.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <returns>List of blocks to free</returns>
    public List<int> RejectAllSpeculation(int sequenceId)
    {
        var freedBlocks = new List<int>();

        // Free all speculative blocks
        // This is a simplified implementation

        return freedBlocks;
    }
}
```

## Requirements
1. **Reference Counting**: Accurate tracking of block references
2. **Beam Search**: Efficient sharing for beam decoding
3. **Speculative Decoding**: Support for speculative token generation
4. **Memory Efficiency**: Minimize memory usage through sharing
5. **Thread Safety**: Support concurrent access
6. **Divergence Handling**: Proper handling when sequences diverge

## Testing Requirements
1. Unit tests for block sharing reference counting
2. Unit tests for beam search sharing
3. Unit tests for speculative decoding sharing
4. Unit tests for divergence handling
5. Unit tests for cleanup operations
6. Concurrent access tests
7. Memory usage tests (verify sharing reduces memory)

## Estimated Time
45-60 minutes

## Dependencies
- spec_pagedattention_models.md
- spec_kvcache_block_manager.md

## Success Criteria
- Accurate reference counting
- Proper beam search sharing
- Correct speculative decoding handling
- Memory savings through sharing
- Thread-safe operations
- Clean resource cleanup
