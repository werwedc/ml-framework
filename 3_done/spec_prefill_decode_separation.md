# Spec: Prefill/Decode Phase Separation

## Overview
Implement phase-specific optimization strategies for PagedAttention, distinguishing between the prefill phase (processing prompt tokens) and decode phase (generating tokens one at a time). Each phase requires different optimization approaches.

## Target Directory
`src/MlFramework/Inference/PagedAttention/Phases/`

## Classes to Implement

### PrefillOrchestrator
```csharp
using MlFramework.Tensor;

namespace MlFramework.Inference.PagedAttention.Phases;

/// <summary>
/// Orchestrates the prefill phase where prompt tokens are processed in parallel.
/// This phase is compute-bound and benefits from high parallelism.
/// </summary>
public class PrefillOrchestrator
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly BlockTable _blockTable;
    private readonly int _blockSize;

    public PrefillOrchestrator(
        KVCacheBlockManager blockManager,
        BlockTable blockTable)
    {
        _blockManager = blockManager;
        _blockTable = blockTable;
        _blockSize = blockManager.BlockSize;
    }

    /// <summary>
    /// Pre-allocate blocks for the entire prompt.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="promptLength">Length of the prompt</param>
    /// <returns>List of allocated block IDs</returns>
    public List<int> PrefillAllocateBlocks(int sequenceId, int promptLength)
    {
        // Calculate number of blocks needed
        int blocksNeeded = (promptLength + _blockSize - 1) / _blockSize;

        // Allocate all blocks atomically
        var blockIds = _blockManager.AllocateBlocks(sequenceId, blocksNeeded);

        if (blockIds.Count < blocksNeeded)
        {
            throw new InvalidOperationException(
                $"Insufficient blocks for prefill: need {blocksNeeded}, got {blockIds.Count}");
        }

        // Append all blocks to the block table
        for (int i = 0; i < blockIds.Count; i++)
        {
            _blockTable.AppendBlock(sequenceId, blockIds[i]);
        }

        return blockIds;
    }

    /// <summary>
    /// Compute attention during prefill phase.
    /// </summary>
    /// <param name="query">Query tensors for all prompt tokens</param>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="startToken">Starting token index</param>
    /// <param name="endToken">Ending token index</param>
    /// <returns>Attention output for all tokens</returns>
    public Tensor ComputePrefillAttention(
        Tensor query,
        int sequenceId,
        int startToken,
        int endToken,
        IPagedAttentionKernel kernel)
    {
        // Gather KV cache for the entire range
        var blockIds = _blockTable.GetSequenceBlocks(sequenceId);

        // Prefill uses parallel computation on all tokens
        return kernel.ComputePagedAttention(
            query,
            GatherPrefillKeys(blockIds, startToken, endToken),
            GatherPrefillValues(blockIds, startToken, endToken),
            AttentionPhase.Prefill
        );
    }

    /// <summary>
    /// Batch prefill for multiple sequences.
    /// </summary>
    public Dictionary<int, Tensor> BatchPrefill(
        Dictionary<int, Tensor> queries,
        Dictionary<int, (int start, int end)> ranges,
        IPagedAttentionKernel kernel)
    {
        var results = new Dictionary<int, Tensor>();

        // Process each sequence independently (can be parallelized)
        foreach (var kvp in queries)
        {
            var sequenceId = kvp.Key;
            var query = kvp.Value;
            var (start, end) = ranges[sequenceId];

            results[sequenceId] = ComputePrefillAttention(
                query,
                sequenceId,
                start,
                end,
                kernel
            );
        }

        return results;
    }

    private Tensor GatherPrefillKeys(List<int> blockIds, int start, int end)
    {
        // Implementation similar to GatherKVFromBlocks but optimized for parallel access
        // This is a placeholder - actual implementation should be optimized
        throw new NotImplementedException();
    }

    private Tensor GatherPrefillValues(List<int> blockIds, int start, int end)
    {
        // Implementation similar to GatherKVFromBlocks but optimized for parallel access
        // This is a placeholder - actual implementation should be optimized
        throw new NotImplementedException();
    }
}
```

### DecodeOrchestrator
```csharp
namespace MlFramework.Inference.PagedAttention.Phases;

/// <summary>
/// Orchestrates the decode phase where tokens are generated one at a time.
/// This phase is memory-bound and benefits from optimized memory access patterns.
/// </summary>
public class DecodeOrchestrator
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly BlockTable _blockTable;
    private readonly int _blockSize;

    public DecodeOrchestrator(
        KVCacheBlockManager blockManager,
        BlockTable blockTable)
    {
        _blockManager = blockManager;
        _blockTable = blockTable;
        _blockSize = blockManager.BlockSize;
    }

    /// <summary>
    /// Allocate and append a block for a new token during decode.
    /// </summary>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <returns>Block ID for the new token</returns>
    public int AllocateDecodeBlock(int sequenceId)
    {
        // Check if current last block has space
        var sequenceBlocks = _blockTable.GetSequenceBlocks(sequenceId);
        if (sequenceBlocks.Count > 0)
        {
            var lastBlockId = sequenceBlocks.Last();
            var lastBlock = _blockManager.GetBlock(lastBlockId);

            if (lastBlock != null && lastBlock.TokenCount < _blockSize)
            {
                // Existing block has space
                return lastBlockId;
            }
        }

        // Need to allocate a new block
        var newBlockId = _blockTable.AllocateAndAppendBlock(sequenceId);
        if (newBlockId == -1)
        {
            throw new InvalidOperationException(
                $"Failed to allocate block for sequence {sequenceId}");
        }

        return newBlockId;
    }

    /// <summary>
    /// Compute attention for a single token during decode phase.
    /// </summary>
    /// <param name="query">Query tensor for the new token [1, 1, numHeads, headDim]</param>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="currentLength">Current length of the sequence (before this token)</param>
    /// <returns>Attention output for the new token</returns>
    public Tensor ComputeDecodeAttention(
        Tensor query,
        int sequenceId,
        int currentLength,
        IPagedAttentionKernel kernel)
    {
        // Gather all previous KV cache
        var blockIds = _blockTable.GetSequenceBlocks(sequenceId);
        var newLength = currentLength + 1;

        return kernel.ComputePagedAttention(
            query,
            GatherDecodeKeys(blockIds, newLength),
            GatherDecodeValues(blockIds, newLength),
            AttentionPhase.Decode
        );
    }

    /// <summary>
    /// Batch decode for multiple sequences (continuous batching).
    /// </summary>
    public Dictionary<int, Tensor> BatchDecode(
        Dictionary<int, Tensor> queries,
        Dictionary<int, int> lengths,
        IPagedAttentionKernel kernel)
    {
        var results = new Dictionary<int, Tensor>();

        // Process each sequence
        // Note: For continuous batching, we could interleave processing
        foreach (var kvp in queries)
        {
            var sequenceId = kvp.Key;
            var query = kvp.Value;
            var length = lengths[sequenceId];

            results[sequenceId] = ComputeDecodeAttention(
                query,
                sequenceId,
                length,
                kernel
            );
        }

        return results;
    }

    private Tensor GatherDecodeKeys(List<int> blockIds, int length)
    {
        // Implementation optimized for sequential access
        // This is a placeholder - actual implementation should be optimized
        throw new NotImplementedException();
    }

    private Tensor GatherDecodeValues(List<int> blockIds, int length)
    {
        // Implementation optimized for sequential access
        // This is a placeholder - actual implementation should be optimized
        throw new NotImplementedException();
    }
}
```

### PhaseManager
```csharp
namespace MlFramework.Inference.PagedAttention.Phases;

/// <summary>
/// Manages transitions between prefill and decode phases.
/// </summary>
public class PhaseManager
{
    private readonly PrefillOrchestrator _prefillOrchestrator;
    private readonly DecodeOrchestrator _decodeOrchestrator;
    private readonly Dictionary<int, SequencePhase> _sequencePhases;

    public PhaseManager(
        PrefillOrchestrator prefillOrchestrator,
        DecodeOrchestrator decodeOrchestrator)
    {
        _prefillOrchestrator = prefillOrchestrator;
        _decodeOrchestrator = decodeOrchestrator;
        _sequencePhases = new Dictionary<int, SequencePhase>();
    }

    /// <summary>
    /// Get the current phase for a sequence.
    /// </summary>
    public SequencePhase GetPhase(int sequenceId)
    {
        return _sequencePhases.TryGetValue(sequenceId, out var phase)
            ? phase
            : SequencePhase.Prefill;
    }

    /// <summary>
    /// Set the phase for a sequence.
    /// </summary>
    public void SetPhase(int sequenceId, SequencePhase phase)
    {
        _sequencePhases[sequenceId] = phase;
    }

    /// <summary>
    /// Transition a sequence from prefill to decode.
    /// </summary>
    public void TransitionToDecode(int sequenceId)
    {
        if (_sequencePhases.ContainsKey(sequenceId))
        {
            _sequencePhases[sequenceId] = SequencePhase.Decode;
        }
    }

    /// <summary>
    /// Check if a sequence is in prefill phase.
    /// </summary>
    public bool IsPrefilling(int sequenceId)
    {
        return GetPhase(sequenceId) == SequencePhase.Prefill;
    }

    /// <summary>
    /// Check if a sequence is in decode phase.
    /// </summary>
    public bool IsDecoding(int sequenceId)
    {
        return GetPhase(sequenceId) == SequencePhase.Decode;
    }

    /// <summary>
    /// Remove a sequence from phase tracking.
    /// </summary>
    public void RemoveSequence(int sequenceId)
    {
        _sequencePhases.Remove(sequenceId);
    }
}

/// <summary>
/// Phase of a sequence in the inference pipeline.
/// </summary>
public enum SequencePhase
{
    /// <summary>
    /// Currently processing prompt tokens.
    /// </summary>
    Prefill,

    /// <summary>
    /// Currently generating tokens one at a time.
    /// </summary>
    Decode
}
```

## Requirements
1. **Phase Separation**: Clear distinction between prefill and compute strategies
2. **Optimization**: Each phase should use appropriate optimization patterns
3. **Batching**: Support for batched prefill and decode operations
4. **State Management**: Track phase transitions for each sequence
5. **Efficiency**: Minimize overhead during phase transitions

## Testing Requirements
1. Unit tests for prefill block allocation
2. Unit tests for decode block allocation
3. Unit tests for phase transitions
4. Unit tests for batch prefill
5. Unit tests for batch decode
6. Integration tests with PagedAttentionLayer

## Estimated Time
45-60 minutes

## Dependencies
- spec_kvcache_block_manager.md
- spec_block_table.md
- spec_attention_kernel_interface.md

## Success Criteria
- Correct phase management
- Efficient prefill computation
- Optimized decode computation
- Smooth phase transitions
- Proper batching support
