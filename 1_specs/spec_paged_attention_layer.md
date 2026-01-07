# Spec: Paged Attention Layer

## Overview
Implement the PagedAttentionLayer class that integrates paged KV caching with transformer attention computation. This layer queries the block table to access cached KV tensors stored in non-contiguous memory blocks.

## Target Directory
`src/MlFramework/Inference/PagedAttention/Layers/`

## Class to Implement

### PagedAttentionLayer
```csharp
using MlFramework.Inference.PagedAttention.Models;
using MlFramework.Tensor;
using MlFramework.NeuralNetwork.Layers;

namespace MlFramework.Inference.PagedAttention.Layers;

/// <summary>
/// Attention layer that uses paged KV cache for efficient memory management.
/// Supports both prefill and decode phases with optimized memory access patterns.
/// </summary>
public class PagedAttentionLayer : IAttentionLayer
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly BlockTable _blockTable;
    private readonly IPagedAttentionKernel _attentionKernel;
    private readonly int _layerIndex;
    private readonly int _numAttentionHeads;
    private readonly int _headDim;
    private readonly int _numLayers;
    private readonly int _blockSize;

    // Query, Key, Value projections
    private readonly LinearLayer _queryProj;
    private readonly LinearLayer _keyProj;
    private readonly LinearLayer _valueProj;
    private readonly LinearLayer _outputProj;

    public PagedAttentionLayer(
        int embeddingDim,
        int numAttentionHeads,
        int headDim,
        int layerIndex,
        int numLayers,
        int blockSize,
        KVCacheBlockManager blockManager,
        BlockTable blockTable,
        IPagedAttentionKernel attentionKernel,
        Device device)
    {
        _numAttentionHeads = numAttentionHeads;
        _headDim = headDim;
        _layerIndex = layerIndex;
        _numLayers = numLayers;
        _blockSize = blockSize;
        _blockManager = blockManager;
        _blockTable = blockTable;
        _attentionKernel = attentionKernel;

        // Initialize projection layers
        _queryProj = new LinearLayer(embeddingDim, numAttentionHeads * headDim, device);
        _keyProj = new LinearLayer(embeddingDim, numAttentionHeads * headDim, device);
        _valueProj = new LinearLayer(embeddingDim, numAttentionHeads * headDim, device);
        _outputProj = new LinearLayer(numAttentionHeads * headDim, embeddingDim, device);
    }

    /// <summary>
    /// Compute attention output using paged KV cache.
    /// </summary>
    /// <param name="hiddenStates">Input hidden states [batchSize, seqLen, hiddenSize]</param>
    /// <param name="sequenceId">ID of the sequence</param>
    /// <param name="startToken">Starting token index in the sequence</param>
    /// <param name="endToken">Ending token index (exclusive)</param>
    /// <param name="phase">Attention phase (Prefill or Decode)</param>
    /// <returns>Attention output tensor</returns>
    public Tensor ComputeAttention(
        Tensor hiddenStates,
        int sequenceId,
        int startToken,
        int endToken,
        AttentionPhase phase = AttentionPhase.Decode)
    {
        // Project input to Q, K, V
        var query = _queryProj.Forward(hiddenStates); // [batch, seqLen, numHeads * headDim]
        var key = _keyProj.Forward(hiddenStates);     // [batch, seqLen, numHeads * headDim]
        var value = _valueProj.Forward(hiddenStates); // [batch, seqLen, numHeads * headDim]

        // Reshape for multi-head attention
        query = query.Reshape(
            query.Shape[0],
            query.Shape[1],
            _numAttentionHeads,
            _headDim
        ); // [batch, seqLen, numHeads, headDim]

        key = key.Reshape(
            key.Shape[0],
            key.Shape[1],
            _numAttentionHeads,
            _headDim
        ); // [batch, seqLen, numHeads, headDim]

        value = value.Reshape(
            value.Shape[0],
            value.Shape[1],
            _numAttentionHeads,
            _headDim
        ); // [batch, seqLen, numHeads, headDim]

        // Store new K, V in the KV cache
        UpdateKVCache(sequenceId, startToken, endToken, key, value);

        // Compute attention using cached KV
        var attentionOutput = ComputePagedAttention(
            sequenceId,
            query,
            startToken,
            endToken,
            phase
        );

        // Project output
        attentionOutput = attentionOutput.Reshape(
            attentionOutput.Shape[0],
            attentionOutput.Shape[1],
            _numAttentionHeads * _headDim
        );

        var output = _outputProj.Forward(attentionOutput);
        return output;
    }

    /// <summary>
    /// Update the KV cache with new key and value tensors.
    /// </summary>
    private void UpdateKVCache(
        int sequenceId,
        int startToken,
        int endToken,
        Tensor key,
        Tensor value)
    {
        var sequenceLength = endToken - startToken;

        // Calculate which blocks need to be updated
        for (int tokenIdx = startToken; tokenIdx < endToken; tokenIdx++)
        {
            var blockId = _blockTable.GetBlock(sequenceId, tokenIdx);
            if (blockId == -1)
            {
                // Allocate new block if needed
                blockId = _blockTable.AllocateAndAppendBlock(sequenceId);
                if (blockId == -1)
                {
                    throw new InvalidOperationException(
                        $"Failed to allocate block for sequence {sequenceId}");
                }
            }

            var block = _blockManager.GetBlock(blockId);
            if (block == null) continue;

            // Calculate position within the block
            var positionInBlock = tokenIdx - block.StartTokenIndex;
            var relativeIdx = tokenIdx - startToken;

            // Copy key and value tensors to the block
            // Access specific layers for this attention layer
            CopyTensorToBlock(
                block,
                key,
                relativeIdx,
                positionInBlock,
                isKey: true
            );

            CopyTensorToBlock(
                block,
                value,
                relativeIdx,
                positionInBlock,
                isKey: false
            );

            // Update token count
            block.TokenCount = Math.Max(block.TokenCount, positionInBlock + 1);
        }
    }

    /// <summary>
    /// Copy tensor data to a specific position in a block.
    /// </summary>
    private void CopyTensorToBlock(
        MemoryBlock block,
        Tensor source,
        int sourceIdx,
        int targetIdx,
        bool isKey)
    {
        var targetTensor = isKey ? block.KeyTensor : block.ValueTensor;

        // Slice for this layer: [numLayers, numHeads, blockSize, headDim]
        // We want to copy to [layerIndex, :, targetIdx, :]
        var layerSlice = targetTensor.Slice(
            _layerIndex,
            _layerIndex + 1,
            0,
            _numAttentionHeads,
            targetIdx,
            targetIdx + 1,
            0,
            _headDim
        ); // Shape: [1, numHeads, 1, headDim]

        // Source is [batch, seqLen, numHeads, headDim]
        // Extract for this token: [:, sourceIdx, :, :] -> [1, numHeads, headDim]
        var sourceSlice = source.Slice(
            0,
            source.Shape[0],
            sourceIdx,
            sourceIdx + 1,
            0,
            _numAttentionHeads,
            0,
            _headDim
        ); // Shape: [batch, 1, numHeads, headDim]

        // Expand layerSlice to match source dimensions and copy
        // For simplicity, batch_size is assumed to be 1 for inference
        layerSlice.CopyFrom(sourceSlice);
    }

    /// <summary>
    /// Compute attention using paged KV cache.
    /// </summary>
    private Tensor ComputePagedAttention(
        int sequenceId,
        Tensor query,
        int startToken,
        int endToken,
        AttentionPhase phase)
    {
        // Get all blocks for the sequence
        var blockIds = _blockTable.GetSequenceBlocks(sequenceId);
        if (blockIds.Count == 0)
        {
            throw new InvalidOperationException(
                $"No blocks found for sequence {sequenceId}");
        }

        // Gather key and value tensors from all blocks
        var (cachedKeys, cachedValues) = GatherKVFromBlocks(
            blockIds,
            startToken,
            endToken
        );

        // Use the custom attention kernel
        return _attentionKernel.ComputePagedAttention(
            query,
            cachedKeys,
            cachedValues,
            phase
        );
    }

    /// <summary>
    /// Gather KV tensors from multiple blocks for a token range.
    /// </summary>
    private (Tensor keys, Tensor values) GatherKVFromBlocks(
        List<int> blockIds,
        int startToken,
        int endToken)
    {
        // Calculate total tokens needed
        var totalTokens = endToken - startToken;

        // Pre-allocate tensors for gathered KV
        var keys = Tensor.Zeros(
            new[] { 1, totalTokens, _numAttentionHeads, _headDim },
            query.Device
        );
        var values = Tensor.Zeros(
            new[] { 1, totalTokens, _numAttentionHeads, _headDim },
            query.Device
        );

        int targetIdx = 0;

        // Iterate through blocks and copy relevant data
        foreach (var blockId in blockIds)
        {
            var block = _blockManager.GetBlock(blockId);
            if (block == null) continue;

            var blockStart = block.StartTokenIndex;
            var blockEnd = blockStart + _blockSize;

            // Find overlap with requested range
            var overlapStart = Math.Max(blockStart, startToken);
            var overlapEnd = Math.Min(blockEnd, endToken);

            if (overlapStart >= overlapEnd) continue;

            // Copy from block to output tensors
            for (int tokenIdx = overlapStart; tokenIdx < overlapEnd; tokenIdx++)
            {
                var positionInBlock = tokenIdx - blockStart;
                var sourceIdx = tokenIdx - startToken;

                // Extract from block tensors
                // Block: [numLayers, numHeads, blockSize, headDim]
                var keySlice = block.KeyTensor.Slice(
                    _layerIndex,
                    _layerIndex + 1,
                    0,
                    _numAttentionHeads,
                    positionInBlock,
                    positionInBlock + 1,
                    0,
                    _headDim
                ); // [1, numHeads, 1, headDim]

                var valueSlice = block.ValueTensor.Slice(
                    _layerIndex,
                    _layerIndex + 1,
                    0,
                    _numAttentionHeads,
                    positionInBlock,
                    positionInBlock + 1,
                    0,
                    _headDim
                ); // [1, numHeads, 1, headDim]

                // Copy to output
                keys.Slice(0, 1, sourceIdx, sourceIdx + 1, 0, _numAttentionHeads, 0, _headDim)
                   .CopyFrom(keySlice);

                values.Slice(0, 1, sourceIdx, sourceIdx + 1, 0, _numAttentionHeads, 0, _headDim)
                     .CopyFrom(valueSlice);
            }
        }

        return (keys, values);
    }

    public void Dispose()
    {
        _queryProj?.Dispose();
        _keyProj?.Dispose();
        _valueProj?.Dispose();
        _outputProj?.Dispose();
    }
}

/// <summary>
/// Attention execution phase.
/// </summary>
public enum AttentionPhase
{
    /// <summary>
    /// Prefill phase: process multiple tokens at once (parallel).
    /// </summary>
    Prefill,

    /// <summary>
    /// Decode phase: process one token at a time (sequential).
    /// </summary>
    Decode
}

/// <summary>
/// Interface for paged attention computation kernels.
/// </summary>
public interface IPagedAttentionKernel
{
    /// <summary>
    /// Compute attention with paged KV cache.
    /// </summary>
    Tensor ComputePagedAttention(
        Tensor query,
        Tensor cachedKeys,
        Tensor cachedValues,
        AttentionPhase phase
    );
}
```

## Requirements
1. **Integration**: Work seamlessly with existing transformer architecture
2. **Cache Update**: Correctly update KV cache for each new token
3. **Memory Safety**: Properly handle block allocation and tensor copying
4. **Phase Awareness**: Different optimization strategies for prefill vs decode
5. **Error Handling**: Graceful handling of allocation failures

## Testing Requirements
1. Unit tests for KV cache updates
2. Unit tests for attention computation with paged cache
3. Unit tests for tensor gathering from multiple blocks
4. Functional tests comparing output to contiguous baseline
5. Integration tests with actual transformer models

## Estimated Time
60 minutes

## Dependencies
- spec_pagedattention_models.md
- spec_kvcache_block_manager.md
- spec_block_table.md
- spec_attention_kernel_interface.md

## Success Criteria
- Correct attention computation using paged KV cache
- Proper KV cache updates for new tokens
- Efficient tensor gathering from multiple blocks
- Compatible with existing transformer layers
