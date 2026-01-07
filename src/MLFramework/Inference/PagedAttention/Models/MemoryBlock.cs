using RitterFramework.Core.Tensor;

namespace MlFramework.Inference.PagedAttention.Models;

/// <summary>
/// Represents a fixed-size block of GPU memory for storing KV cache tokens.
/// </summary>
public class MemoryBlock
{
    /// <summary>
    /// Unique identifier for this block.
    /// </summary>
    public int BlockId { get; }

    /// <summary>
    /// The sequence ID currently using this block.
    /// Null if block is free.
    /// </summary>
    public int? SequenceId { get; set; }

    /// <summary>
    /// Starting token index within the sequence.
    /// </summary>
    public int StartTokenIndex { get; set; }

    /// <summary>
    /// Number of tokens stored in this block (up to BlockSize).
    /// </summary>
    public int TokenCount { get; set; }

    /// <summary>
    /// Flag indicating if this block is currently allocated.
    /// </summary>
    public bool IsAllocated => SequenceId.HasValue;

    /// <summary>
    /// Reference to the actual tensor storing keys/values.
    /// </summary>
    public Tensor? KeyTensor { get; set; }
    public Tensor? ValueTensor { get; set; }

    public MemoryBlock(int blockId, int blockSize)
    {
        BlockId = blockId;
        TokenCount = 0;
    }

    /// <summary>
    /// Resets the block to free state.
    /// </summary>
    public void Reset()
    {
        SequenceId = null;
        StartTokenIndex = 0;
        TokenCount = 0;
    }
}
