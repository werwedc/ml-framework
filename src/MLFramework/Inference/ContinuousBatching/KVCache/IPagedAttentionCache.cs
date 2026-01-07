namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Interface for PagedAttention cache management.
/// Provides block-based allocation and deallocation for KV cache.
/// </summary>
public interface IPagedAttentionCache
{
    /// <summary>
    /// Allocates a new cache block (or multiple contiguous blocks).
    /// </summary>
    /// <param name="blockCount">Number of blocks to allocate (default: 1).</param>
    /// <returns>Index of the first allocated block.</returns>
    int AllocateBlock(int blockCount = 1);

    /// <summary>
    /// Releases one or more cache blocks.
    /// </summary>
    /// <param name="blockIndex">Index of the first block to release.</param>
    /// <param name="blockCount">Number of blocks to release (default: 1).</param>
    void ReleaseBlock(int blockIndex, int blockCount = 1);

    /// <summary>
    /// Gets the number of free (available) blocks.
    /// </summary>
    /// <returns>Number of free blocks.</returns>
    int GetFreeBlockCount();

    /// <summary>
    /// Gets the total number of blocks in the cache.
    /// </summary>
    /// <returns>Total block count.</returns>
    int GetTotalBlockCount();

    /// <summary>
    /// Gets block data for model execution.
    /// </summary>
    /// <param name="blockIndex">Index of the block to retrieve.</param>
    /// <returns>Cache block data containing key/value tensors.</returns>
    CacheBlockData GetBlockData(int blockIndex);

    /// <summary>
    /// Stores cache block data after computation.
    /// </summary>
    /// <param name="blockIndex">Index of the block to store data in.</param>
    /// <param name="data">Cache block data to store.</param>
    void StoreBlockData(int blockIndex, CacheBlockData data);
}
