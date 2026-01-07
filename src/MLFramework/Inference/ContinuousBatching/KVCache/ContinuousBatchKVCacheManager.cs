namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Manages KV cache allocation and deallocation for continuous batching.
/// Integrates with PagedAttention cache system for efficient memory management.
/// </summary>
public class ContinuousBatchKVCacheManager : IKVCacheManager
{
    private readonly IPagedAttentionCache _pagedCache;
    private readonly KVCacheConfiguration _config;
    private readonly Dictionary<RequestId, CacheAllocation> _allocations;
    private readonly object _lock;

    /// <summary>
    /// Creates a new KV cache manager.
    /// </summary>
    /// <param name="pagedCache">The PagedAttention cache implementation.</param>
    /// <param name="config">Configuration for cache behavior.</param>
    public ContinuousBatchKVCacheManager(
        IPagedAttentionCache pagedCache,
        KVCacheConfiguration config)
    {
        _pagedCache = pagedCache ?? throw new ArgumentNullException(nameof(pagedCache));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _allocations = new Dictionary<RequestId, CacheAllocation>();
        _lock = new object();
    }

    /// <summary>
    /// Allocates KV cache for a request.
    /// </summary>
    /// <param name="requestId">The request ID to allocate cache for.</param>
    /// <param name="maxTokens">Maximum number of tokens for the request.</param>
    /// <returns>The allocated cache size in bytes.</returns>
    public long AllocateCache(RequestId requestId, int maxTokens)
    {
        if (maxTokens < 0)
            throw new ArgumentException("Max tokens cannot be negative", nameof(maxTokens));

        lock (_lock)
        {
            // Check if allocation already exists
            if (_allocations.ContainsKey(requestId))
                throw new InvalidOperationException($"Cache already allocated for request {requestId}");

            // Calculate required pages
            int requiredPages = CalculateRequiredPages(maxTokens);
            requiredPages = Math.Min(requiredPages, _config.MaxPagesPerRequest);

            // Allocate cache blocks
            int allocatedBlocks = 0;
            var pages = new List<CachePage>();

            try
            {
                for (int i = 0; i < requiredPages; i++)
                {
                    int blockIndex = _pagedCache.AllocateBlock();
                    var page = new CachePage(
                        i,
                        i * _config.PageSizeTokens,
                        _config.PageSizeTokens,
                        blockIndex
                    );
                    pages.Add(page);
                    allocatedBlocks++;
                }

                // Create allocation record
                long totalBytes = allocatedBlocks * _config.PageSizeBytes;
                var allocation = new CacheAllocation(
                    requestId,
                    pages,
                    allocatedBlocks * _config.PageSizeTokens,
                    totalBytes,
                    DateTime.UtcNow
                );

                _allocations[requestId] = allocation;
                return totalBytes;
            }
            catch (Exception)
            {
                // Rollback on failure
                foreach (var page in pages)
                {
                    _pagedCache.ReleaseBlock(page.BlockIndex);
                }
                throw;
            }
        }
    }

    /// <summary>
    /// Releases KV cache for a request.
    /// </summary>
    /// <param name="requestId">The request ID to release cache for.</param>
    public void ReleaseCache(RequestId requestId)
    {
        lock (_lock)
        {
            if (!_allocations.TryGetValue(requestId, out var allocation))
                return;

            // Release all blocks
            foreach (var page in allocation.Pages)
            {
                _pagedCache.ReleaseBlock(page.BlockIndex);
            }

            // Remove allocation record
            _allocations.Remove(requestId);
        }
    }

    /// <summary>
    /// Gets the current KV cache usage in bytes.
    /// </summary>
    /// <returns>Current usage in bytes.</returns>
    public long GetCurrentUsageBytes()
    {
        lock (_lock)
        {
            return _allocations.Values.Sum(a => a.TotalBytesAllocated);
        }
    }

    /// <summary>
    /// Attempts to extend the cache allocation for a request.
    /// </summary>
    /// <param name="requestId">The request ID to extend cache for.</param>
    /// <param name="additionalTokens">Additional tokens needed.</param>
    /// <returns>True if extension succeeded, false otherwise.</returns>
    public bool TryExtendCache(RequestId requestId, int additionalTokens)
    {
        if (additionalTokens <= 0)
            return false;

        lock (_lock)
        {
            if (!_allocations.TryGetValue(requestId, out var allocation))
                return false;

            // Check if already at max
            if (allocation.PageCount >= _config.MaxPagesPerRequest)
                return false;

            // Calculate additional pages needed
            int additionalPages = (additionalTokens + _config.PageSizeTokens - 1)
                                / _config.PageSizeTokens;

            // Try to allocate
            var newPages = new List<CachePage>();
            try
            {
                for (int i = 0; i < additionalPages; i++)
                {
                    if (allocation.PageCount + newPages.Count >= _config.MaxPagesPerRequest)
                        break;

                    int blockIndex = _pagedCache.AllocateBlock();
                    var page = new CachePage(
                        allocation.PageCount + newPages.Count,
                        (allocation.PageCount + newPages.Count) * _config.PageSizeTokens,
                        _config.PageSizeTokens,
                        blockIndex
                    );
                    newPages.Add(page);
                }

                if (newPages.Count == 0)
                    return false;

                // Update allocation
                var updatedPages = new List<CachePage>(allocation.Pages);
                updatedPages.AddRange(newPages);

                long additionalBytes = newPages.Count * _config.PageSizeBytes;
                var updatedAllocation = allocation with
                {
                    Pages = updatedPages,
                    TotalTokensAllocated = allocation.TotalTokensAllocated + newPages.Count * _config.PageSizeTokens,
                    TotalBytesAllocated = allocation.TotalBytesAllocated + additionalBytes
                };

                _allocations[requestId] = updatedAllocation;
                return true;
            }
            catch
            {
                // Rollback new pages
                foreach (var page in newPages)
                {
                    _pagedCache.ReleaseBlock(page.BlockIndex);
                }
                return false;
            }
        }
    }

    /// <summary>
    /// Gets allocation details for a specific request.
    /// </summary>
    /// <param name="requestId">The request ID to get allocation for.</param>
    /// <returns>Allocation details, or null if not found.</returns>
    public CacheAllocation? GetAllocation(RequestId requestId)
    {
        lock (_lock)
        {
            return _allocations.TryGetValue(requestId, out var allocation)
                ? allocation
                : null;
        }
    }

    /// <summary>
    /// Compacts fragmented cache to improve utilization.
    /// </summary>
    public void CompactCache()
    {
        lock (_lock)
        {
            if (!_config.EnableCompaction)
                return;

            // Check fragmentation
            double utilization = CalculateFragmentationUtilization();
            if (utilization >= _config.TargetUtilization)
                return;

            // Note: Actual compaction logic depends on PagedAttention implementation
            // This is a placeholder for future optimization
            // In practice, might involve:
            // 1. Defragmenting blocks
            // 2. Merging small allocations
            // 3. Releasing unused pages
        }
    }

    /// <summary>
    /// Calculates the current cache utilization.
    /// </summary>
    /// <returns>Utilization ratio (0-1).</returns>
    private double CalculateFragmentationUtilization()
    {
        int totalBlocks = _pagedCache.GetTotalBlockCount();
        int freeBlocks = _pagedCache.GetFreeBlockCount();

        if (totalBlocks == 0)
            return 0.0;

        return (double)(totalBlocks - freeBlocks) / totalBlocks;
    }

    /// <summary>
    /// Calculates the required number of pages for a given token count.
    /// </summary>
    /// <param name="maxTokens">Maximum tokens needed.</param>
    /// <returns>Required number of pages.</returns>
    private int CalculateRequiredPages(int maxTokens)
    {
        int initialPages = _config.InitialPagesPerRequest;
        int requiredPages = (maxTokens + _config.PageSizeTokens - 1) / _config.PageSizeTokens;

        // Start with initial pages, allow growth
        return Math.Max(initialPages, requiredPages);
    }

    /// <summary>
    /// Gets the number of active allocations.
    /// </summary>
    public int AllocationCount
    {
        get
        {
            lock (_lock)
            {
                return _allocations.Count;
            }
        }
    }
}
