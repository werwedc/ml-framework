# Spec: KV Cache Integration for Continuous Batching

## Overview
Implement the integration layer between the continuous batch scheduler and the PagedAttention KV cache system. This integration handles dynamic memory allocation, deallocation, and cache management for requests entering and leaving batches.

## Class: ContinuousBatchKVCacheManager
```csharp
public class ContinuousBatchKVCacheManager : IKVCacheManager
{
    private readonly IPagedAttentionCache _pagedCache;
    private readonly KVCacheConfiguration _config;
    private readonly Dictionary<RequestId, CacheAllocation> _allocations;
    private readonly object _lock;

    public ContinuousBatchKVCacheManager(
        IPagedAttentionCache pagedCache,
        KVCacheConfiguration config)
    {
        _pagedCache = pagedCache;
        _config = config;
        _allocations = new Dictionary<RequestId, CacheAllocation>();
        _lock = new object();
    }

    // Allocate cache for a request
    public long AllocateCache(RequestId requestId, int maxTokens);

    // Release cache for a request
    public void ReleaseCache(RequestId requestId);

    // Get current total usage
    public long GetCurrentUsageBytes();

    // Extend cache for a request (if needed)
    public bool TryExtendCache(RequestId requestId, int additionalTokens);

    // Get allocation details
    public CacheAllocation? GetAllocation(RequestId requestId);

    // Compact fragmented cache (optional optimization)
    public void CompactCache();
}
```

---

## Class: KVCacheConfiguration
```csharp
public record class KVCacheConfiguration(
    int PageSizeTokens,               // Tokens per cache page
    int InitialPagesPerRequest,       // Initial pages to allocate
    int MaxPagesPerRequest,          // Maximum pages per request
    int CacheBlockSizeBytes,         // Size of each cache block in bytes
    double TargetUtilization,        // Target cache utilization (0-1)
    bool EnableCompaction            // Enable automatic compaction
)
{
    public static readonly KVCacheConfiguration Default = new(
        PageSizeTokens: 16,
        InitialPagesPerRequest: 16,     // 256 tokens initially
        MaxPagesPerRequest: 256,        // 4096 tokens max
        CacheBlockSizeBytes: 1024,     // 1KB per block (adjust based on model)
        TargetUtilization: 0.85,
        EnableCompaction: true
    );

    public long PageSizeBytes => PageSizeTokens * CacheBlockSizeBytes;
}
```

**Purpose**: Configure KV cache behavior.

---

## Class: CacheAllocation
```csharp
public record class CacheAllocation(
    RequestId RequestId,
    List<CachePage> Pages,
    int TotalTokensAllocated,
    long TotalBytesAllocated,
    DateTime AllocatedTime
)
{
    public int PageCount => Pages.Count;
    public TimeSpan Age => DateTime.UtcNow - AllocatedTime;
}
```

---

## Class: CachePage
```csharp
public record class CachePage(
    int PageIndex,
    int StartToken,
    int TokenCount,
    int BlockIndex
)
{
    public int EndToken => StartToken + TokenCount;
}
```

**Purpose**: Track individual cache pages for a request.

---

## Interface: IPagedAttentionCache
```csharp
public interface IPagedAttentionCache
{
    // Allocate a new cache block
    int AllocateBlock(int blockCount = 1);

    // Release cache blocks
    void ReleaseBlock(int blockIndex, int blockCount = 1);

    // Get number of free blocks
    int GetFreeBlockCount();

    // Get total blocks
    int GetTotalBlockCount();

    // Get block data for execution
    CacheBlockData GetBlockData(int blockIndex);

    // Store cache block data after computation
    void StoreBlockData(int blockIndex, CacheBlockData data);
}
```

---

## Class: CacheBlockData
```csharp
public record class CacheBlockData(
    int BlockIndex,
    float[] KeyCache,
    float[] ValueCache,
    int TokenCount
)
```

---

## Interface: IKVCacheManager
```csharp
public interface IKVCacheManager
{
    long AllocateCache(RequestId requestId, int maxTokens);
    void ReleaseCache(RequestId requestId);
    long GetCurrentUsageBytes();
}
```

---

## Implementation Details

### AllocateCache
```csharp
public long AllocateCache(RequestId requestId, int maxTokens)
{
    lock (_lock)
    {
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
```

**Requirements**:
- Calculate required pages
- Allocate from PagedAttention cache
- Handle allocation failures with rollback
- Return total bytes allocated
- Thread-safe

---

### CalculateRequiredPages
```csharp
private int CalculateRequiredPages(int maxTokens)
{
    int initialPages = _config.InitialPagesPerRequest;
    int requiredPages = (maxTokens + _config.PageSizeTokens - 1) / _config.PageSizeTokens;

    // Start with initial pages, allow growth
    return Math.Max(initialPages, requiredPages);
}
```

**Requirements**:
- Account for page size
- Use initial pages as minimum
- Round up to full pages

---

### ReleaseCache
```csharp
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
```

**Requirements**:
- Release all allocated blocks
- Handle missing allocations
- Thread-safe

---

### GetCurrentUsageBytes
```csharp
public long GetCurrentUsageBytes()
{
    lock (_lock)
    {
        return _allocations.Values.Sum(a => a.TotalBytesAllocated);
    }
}
```

**Requirements**:
- Sum all allocations
- Thread-safe

---

### TryExtendCache
```csharp
public bool TryExtendCache(RequestId requestId, int additionalTokens)
{
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
```

**Requirements**:
- Check capacity limits
- Allocate additional pages
- Rollback on failure
- Update allocation record
- Thread-safe

---

### GetAllocation
```csharp
public CacheAllocation? GetAllocation(RequestId requestId)
{
    lock (_lock)
    {
        return _allocations.TryGetValue(requestId, out var allocation)
            ? allocation
            : null;
    }
}
```

**Requirements**:
- Return allocation or null
- Thread-safe

---

### CompactCache
```csharp
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

private double CalculateFragmentationUtilization()
{
    int totalBlocks = _pagedCache.GetTotalBlockCount();
    int freeBlocks = _pagedCache.GetFreeBlockCount();

    if (totalBlocks == 0)
        return 0.0;

    return (double)(totalBlocks - freeBlocks) / totalBlocks;
}
```

**Requirements**:
- Check if compaction needed
- Check fragmentation level
- Compact if below threshold
- Thread-safe

---

## Files to Create

### Implementation
- `src/MLFramework/Inference/ContinuousBatching/KVCache/ContinuousBatchKVCacheManager.cs`
- `src/MLFramework/Inference/ContinuousBatching/KVCache/KVCacheConfiguration.cs`
- `src/MLFramework/Inference/ContinuousBatching/KVCache/CacheAllocation.cs`
- `src/MLFramework/Inference/ContinuousBatching/KVCache/CachePage.cs`

### Tests
- `tests/MLFramework.Tests/Inference/ContinuousBatching/KVCache/ContinuousBatchKVCacheManagerTests.cs`

---

## Dependencies
- `spec_continuous_batching_models.md` (RequestId)
- IPagedAttentionCache (from PagedAttention implementation)

---

## Testing Requirements

### Unit Tests (with Mock IPagedAttentionCache)
1. **Basic Allocation**:
   - Allocate cache successfully
   - Allocate correct number of pages
   - Return correct byte count

2. **Release Operations**:
   - Release allocated cache
   - Free all blocks
   - Handle missing allocations

3. **Cache Extension**:
   - Extend cache successfully
   - Respect max pages limit
   - Handle extension failures gracefully

4. **Usage Tracking**:
   - GetCurrentUsageBytes returns correct total
   - Updates after allocations
   - Updates after releases

5. **Allocation Retrieval**:
   - GetAllocation returns correct details
   - Handle missing allocations

6. **Edge Cases**:
   - Allocate with zero max tokens
   - Release non-existent allocation
   - Extend at maximum capacity
   - Concurrent allocation/release

7. **Compaction**:
   - Compact when needed
   - Skip when not needed
   - Handle configuration changes

8. **Thread Safety**:
   - Concurrent allocations
   - Concurrent releases
   - Mixed concurrent operations

---

## Success Criteria
- [ ] All public methods implemented
- [ ] Allocations work correctly
- [ ] Releases free blocks properly
- [ ] Extension handles edge cases
- [ ] Usage tracking accurate
- [ ] Thread-safe operations
- [ ] Integration with PagedAttention tested
- [ ] Unit tests cover all scenarios
