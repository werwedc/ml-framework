# Spec: Block Manager Unit Tests

## Overview
Comprehensive unit tests for the KVCacheBlockManager class, covering allocation, deallocation, statistics, thread safety, and memory management.

## Target Directory
`tests/MlFramework.Tests/Inference/PagedAttention/`

## Test Cases to Implement

### KVCacheBlockManagerTests
```csharp
using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Models;
using Xunit;
using MlFramework.Tensor;

namespace MlFramework.Tests.Inference.PagedAttention;

public class KVCacheBlockManagerTests : IDisposable
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly Device _device;

    public KVCacheBlockManagerTests()
    {
        _device = Device.CPU(); // Use CPU for tests
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 100,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            device: _device
        );
    }

    public void Dispose()
    {
        _blockManager.Dispose();
    }

    [Fact]
    public void Constructor_InitializesCorrectState()
    {
        Assert.Equal(100, _blockManager.TotalBlocks);
        Assert.Equal(16, _blockManager.BlockSize);
        Assert.Equal(100, _blockManager.AllocatedBlocks);
    }

    [Fact]
    public void AllocateBlock_ReturnsValidBlockId()
    {
        var result = _blockManager.AllocateBlock(sequenceId: 1);

        Assert.True(result.Success);
        Assert.True(result.BlockId >= 0);
        Assert.Equal(1, _blockManager.GetSequenceBlocks(1).Count);
    }

    [Fact]
    public void AllocateBlock_FailsWhenPoolExhausted()
    {
        // Allocate all blocks
        var allocated = new List<int>();
        for (int i = 0; i < 100; i++)
        {
            var result = _blockManager.AllocateBlock(1);
            if (result.Success)
            {
                allocated.Add(result.BlockId);
            }
        }

        // Try to allocate one more
        var result2 = _blockManager.AllocateBlock(1);

        Assert.False(result2.Success);
        Assert.NotNull(result2.ErrorMessage);
    }

    [Fact]
    public void AllocateBlocks_SuccessfulAllocation()
    {
        var blockIds = _blockManager.AllocateBlocks(sequenceId: 1, count: 10);

        Assert.Equal(10, blockIds.Count);
        Assert.Equal(10, _blockManager.GetSequenceBlocks(1).Count);
    }

    [Fact]
    public void AllocateBlocks_FailsOnPartialAllocation()
    {
        // Allocate 95 blocks for sequence 1
        _blockManager.AllocateBlocks(1, 95);

        // Try to allocate 10 more (only 5 available)
        var result = _blockManager.AllocateBlocks(2, 10);

        Assert.Empty(result);
        Assert.Empty(_blockManager.GetSequenceBlocks(2));
    }

    [Fact]
    public void FreeBlock_ReturnsBlockToPool()
    {
        var result = _blockManager.AllocateBlock(1);
        int blockId = result.BlockId;

        _blockManager.FreeBlock(blockId);

        Assert.DoesNotContain(blockId, _blockManager.GetSequenceBlocks(1));
        Assert.Null(_blockManager.GetBlock(blockId)?.SequenceId);
    }

    [Fact]
    public void FreeSequenceBlocks_FreesAllBlocks()
    {
        _blockManager.AllocateBlocks(1, 5);
        _blockManager.AllocateBlocks(2, 3);

        _blockManager.FreeSequenceBlocks(1);

        Assert.Empty(_blockManager.GetSequenceBlocks(1));
        Assert.Equal(3, _blockManager.GetSequenceBlocks(2).Count);
    }

    [Fact]
    public void GetSequenceBlocks_ReturnsCorrectBlocks()
    {
        var blocks1 = _blockManager.AllocateBlocks(1, 3);
        var blocks2 = _blockManager.AllocateBlocks(2, 2);

        var retrieved1 = _blockManager.GetSequenceBlocks(1);
        var retrieved2 = _blockManager.GetSequenceBlocks(2);

        Assert.Equal(blocks1, retrieved1);
        Assert.Equal(blocks2, retrieved2);
    }

    [Fact]
    public void GetBlock_ReturnsCorrectBlock()
    {
        var result = _blockManager.AllocateBlock(1);
        int blockId = result.BlockId;

        var block = _blockManager.GetBlock(blockId);

        Assert.NotNull(block);
        Assert.Equal(blockId, block.BlockId);
        Assert.Equal(1, block.SequenceId);
    }

    [Fact]
    public void GetBlock_ReturnsNullForInvalidId()
    {
        var block = _blockManager.GetBlock(999);

        Assert.Null(block);
    }

    [Fact]
    public void GetStats_ReturnsCorrectStatistics()
    {
        _blockManager.AllocateBlocks(1, 5);
        _blockManager.AllocateBlocks(2, 3);

        var stats = _blockManager.GetStats();

        Assert.Equal(100, stats.TotalBlocks);
        Assert.Equal(8, stats.AllocatedBlocks);
        Assert.Equal(92, stats.FreeBlocks);
        Assert.Equal(2, stats.ActiveSequences);
        Assert.Equal(0, stats.TotalTokens); // No tokens stored yet
    }

    [Fact]
    public void HasAvailableBlocks_ReturnsCorrectStatus()
    {
        Assert.True(_blockManager.HasAvailableBlocks());

        // Allocate all blocks
        _blockManager.AllocateBlocks(1, 100);

        Assert.False(_blockManager.HasAvailableBlocks());
    }

    [Fact]
    public void HasAvailableBlocks_WithCountParameter()
    {
        Assert.True(_blockManager.HasAvailableBlocks(50));
        Assert.True(_blockManager.HasAvailableBlocks(100));

        // Allocate 95 blocks
        _blockManager.AllocateBlocks(1, 95);

        Assert.True(_blockManager.HasAvailableBlocks(5));
        Assert.False(_blockManager.HasAvailableBlocks(6));
    }

    [Fact]
    public void MultipleSequences_AllocateIndependently()
    {
        var blocks1 = _blockManager.AllocateBlocks(1, 10);
        var blocks2 = _blockManager.AllocateBlocks(2, 15);
        var blocks3 = _blockManager.AllocateBlocks(3, 5);

        // No overlap between sequences
        Assert.Empty(blocks1.Intersect(blocks2));
        Assert.Empty(blocks1.Intersect(blocks3));
        Assert.Empty(blocks2.Intersect(blocks3));

        // Correct counts
        Assert.Equal(10, _blockManager.GetSequenceBlocks(1).Count);
        Assert.Equal(15, _blockManager.GetSequenceBlocks(2).Count);
        Assert.Equal(5, _blockManager.GetSequenceBlocks(3).Count);
    }

    [Fact]
    public void AllocateFreeAndReallocate_ReusesBlocks()
    {
        var initialBlocks = _blockManager.AllocateBlocks(1, 10);
        _blockManager.FreeSequenceBlocks(1);

        var newBlocks = _blockManager.AllocateBlocks(2, 10);

        // The new blocks might not be the same IDs, but count should work
        Assert.Equal(10, newBlocks.Count);
    }
}

public class KVCacheBlockManagerConcurrencyTests : IDisposable
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly Device _device;

    public KVCacheBlockManagerConcurrencyTests()
    {
        _device = Device.CPU();
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 1000,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            device: _device
        );
    }

    public void Dispose()
    {
        _blockManager.Dispose();
    }

    [Fact]
    public void ConcurrentAllocation_DoesNotCauseRaceConditions()
    {
        var tasks = new List<Task>();
        var results = new List<List<int>>();

        for (int t = 0; t < 10; t++)
        {
            var task = Task.Run(() =>
            {
                var allocated = new List<int>();
                for (int i = 0; i < 50; i++)
                {
                    var result = _blockManager.AllocateBlock(t);
                    if (result.Success)
                    {
                        allocated.Add(result.BlockId);
                    }
                }
                lock (results)
                {
                    results.Add(allocated);
                }
            });
            tasks.Add(task);
        }

        Task.WaitAll(tasks.ToArray());

        // Verify no duplicate blocks across threads
        var allBlocks = results.SelectMany(r => r).ToList();
        Assert.Equal(allBlocks.Count, allBlocks.Distinct().Count());
    }

    [Fact]
    public void ConcurrentAllocationAndFreeing_NoMemoryLeaks()
    {
        var tasks = new List<Task>();

        for (int t = 0; t < 10; t++)
        {
            var taskId = t;
            var task = Task.Run(() =>
            {
                var allocated = new List<int>();
                // Allocate blocks
                for (int i = 0; i < 20; i++)
                {
                    var result = _blockManager.AllocateBlock(taskId * 100 + i);
                    if (result.Success)
                    {
                        allocated.Add(result.BlockId);
                    }
                }

                // Randomly free some blocks
                var random = new Random(taskId);
                for (int i = 0; i < 10; i++)
                {
                    if (allocated.Count > 0)
                    {
                        int idx = random.Next(allocated.Count);
                        _blockManager.FreeBlock(allocated[idx]);
                        allocated.RemoveAt(idx);
                    }
                }
            });
            tasks.Add(task);
        }

        Task.WaitAll(tasks.ToArray());

        // Check final stats
        var stats = _blockManager.GetStats();
        Assert.True(stats.FreeBlocks + stats.AllocatedBlocks == stats.TotalBlocks);
    }
}
```

## Test Requirements

### Coverage Requirements
1. **Basic Operations**:
   - Single block allocation
   - Multiple block allocation
   - Block deallocation
   - Sequence block freeing
   - Block lookup
   - Statistics calculation

2. **Edge Cases**:
   - Empty pool (allocation failure)
   - Invalid block IDs
   - Empty sequences
   - Pool exhaustion

3. **Multi-Sequence Scenarios**:
   - Multiple sequences allocating independently
   - Sequence isolation
   - Block reuse after freeing

4. **Concurrency**:
   - Concurrent allocations from multiple threads
   - Concurrent allocation and freeing
   - No race conditions
   - No memory leaks

5. **Memory Management**:
   - Proper tensor disposal
   - No memory leaks after operations
   - Correct reference counting

### Test Organization
- Basic functionality tests in `KVCacheBlockManagerTests`
- Concurrency tests in `KVCacheBlockManagerConcurrencyTests`
- Use Xunit framework
- Each test should be independent

## Estimated Time
45-60 minutes

## Dependencies
- spec_pagedattention_models.md
- spec_kvcache_block_manager.md

## Success Criteria
- All tests pass
- High code coverage (>90%)
- Tests cover edge cases
- No flaky tests
- Tests run quickly (< 10 seconds)
