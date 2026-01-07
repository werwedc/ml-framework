using MLFramework.Core;
using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Models;
using Xunit;

namespace MLFramework.Tests.Inference.PagedAttention;

/// <summary>
/// Tests for KVCacheBlockManager
/// </summary>
public class KVCacheBlockManagerTests
{
    private readonly DeviceId _deviceId = DeviceId.CPU;

    [Fact]
    public void Constructor_InitializesFreeBlockPool()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);

        Assert.Equal(100, manager.TotalBlocks);
        Assert.Equal(100, manager.AllocatedBlocks);
        Assert.Equal(16, manager.BlockSize);
    }

    [Fact]
    public void AllocateBlock_ReturnsBlockId()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);
        var result = manager.AllocateBlock(1);

        Assert.True(result.Success);
        Assert.Equal(0, result.BlockId);
        Assert.Equal(1, manager.AllocatedBlocks);
    }

    [Fact]
    public void AllocateBlock_TracksSequence()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);
        manager.AllocateBlock(1);
        manager.AllocateBlock(1);

        var blocks = manager.GetSequenceBlocks(1);
        Assert.Equal(2, blocks.Count);
    }

    [Fact]
    public void AllocateBlock_FailsWhenNoBlocks()
    {
        var manager = new KVCacheBlockManager(2, 16, 128, 32, 32, _deviceId);
        manager.AllocateBlock(1);
        manager.AllocateBlock(2);

        var result = manager.AllocateBlock(3);

        Assert.False(result.Success);
        Assert.Equal("No free blocks available", result.ErrorMessage);
    }

    [Fact]
    public void AllocateBlocks_AllocatesMultiple()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);
        var blocks = manager.AllocateBlocks(1, 3);

        Assert.Equal(3, blocks.Count);
        Assert.Equal(3, manager.AllocatedBlocks);
    }

    [Fact]
    public void AllocateBlocks_RollsBackOnFailure()
    {
        var manager = new KVCacheBlockManager(3, 16, 128, 32, 32, _deviceId);
        // Use 2 blocks
        manager.AllocateBlock(1);
        manager.AllocateBlock(2);

        // Try to allocate 3 more, should fail and rollback
        var blocks = manager.AllocateBlocks(3, 3);

        Assert.Empty(blocks);
        Assert.Equal(2, manager.AllocatedBlocks); // Still only 2 blocks allocated
    }

    [Fact]
    public void FreeBlock_ReturnsToPool()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);
        manager.AllocateBlock(1);
        manager.FreeBlock(0);

        Assert.Equal(0, manager.AllocatedBlocks);
    }

    [Fact]
    public void FreeBlock_ResetsBlockState()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);
        manager.AllocateBlock(1);
        var block = manager.GetBlock(0);
        Assert.NotNull(block);
        Assert.True(block.IsAllocated);

        manager.FreeBlock(0);
        block = manager.GetBlock(0);
        Assert.NotNull(block);
        Assert.False(block.IsAllocated);
        Assert.Null(block.SequenceId);
    }

    [Fact]
    public void FreeSequenceBlocks_FreesAllBlocks()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);
        manager.AllocateBlocks(1, 3);
        manager.AllocateBlocks(2, 2);

        manager.FreeSequenceBlocks(1);

        Assert.Equal(2, manager.AllocatedBlocks); // Only sequence 2's blocks remain
        var blocks = manager.GetSequenceBlocks(1);
        Assert.Empty(blocks);
    }

    [Fact]
    public void GetBlock_ReturnsCorrectBlock()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);
        manager.AllocateBlock(1);

        var block = manager.GetBlock(0);
        Assert.NotNull(block);
        Assert.Equal(0, block.BlockId);
        Assert.Equal(1, block.SequenceId);
    }

    [Fact]
    public void GetBlock_ReturnsNullForInvalidId()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);

        var block = manager.GetBlock(999);
        Assert.Null(block);
    }

    [Fact]
    public void GetStats_CalculatesCorrectly()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);
        manager.AllocateBlocks(1, 10);
        manager.AllocateBlocks(2, 5);

        var stats = manager.GetStats();

        Assert.Equal(100, stats.TotalBlocks);
        Assert.Equal(15, stats.AllocatedBlocks);
        Assert.Equal(85, stats.FreeBlocks);
        Assert.Equal(2, stats.ActiveSequences);
    }

    [Fact]
    public void HasAvailableBlocks_ReturnsCorrect()
    {
        var manager = new KVCacheBlockManager(5, 16, 128, 32, 32, _deviceId);

        Assert.True(manager.HasAvailableBlocks(5));
        Assert.False(manager.HasAvailableBlocks(6));

        manager.AllocateBlocks(1, 3);
        Assert.True(manager.HasAvailableBlocks(2));
        Assert.False(manager.HasAvailableBlocks(3));
    }

    [Fact]
    public void ConcurrentAllocations_ThreadSafe()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);
        var tasks = new List<Task>();

        for (int i = 0; i < 10; i++)
        {
            int seqId = i;
            tasks.Add(Task.Run(() =>
            {
                for (int j = 0; j < 5; j++)
                {
                    manager.AllocateBlock(seqId);
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        Assert.Equal(50, manager.AllocatedBlocks);
        Assert.Equal(10, manager.GetSequenceBlocks(1).Count);
    }

    [Fact]
    public void AllocateAndFree_TracksCounts()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);

        manager.AllocateBlocks(1, 10);
        manager.FreeSequenceBlocks(1);

        var stats = manager.GetStats();
        Assert.Equal(10, stats.AllocationCount);
        Assert.Equal(10, stats.DeallocationCount);
    }

    [Fact]
    public void MultipleSequences_AllocateIndependently()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);

        var blocks1 = manager.AllocateBlocks(1, 10);
        var blocks2 = manager.AllocateBlocks(2, 15);
        var blocks3 = manager.AllocateBlocks(3, 5);

        // No overlap between sequences
        Assert.Empty(blocks1.Intersect(blocks2));
        Assert.Empty(blocks1.Intersect(blocks3));
        Assert.Empty(blocks2.Intersect(blocks3));

        // Correct counts
        Assert.Equal(10, manager.GetSequenceBlocks(1).Count);
        Assert.Equal(15, manager.GetSequenceBlocks(2).Count);
        Assert.Equal(5, manager.GetSequenceBlocks(3).Count);
    }

    [Fact]
    public void AllocateFreeAndReallocate_ReusesBlocks()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);

        var initialBlocks = manager.AllocateBlocks(1, 10);
        manager.FreeSequenceBlocks(1);

        var newBlocks = manager.AllocateBlocks(2, 10);

        // The new blocks might not be the same IDs, but count should work
        Assert.Equal(10, newBlocks.Count);
        // All blocks should be allocated
        Assert.Equal(10, manager.AllocatedBlocks);
    }

    [Fact]
    public void AllocateBlock_UpdatesSequenceTracking()
    {
        var manager = new KVCacheBlockManager(100, 16, 128, 32, 32, _deviceId);

        var result1 = manager.AllocateBlock(1);
        var result2 = manager.AllocateBlock(1);
        var result3 = manager.AllocateBlock(2);

        Assert.True(result1.Success);
        Assert.True(result2.Success);
        Assert.True(result3.Success);

        var blocks1 = manager.GetSequenceBlocks(1);
        var blocks2 = manager.GetSequenceBlocks(2);

        Assert.Equal(2, blocks1.Count);
        Assert.Equal(1, blocks2.Count);
        Assert.Contains(result1.BlockId, blocks1);
        Assert.Contains(result2.BlockId, blocks1);
        Assert.Contains(result3.BlockId, blocks2);
    }
}

/// <summary>
/// Concurrency tests for KVCacheBlockManager
/// </summary>
[Collection("Sequential")] // Run sequentially to avoid interference
public class KVCacheBlockManagerConcurrencyTests : IDisposable
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly DeviceId _deviceId = DeviceId.CPU;

    public KVCacheBlockManagerConcurrencyTests()
    {
        _blockManager = new KVCacheBlockManager(1000, 16, 128, 32, 32, _deviceId);
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
            int taskId = t;
            var task = Task.Run(() =>
            {
                var allocated = new List<int>();
                for (int i = 0; i < 50; i++)
                {
                    var result = _blockManager.AllocateBlock(taskId);
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

        // Verify all blocks were allocated
        Assert.Equal(500, _blockManager.AllocatedBlocks);
    }

    [Fact]
    public void ConcurrentAllocationAndFreeing_NoMemoryLeaks()
    {
        var tasks = new List<Task>();

        for (int t = 0; t < 10; t++)
        {
            int taskId = t;
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

        // Some blocks should still be allocated (not all were freed)
        Assert.True(stats.AllocatedBlocks > 0);
    }

    [Fact]
    public void ConcurrentMultipleAllocation_ThreadSafe()
    {
        var tasks = new List<Task>();
        var results = new List<List<int>>();

        for (int t = 0; t < 5; t++)
        {
            int taskId = t;
            var task = Task.Run(() =>
            {
                var allocated = new List<int>();
                // Try to allocate 20 blocks multiple times
                for (int attempt = 0; attempt < 10; attempt++)
                {
                    var blocks = _blockManager.AllocateBlocks(taskId * 1000 + attempt, 20);
                    if (blocks.Count > 0)
                    {
                        allocated.AddRange(blocks);
                    }
                    else
                    {
                        // Free and try again
                        _blockManager.FreeSequenceBlocks(taskId * 1000 + attempt);
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
    public void ConcurrentSequenceBlockAccess_ThreadSafe()
    {
        var tasks = new List<Task>();

        // Pre-allocate blocks for sequence 1
        _blockManager.AllocateBlocks(1, 100);

        for (int t = 0; t < 10; t++)
        {
            var task = Task.Run(() =>
            {
                for (int i = 0; i < 100; i++)
                {
                    // Concurrently read sequence blocks
                    var blocks = _blockManager.GetSequenceBlocks(1);
                    Assert.True(blocks.Count >= 0);
                }
            });
            tasks.Add(task);
        }

        Task.WaitAll(tasks.ToArray());

        // Verify sequence still has correct number of blocks
        Assert.Equal(100, _blockManager.GetSequenceBlocks(1).Count);
    }
}
