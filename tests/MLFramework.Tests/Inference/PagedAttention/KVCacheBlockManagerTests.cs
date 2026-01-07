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
}
