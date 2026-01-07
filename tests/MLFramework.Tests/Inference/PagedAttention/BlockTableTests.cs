using MLFramework.Core;
using MlFramework.Inference.PagedAttention;
using Xunit;

namespace MLFramework.Tests.Inference.PagedAttention;

/// <summary>
/// Tests for BlockTable
/// </summary>
public class BlockTableTests
{
    private KVCacheBlockManager CreateManager()
    {
        return new KVCacheBlockManager(100, 16, 128, 32, 32, DeviceId.CPU);
    }

    [Fact]
    public void Constructor_InitializesEmpty()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        var stats = table.GetStats();
        Assert.Equal(0, stats.TotalSequences);
        Assert.Equal(0, stats.TotalBlocks);
    }

    [Fact]
    public void GetBlock_ReturnsMinusOneWhenNotFound()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        var blockId = table.GetBlock(1, 0);
        Assert.Equal(-1, blockId);
    }

    [Fact]
    public void GetSequenceBlocks_ReturnsEmptyListForNewSequence()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        var blocks = table.GetSequenceBlocks(1);
        Assert.Empty(blocks);
    }

    [Fact]
    public void GetSequenceBlockCount_ReturnsZeroForNewSequence()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        var count = table.GetSequenceBlockCount(1);
        Assert.Equal(0, count);
    }

    [Fact]
    public void GetSequenceLength_ReturnsZeroForNewSequence()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        var length = table.GetSequenceLength(1);
        Assert.Equal(0, length);
    }

    [Fact]
    public void AppendBlock_AddsBlockToSequence()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlock(1);
        table.AppendBlock(1, 0);

        var blocks = table.GetSequenceBlocks(1);
        Assert.Single(blocks);
        Assert.Equal(0, blocks[0]);
    }

    [Fact]
    public void AppendBlock_MapsTokensToBlock()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlock(1);
        table.AppendBlock(1, 0);

        // Block size is 16, so tokens 0-15 should map to block 0
        Assert.Equal(0, table.GetBlock(1, 0));
        Assert.Equal(0, table.GetBlock(1, 15));
    }

    [Fact]
    public void AppendMultipleBlocks_MaintainsOrder()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlocks(1, 3);
        table.AppendBlock(1, 0);
        table.AppendBlock(1, 1);
        table.AppendBlock(1, 2);

        var blocks = table.GetSequenceBlocks(1);
        Assert.Equal(3, blocks.Count);
        Assert.Equal(0, blocks[0]);
        Assert.Equal(1, blocks[1]);
        Assert.Equal(2, blocks[2]);
    }

    [Fact]
    public void GetBlock_MapsCorrectBlockForToken()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlocks(1, 3);
        table.AppendBlock(1, 0);
        table.AppendBlock(1, 1);
        table.AppendBlock(1, 2);

        // Block 0: tokens 0-15
        // Block 1: tokens 16-31
        // Block 2: tokens 32-47
        Assert.Equal(0, table.GetBlock(1, 0));
        Assert.Equal(0, table.GetBlock(1, 15));
        Assert.Equal(1, table.GetBlock(1, 16));
        Assert.Equal(1, table.GetBlock(1, 31));
        Assert.Equal(2, table.GetBlock(1, 32));
        Assert.Equal(2, table.GetBlock(1, 47));
    }

    [Fact]
    public void GetSequenceLength_CalculatesCorrectly()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlocks(1, 3);
        table.AppendBlock(1, 0);
        table.AppendBlock(1, 1);
        table.AppendBlock(1, 2);

        // 3 blocks * 16 tokens/block = 48 tokens
        Assert.Equal(48, table.GetSequenceLength(1));
    }

    [Fact]
    public void AllocateAndAppendBlock_CombinesOperations()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        var blockId = table.AllocateAndAppendBlock(1);

        Assert.NotEqual(-1, blockId);
        Assert.Equal(1, table.GetSequenceBlockCount(1));
    }

    [Fact]
    public void AllocateAndAppendBlock_ReturnsMinusOneOnFailure()
    {
        var manager = new KVCacheBlockManager(2, 16, 128, 32, 32, DeviceId.CPU);
        var table = new BlockTable(manager);

        // Allocate all blocks
        manager.AllocateBlocks(1, 2);

        // Try to allocate more
        var blockId = table.AllocateAndAppendBlock(2);

        Assert.Equal(-1, blockId);
    }

    [Fact]
    public void RemoveSequence_RemovesMappings()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlocks(1, 2);
        table.AppendBlock(1, 0);
        table.AppendBlock(1, 1);

        table.RemoveSequence(1);

        Assert.Equal(0, table.GetSequenceBlockCount(1));
        Assert.Equal(-1, table.GetBlock(1, 0));
    }

    [Fact]
    public void RemoveSequence_DoesNotDeallocateBlocks()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlocks(1, 2);
        table.AppendBlock(1, 0);
        table.AppendBlock(1, 1);

        table.RemoveSequence(1);

        // Blocks should still be allocated in the manager
        Assert.Equal(2, manager.AllocatedBlocks);
    }

    [Fact]
    public void Clear_RemovesAllMappings()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlocks(1, 2);
        manager.AllocateBlocks(2, 1);
        table.AppendBlock(1, 0);
        table.AppendBlock(1, 1);
        table.AppendBlock(2, 2);

        table.Clear();

        Assert.Equal(0, table.GetSequenceBlockCount(1));
        Assert.Equal(0, table.GetSequenceBlockCount(2));
    }

    [Fact]
    public void GetActiveSequenceIds_ReturnsAllSequences()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlocks(1, 1);
        manager.AllocateBlocks(2, 1);
        manager.AllocateBlocks(3, 1);
        table.AppendBlock(1, 0);
        table.AppendBlock(2, 1);
        table.AppendBlock(3, 2);

        var ids = table.GetActiveSequenceIds().ToList();
        Assert.Equal(3, ids.Count);
        Assert.Contains(1, ids);
        Assert.Contains(2, ids);
        Assert.Contains(3, ids);
    }

    [Fact]
    public void ContainsSequence_ReturnsCorrectly()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlock(1);
        table.AppendBlock(1, 0);

        Assert.True(table.ContainsSequence(1));
        Assert.False(table.ContainsSequence(2));
    }

    [Fact]
    public void GetStats_CalculatesCorrectly()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlocks(1, 3);
        manager.AllocateBlocks(2, 2);
        table.AppendBlock(1, 0);
        table.AppendBlock(1, 1);
        table.AppendBlock(1, 2);
        table.AppendBlock(2, 3);
        table.AppendBlock(2, 4);

        var stats = table.GetStats();

        Assert.Equal(2, stats.TotalSequences);
        Assert.Equal(5, stats.TotalBlocks);
        Assert.Equal(80, stats.TotalTokens); // 5 blocks * 16 tokens
        Assert.Equal(2.5, stats.AverageBlocksPerSequence);
    }

    [Fact]
    public void ConcurrentOperations_ThreadSafe()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);
        var tasks = new List<Task>();

        // Pre-allocate blocks for all sequences
        var blocks = manager.AllocateBlocks(0, 50);

        for (int i = 0; i < 10; i++)
        {
            int seqId = i;
            tasks.Add(Task.Run(() =>
            {
                for (int j = 0; j < 5; j++)
                {
                    int blockIndex = seqId * 5 + j;
                    table.AppendBlock(seqId, blocks[blockIndex]);
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        var stats = table.GetStats();
        Assert.Equal(10, stats.TotalSequences);
        Assert.Equal(50, stats.TotalBlocks);
    }

    [Fact]
    public void EmptySequence_HandlesCorrectly()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        Assert.Equal(0, table.GetSequenceBlockCount(999));
        Assert.Empty(table.GetSequenceBlocks(999));
        Assert.Equal(0, table.GetSequenceLength(999));
        Assert.Equal(-1, table.GetBlock(999, 0));
    }

    [Fact]
    public void SingleTokenSequence_HandlesCorrectly()
    {
        var manager = CreateManager();
        var table = new BlockTable(manager);

        manager.AllocateBlock(1);
        table.AppendBlock(1, 0);

        // Even with one block, we should be able to look up token 0
        Assert.Equal(0, table.GetBlock(1, 0));
        Assert.Equal(1, table.GetSequenceBlockCount(1));
    }
}
