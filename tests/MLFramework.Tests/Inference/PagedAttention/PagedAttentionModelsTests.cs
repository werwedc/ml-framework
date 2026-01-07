using MlFramework.Core;
using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Models;
using Xunit;

namespace MLFramework.Tests.Inference.PagedAttention;

/// <summary>
/// Tests for MemoryBlock, BlockManagerStats, and BlockAllocationResult models
/// </summary>
public class PagedAttentionModelsTests
{
    [Fact]
    public void MemoryBlock_Constructor_InitializesCorrectly()
    {
        var block = new MemoryBlock(42, 16);

        Assert.Equal(42, block.BlockId);
        Assert.Equal(0, block.TokenCount);
        Assert.Null(block.SequenceId);
        Assert.False(block.IsAllocated);
    }

    [Fact]
    public void MemoryBlock_Allocate_SetsSequenceId()
    {
        var block = new MemoryBlock(1, 16);
        block.SequenceId = 100;

        Assert.Equal(100, block.SequenceId);
        Assert.True(block.IsAllocated);
    }

    [Fact]
    public void MemoryBlock_Reset_ClearsState()
    {
        var block = new MemoryBlock(1, 16);
        block.SequenceId = 100;
        block.StartTokenIndex = 32;
        block.TokenCount = 10;

        block.Reset();

        Assert.Null(block.SequenceId);
        Assert.Equal(0, block.StartTokenIndex);
        Assert.Equal(0, block.TokenCount);
        Assert.False(block.IsAllocated);
    }

    [Fact]
    public void BlockAllocationResult_Successful_CreatesSuccessResult()
    {
        var result = BlockAllocationResult.Successful(42);

        Assert.True(result.Success);
        Assert.Equal(42, result.BlockId);
        Assert.Null(result.ErrorMessage);
    }

    [Fact]
    public void BlockAllocationResult_Failed_CreatesFailureResult()
    {
        var result = BlockAllocationResult.Failed("No memory");

        Assert.False(result.Success);
        Assert.Equal("No memory", result.ErrorMessage);
    }

    [Fact]
    public void BlockManagerStats_CalculatesFreeBlocks()
    {
        var stats = new BlockManagerStats
        {
            TotalBlocks = 100,
            AllocatedBlocks = 75
        };

        Assert.Equal(25, stats.FreeBlocks);
    }

    [Fact]
    public void BlockManagerStats_ToString_FormatsCorrectly()
    {
        var stats = new BlockManagerStats
        {
            TotalBlocks = 100,
            AllocatedBlocks = 75,
            ActiveSequences = 10,
            TotalTokens = 500,
            MemoryUtilizationPercentage = 75.0
        };

        var str = stats.ToString();
        Assert.Contains("Free=25/100", str);
        Assert.Contains("ActiveSeqs=10", str);
        Assert.Contains("Utilization=75.0%", str);
    }
}
