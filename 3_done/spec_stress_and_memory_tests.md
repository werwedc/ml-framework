# Spec: Stress and Memory Leak Tests

## Overview
Long-running stress tests and memory leak detection tests for PagedAttention. These tests verify system stability under heavy load and ensure proper resource cleanup.

## Target Directory
`tests/MlFramework.Tests/Inference/PagedAttention/Stress/`

## Test Cases to Implement

### PagedAttentionStressTests
```csharp
using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Kernels;
using MlFramework.Inference.PagedAttention.Phases;
using MlFramework.Tensor;
using Xunit;
using System.Diagnostics;

namespace MlFramework.Tests.Inference.PagedAttention.Stress;

[Collection("Sequential")] // Run sequentially to avoid interference
public class PagedAttentionStressTests : IDisposable
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly BlockTable _blockTable;
    private readonly StandardPagedAttentionKernel _kernel;
    private readonly Device _device;

    public PagedAttentionStressTests()
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
        _blockTable = new BlockTable(_blockManager);
        _kernel = new StandardPagedAttentionKernel(128);
    }

    public void Dispose()
    {
        _blockManager.Dispose();
    }

    [Fact(Skip = "Long-running test - run manually")]
    public void LongRunningInference_MaintainsStability()
    {
        // Run for a significant duration (e.g., 10 minutes)
        var duration = TimeSpan.FromMinutes(10);
        var startTime = DateTime.UtcNow;
        var random = new Random(42);

        int iteration = 0;
        while (DateTime.UtcNow - startTime < duration)
        {
            iteration++;

            // Simulate batch of sequences
            int numSequences = random.Next(10, 50);
            var activeSequences = new List<int>();

            for (int seqId = 0; seqId < numSequences; seqId++)
            {
                int seqLength = random.Next(64, 512);

                // Allocate blocks
                int blocksNeeded = (seqLength + 15) / 16;
                var blocks = _blockManager.AllocateBlocks(seqId, blocksNeeded);

                if (blocks.Count > 0)
                {
                    activeSequences.Add(seqId);
                    _blockTable.AllocateAndAppendBlock(seqId);

                    // Simulate attention computation
                    var query = CreateRandomTensor(new[] { 1, 1, 32, 128 });
                    var cachedKeys = CreateRandomTensor(new[] { 1, seqLength, 32, 128 });
                    var cachedValues = CreateRandomTensor(new[] { 1, seqLength, 32, 128 });

                    _kernel.ComputePagedAttention(
                        query,
                        cachedKeys,
                        cachedValues,
                        AttentionPhase.Decode
                    );
                }
            }

            // Randomly free some sequences
            int numToFree = random.Next(1, activeSequences.Count / 2);
            for (int i = 0; i < numToFree; i++)
            {
                int idx = random.Next(activeSequences.Count);
                int seqId = activeSequences[idx];
                activeSequences.RemoveAt(idx);

                _blockManager.FreeSequenceBlocks(seqId);
                _blockTable.RemoveSequence(seqId);
            }

            // Check stats every 100 iterations
            if (iteration % 100 == 0)
            {
                var stats = _blockManager.GetStats();
                Assert.True(stats.FreeBlocks >= 0);
                Assert.True(stats.AllocatedBlocks <= _blockManager.TotalBlocks);
            }
        }

        // Final validation
        var finalStats = _blockManager.GetStats();
        Assert.Equal(_blockManager.TotalBlocks,
                   finalStats.FreeBlocks + finalStats.AllocatedBlocks);
    }

    [Fact]
    public void RapidSequenceTurnover_NoResourceLeaks()
    {
        // Create and destroy many sequences rapidly
        for (int i = 0; i < 10000; i++)
        {
            int seqId = i % 100; // Reuse IDs
            int seqLength = 16;

            // Allocate
            _blockManager.AllocateBlocks(seqId, 1);
            _blockTable.AllocateAndAppendBlock(seqId);

            // Free
            _blockManager.FreeSequenceBlocks(seqId);
            _blockTable.RemoveSequence(seqId);
        }

        // Verify all blocks are free
        var stats = _blockManager.GetStats();
        Assert.Equal(0, stats.AllocatedBlocks);
        Assert.Equal(_blockManager.TotalBlocks, stats.FreeBlocks);
    }

    [Fact]
    public void VariableLengthSequences_ProperBlockAllocation()
    {
        var random = new Random(42);
        var sequences = new Dictionary<int, int>(); // seqId -> length

        // Create sequences with varying lengths
        for (int i = 0; i < 100; i++)
        {
            int seqId = i;
            int seqLength = random.Next(16, 1000);
            sequences[seqId] = seqLength;

            int blocksNeeded = (seqLength + 15) / 16;
            var blocks = _blockManager.AllocateBlocks(seqId, blocksNeeded);

            Assert.True(blocks.Count > 0);
            Assert.Equal(blocksNeeded, blocks.Count);
        }

        // Verify all sequences have correct number of blocks
        foreach (var kvp in sequences)
        {
            int seqId = kvp.Key;
            int seqLength = kvp.Value;
            int expectedBlocks = (seqLength + 15) / 16;

            var actualBlocks = _blockTable.GetSequenceBlocks(seqId);
            Assert.Equal(expectedBlocks, actualBlocks.Count);
        }

        // Cleanup
        foreach (var seqId in sequences.Keys)
        {
            _blockManager.FreeSequenceBlocks(seqId);
            _blockTable.RemoveSequence(seqId);
        }
    }

    [Fact]
    public void PoolExhaustion_GracefulFailure()
    {
        // Allocate blocks until pool is exhausted
        int seqId = 1;
        while (true)
        {
            var result = _blockManager.AllocateBlock(seqId);
            if (!result.Success)
            {
                break;
            }
        }

        // Try to allocate more - should fail
        var result2 = _blockManager.AllocateBlock(seqId + 1);
        Assert.False(result2.Success);
        Assert.NotNull(result2.ErrorMessage);

        // Free some blocks
        _blockManager.FreeSequenceBlocks(seqId);

        // Should now be able to allocate
        var result3 = _blockManager.AllocateBlock(seqId + 1);
        Assert.True(result3.Success);
    }
}

public class MemoryLeakTests : IDisposable
{
    private readonly Device _device;

    public MemoryLeakTests()
    {
        _device = Device.CPU();
    }

    public void Dispose()
    {
        // Cleanup
    }

    [Fact]
    public void BlockAllocationAndFreeing_NoMemoryLeak()
    {
        var initialMemory = GC.GetTotalMemory(true);
        var allocations = new List<KVCacheBlockManager>();

        // Perform many allocation/free cycles
        for (int i = 0; i < 100; i++)
        {
            var blockManager = new KVCacheBlockManager(
                totalBlocks: 100,
                blockSize: 16,
                headDim: 128,
                numLayers: 32,
                numAttentionHeads: 32,
                device: _device
            );

            // Allocate blocks
            blockManager.AllocateBlocks(1, 50);
            blockManager.AllocateBlocks(2, 50);

            // Free blocks
            blockManager.FreeSequenceBlocks(1);
            blockManager.FreeSequenceBlocks(2);

            allocations.Add(blockManager);
        }

        // Dispose all managers
        foreach (var manager in allocations)
        {
            manager.Dispose();
        }

        allocations.Clear();

        // Force garbage collection
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var finalMemory = GC.GetTotalMemory(true);

        // Memory should not have grown significantly
        var growth = finalMemory - initialMemory;
        Assert.True(growth < 100_000_000, // 100MB threshold
                   $"Memory growth: {growth / 1_000_000}MB");
    }

    [Fact]
    public void TensorDisposal_NoMemoryLeak()
    {
        var initialMemory = GC.GetTotalMemory(true);

        // Create and dispose many tensors
        for (int i = 0; i < 1000; i++)
        {
            var tensor = Tensor.Zeros(new[] { 32, 128, 128, 128 }, _device);
            tensor.Dispose();
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var finalMemory = GC.GetTotalMemory(true);

        // Memory should be stable
        var growth = finalMemory - initialMemory;
        Assert.True(growth < 50_000_000, // 50MB threshold
                   $"Memory growth: {growth / 1_000_000}MB");
    }

    [Fact]
    public void BlockTableOperations_NoMemoryLeak()
    {
        var initialMemory = GC.GetTotalMemory(true);

        var blockManager = new KVCacheBlockManager(
            totalBlocks: 1000,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            device: _device
        );
        var blockTable = new BlockTable(blockManager);

        // Perform many block table operations
        for (int i = 0; i < 10000; i++)
        {
            int seqId = i % 100;

            // Allocate blocks
            blockTable.AllocateAndAppendBlock(seqId);

            // Get stats
            blockTable.GetStats();

            // Lookup
            blockTable.GetBlock(seqId, 0);

            // Occasionally remove sequence
            if (i % 100 == 0)
            {
                blockTable.RemoveSequence(seqId);
            }
        }

        blockManager.Dispose();

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var finalMemory = GC.GetTotalMemory(true);

        var growth = finalMemory - initialMemory;
        Assert.True(growth < 100_000_000,
                   $"Memory growth: {growth / 1_000_000}MB");
    }

    private Tensor CreateRandomTensor(int[] shape)
    {
        var data = new float[shape.Product()];
        var random = new Random(42);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)random.NextDouble();
        }
        return new Tensor(data, shape, _device);
    }
}
```

## Test Requirements

### Stress Test Categories
1. **Long-Running Stability**:
   - Continuous inference for extended periods
   - Memory stability over time
   - Statistics consistency
   - No degradation in performance

2. **Resource Turnover**:
   - Rapid sequence creation/destruction
   - Block allocation/deallocation cycles
   - Variable-length sequences
   - Pool exhaustion scenarios

3. **Memory Leak Detection**:
   - Block manager lifecycle
   - Tensor disposal
   - Block table operations
   - Aggregate resource tracking

### Validation Criteria
- No memory leaks (<100MB growth over 10k operations)
- Stable memory usage during long runs
- All resources properly disposed
- Correct block accounting after operations
- Graceful handling of resource exhaustion

## Estimated Time
45-60 minutes

## Dependencies
- spec_pagedattention_models.md
- spec_kvcache_block_manager.md
- spec_block_table.md

## Success Criteria
- All tests pass
- No memory leaks detected
- Stable long-running behavior
- Proper resource cleanup
- Tests run reliably
