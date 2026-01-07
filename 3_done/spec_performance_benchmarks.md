# Spec: Performance Benchmark Tests

## Overview
Performance benchmark tests to measure throughput improvements of PagedAttention compared to contiguous KV cache. These tests validate the expected 2-4x performance gains in serving workloads.

## Target Directory
`tests/MlFramework.Benchmarks/Inference/PagedAttention/`

## Test Cases to Implement

### PagedAttentionBenchmarks
```csharp
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Kernels;
using MlFramework.Tensor;

namespace MlFramework.Benchmarks.Inference.PagedAttention;

[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class PagedAttentionBenchmarks
{
    private KVCacheBlockManager _blockManager = null!;
    private BlockTable _blockTable = null!;
    private StandardPagedAttentionKernel _kernel = null!;
    private Device _device = null!;
    private Tensor _query = null!;
    private Tensor _cachedKeys = null!;
    private Tensor _cachedValues = null!;
    private Tensor _contiguousQuery = null!;
    private Tensor _contiguousKeys = null!;
    private Tensor _contiguousValues = null!;

    private const int NumHeads = 32;
    private const int HeadDim = 128;
    private const int BlockSize = 16;
    private const int NumLayers = 32;

    [Params(16, 32, 64, 128, 256, 512, 1024)]
    public int SequenceLength { get; set; }

    [Params(1, 4, 8, 16)]
    public int BatchSize { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _device = Device.CPU();
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 1000,
            blockSize: BlockSize,
            headDim: HeadDim,
            numLayers: NumLayers,
            numAttentionHeads: NumHeads,
            device: _device
        );
        _blockTable = new BlockTable(_blockManager);
        _kernel = new StandardPagedAttentionKernel(HeadDim);

        // Create test data
        _query = CreateRandomTensor(new[] { BatchSize, 1, NumHeads, HeadDim });
        _cachedKeys = CreateRandomTensor(new[] { BatchSize, SequenceLength, NumHeads, HeadDim });
        _cachedValues = CreateRandomTensor(new[] { BatchSize, SequenceLength, NumHeads, HeadDim });

        _contiguousQuery = _query.Clone();
        _contiguousKeys = _cachedKeys.Clone();
        _contiguousValues = _cachedValues.Clone();
    }

    [Benchmark]
    [BenchmarkCategory("PagedAttention")]
    public Tensor PagedAttention_SingleToken()
    {
        return _kernel.ComputePagedAttention(
            _query,
            _cachedKeys,
            _cachedValues,
            AttentionPhase.Decode
        );
    }

    [Benchmark]
    [BenchmarkCategory("ContiguousAttention")]
    public Tensor ContiguousAttention_SingleToken()
    {
        return ComputeContiguousAttention(
            _contiguousQuery,
            _contiguousKeys,
            _contiguousValues
        );
    }

    [Benchmark]
    [BenchmarkCategory("PagedAttention")]
    public Tensor PagedAttention_Prefill()
    {
        var prefillQuery = CreateRandomTensor(
            new[] { BatchSize, SequenceLength, NumHeads, HeadDim }
        );
        return _kernel.ComputePagedAttention(
            prefillQuery,
            _cachedKeys,
            _cachedValues,
            AttentionPhase.Prefill
        );
    }

    [Benchmark]
    [BenchmarkCategory("ContiguousAttention")]
    public Tensor ContiguousAttention_Prefill()
    {
        var prefillQuery = CreateRandomTensor(
            new[] { BatchSize, SequenceLength, NumHeads, HeadDim }
        );
        return ComputeContiguousAttention(
            prefillQuery,
            _contiguousKeys,
            _contiguousValues
        );
    }

    private Tensor ComputeContiguousAttention(Tensor query, Tensor keys, Tensor values)
    {
        // Standard contiguous attention implementation
        // This should match the actual contiguous implementation in the framework
        throw new NotImplementedException();
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

[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class BlockManagerBenchmarks
{
    private KVCacheBlockManager _blockManager = null!;
    private Device _device = null!;

    [Params(100, 500, 1000, 5000)]
    public int TotalBlocks { get; set; }

    [Params(16, 32, 64)]
    public int SequenceLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _device = Device.CPU();
        _blockManager = new KVCacheBlockManager(
            totalBlocks: TotalBlocks,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            device: _device
        );
    }

    [Benchmark]
    public int AllocateSingleBlock()
    {
        return _blockManager.AllocateBlock(1).BlockId;
    }

    [Benchmark]
    public List<int> AllocateMultipleBlocks()
    {
        int blocksNeeded = (SequenceLength + 15) / 16;
        return _blockManager.AllocateBlocks(1, blocksNeeded);
    }

    [Benchmark]
    public void FreeSingleBlock()
    {
        var result = _blockManager.AllocateBlock(1);
        _blockManager.FreeBlock(result.BlockId);
    }

    [Benchmark]
    public void FreeSequenceBlocks()
    {
        var blocks = _blockManager.AllocateBlocks(1, 10);
        _blockManager.FreeSequenceBlocks(1);
    }

    [Benchmark]
    public BlockManagerStats GetStatistics()
    {
        return _blockManager.GetStats();
    }
}

[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class BlockTableBenchmarks
{
    private BlockTable _blockTable = null!;
    private KVCacheBlockManager _blockManager = null!;
    private Device _device = null!;

    [Params(100, 1000, 10000)]
    public int SequenceLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _device = Device.CPU();
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 10000,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            device: _device
        );
        _blockTable = new BlockTable(_blockManager);

        // Pre-allocate blocks
        for (int i = 0; i < SequenceLength; i += 16)
        {
            var blockId = _blockManager.AllocateBlock(1).BlockId;
            _blockTable.AppendBlock(1, blockId);
        }
    }

    [Benchmark]
    public int GetBlockLookup()
    {
        return _blockTable.GetBlock(1, SequenceLength / 2);
    }

    [Benchmark]
    public List<int> GetSequenceBlocks()
    {
        return _blockTable.GetSequenceBlocks(1);
    }

    [Benchmark]
    public int GetSequenceLength()
    {
        return _blockTable.GetSequenceLength(1);
    }

    [Benchmark]
    public void AppendBlock()
    {
        var blockId = _blockManager.AllocateBlock(2).BlockId;
        _blockTable.AppendBlock(2, blockId);
    }

    [Benchmark]
    public BlockTableStats GetStats()
    {
        return _blockTable.GetStats();
    }
}

[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class ServingWorkloadBenchmark
{
    private KVCacheBlockManager _blockManager = null!;
    private BlockTable _blockTable = null!;
    private StandardPagedAttentionKernel _kernel = null!;
    private Device _device = null!;

    [Params(10, 50, 100)]
    public int NumSequences { get; set; }

    [Params(64, 128, 256)]
    public int AvgSequenceLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _device = Device.CPU();
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 5000,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            device: _device
        );
        _blockTable = new BlockTable(_blockManager);
        _kernel = new StandardPagedAttentionKernel(128);
    }

    [Benchmark]
    public void SimulateServingWorkload()
    {
        // Simulate continuous batching workload
        var random = new Random(42);

        for (int iter = 0; iter < 100; iter++)
        {
            // Process each sequence
            foreach (var seqId in Enumerable.Range(1, NumSequences))
            {
                // Random sequence length variation
                int seqLength = AvgSequenceLength + random.Next(-32, 33);

                // Simulate decode step
                var query = CreateRandomTensor(new[] { 1, 1, 32, 128 });

                var cachedKeys = CreateRandomTensor(
                    new[] { 1, seqLength, 32, 128 }
                );
                var cachedValues = CreateRandomTensor(
                    new[] { 1, seqLength, 32, 128 }
                );

                _kernel.ComputePagedAttention(
                    query,
                    cachedKeys,
                    cachedValues,
                    AttentionPhase.Decode
                );

                // Occasionally free a sequence and allocate new one
                if (random.Next(100) < 5)
                {
                    _blockManager.FreeSequenceBlocks(seqId);
                    _blockTable.RemoveSequence(seqId);
                }
            }
        }
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

## Benchmark Requirements

### Benchmark Categories
1. **Attention Performance**:
   - Paged vs Contiguous attention for single token
   - Paged vs Contiguous attention for prefill
   - Various sequence lengths (16-1024 tokens)
   - Various batch sizes (1-16)

2. **Block Manager Performance**:
   - Block allocation speed
   - Block deallocation speed
   - Statistics retrieval
   - Sequence management

3. **Block Table Performance**:
   - Token-to-block lookup speed
   - Sequence block retrieval
   - Block append operations
   - Statistics computation

4. **Serving Workload Simulation**:
   - Continuous batching scenarios
   - Multiple concurrent sequences
   - Dynamic sequence arrival/departure
   - Mixed-length sequences

### Metrics to Collect
- **Throughput**: Tokens per second
- **Latency**: Per-token latency
- **Memory Usage**: Peak memory allocation
- **Memory Efficiency**: Utilization percentage
- **Block Reuse**: Rate of block reclamation

### Performance Targets
- **Throughput**: 2-4x improvement vs contiguous baseline
- **Memory Overhead**: <2% for metadata
- **Memory Utilization**: >90% for mixed-length workloads
- **Latency**: No degradation vs contiguous (or improvement)

## Estimated Time
45-60 minutes

## Dependencies
- spec_pagedattention_models.md
- spec_kvcache_block_manager.md
- spec_block_table.md
- spec_attention_kernel_interface.md

## Success Criteria
- Benchmarks run successfully
- Performance targets met (2-4x throughput improvement)
- Memory overhead within limits (<2%)
- Utilization >90% for mixed workloads
- Results are reproducible
- Benchmark documentation clear
