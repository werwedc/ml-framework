using BenchmarkDotNet.Attributes;
using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Kernels;
using MlFramework.Inference.PagedAttention.Models;
using MlFramework.Inference.PagedAttention.Phases;
using MLFramework.Core;
using RitterFramework.Core.Tensor;

namespace MlFramework.Benchmarks.Inference.PagedAttention;

[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class PagedAttentionBenchmarks
{
    private KVCacheBlockManager _blockManager = null!;
    private BlockTable _blockTable = null!;
    private StandardPagedAttentionKernel _kernel = null!;
    private DeviceId _deviceId = DeviceId.CPU;
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
        _deviceId = DeviceId.CPU;
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 1000,
            blockSize: BlockSize,
            headDim: HeadDim,
            numLayers: NumLayers,
            numAttentionHeads: NumHeads,
            deviceId: _deviceId
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
        // This matches the reference implementation in StandardPagedAttentionKernel
        var headDim = query.Shape[3];
        var scale = 1.0f / MathF.Sqrt(headDim);

        var attentionScores = MatmulQueryKey(query, keys);
        attentionScores = ScaleTensor(attentionScores, scale);
        var attentionWeights = Softmax(attentionScores, dim: -1);
        var output = MatmulWeightValue(attentionWeights, values);

        return output;
    }

    private Tensor MatmulQueryKey(Tensor query, Tensor keys)
    {
        var batchSize = query.Shape[0];
        var numQueries = query.Shape[1];
        var numHeads = query.Shape[2];
        var headDim = query.Shape[3];
        var numCached = keys.Shape[1];

        var outputSize = batchSize * numQueries * numHeads * numCached;
        var outputData = new float[outputSize];

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int q = 0; q < numQueries; q++)
                {
                    for (int c = 0; c < numCached; c++)
                    {
                        float sum = 0.0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            float qVal = GetTensorValue(query, b, q, h, d);
                            float kVal = GetTensorValue(keys, b, c, h, d);
                            sum += qVal * kVal;
                        }
                        int outIdx = b * (numQueries * numHeads * numCached) +
                                     q * (numHeads * numCached) +
                                     h * numCached +
                                     c;
                        outputData[outIdx] = sum;
                    }
                }
            }
        }

        return new Tensor(outputData, new[] { batchSize, numQueries, numHeads, numCached });
    }

    private Tensor MatmulWeightValue(Tensor weights, Tensor values)
    {
        var batchSize = weights.Shape[0];
        var numQueries = weights.Shape[1];
        var numHeads = weights.Shape[2];
        var numCached = weights.Shape[3];
        var headDim = values.Shape[3];

        var outputSize = batchSize * numQueries * numHeads * headDim;
        var outputData = new float[outputSize];

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int q = 0; q < numQueries; q++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        float sum = 0.0f;
                        for (int c = 0; c < numCached; c++)
                        {
                            float wVal = GetTensorValue(weights, b, q, h, c);
                            float vVal = GetTensorValue(values, b, c, h, d);
                            sum += wVal * vVal;
                        }
                        int outIdx = b * (numQueries * numHeads * headDim) +
                                     q * (numHeads * headDim) +
                                     h * headDim +
                                     d;
                        outputData[outIdx] = sum;
                    }
                }
            }
        }

        return new Tensor(outputData, new[] { batchSize, numQueries, numHeads, headDim });
    }

    private Tensor Softmax(Tensor input, int dim)
    {
        var inputData = input.Data;
        var outputData = new float[inputData.Length];
        int[] shape = input.Shape;

        int outerDim = shape[0] * shape[1] * shape[2];
        int innerDim = shape[3];

        for (int i = 0; i < outerDim; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < innerDim; j++)
            {
                int idx = i * innerDim + j;
                if (inputData[idx] > max)
                    max = inputData[idx];
            }

            float sum = 0.0f;
            for (int j = 0; j < innerDim; j++)
            {
                int idx = i * innerDim + j;
                outputData[idx] = MathF.Exp(inputData[idx] - max);
                sum += outputData[idx];
            }

            for (int j = 0; j < innerDim; j++)
            {
                int idx = i * innerDim + j;
                outputData[idx] /= sum;
            }
        }

        return new Tensor(outputData, shape);
    }

    private Tensor ScaleTensor(Tensor tensor, float scale)
    {
        var inputData = tensor.Data;
        var outputData = new float[inputData.Length];

        for (int i = 0; i < inputData.Length; i++)
        {
            outputData[i] = inputData[i] * scale;
        }

        return new Tensor(outputData, tensor.Shape);
    }

    private float GetTensorValue(Tensor tensor, int b, int q, int h, int d)
    {
        int[] shape = tensor.Shape;
        int idx = b * (shape[1] * shape[2] * shape[3]) +
                  q * (shape[2] * shape[3]) +
                  h * shape[3] +
                  d;
        return tensor.Data[idx];
    }

    private Tensor CreateRandomTensor(int[] shape)
    {
        var data = new float[Product(shape)];
        var random = new Random(42);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)random.NextDouble();
        }
        return new Tensor(data, shape);
    }

    private static int Product(int[] array)
    {
        int product = 1;
        foreach (int value in array)
        {
            product *= value;
        }
        return product;
    }
}

[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class BlockManagerBenchmarks
{
    private KVCacheBlockManager _blockManager = null!;
    private DeviceId _deviceId = DeviceId.CPU;

    [Params(100, 500, 1000, 5000)]
    public int TotalBlocks { get; set; }

    [Params(16, 32, 64)]
    public int SequenceLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _deviceId = DeviceId.CPU;
        _blockManager = new KVCacheBlockManager(
            totalBlocks: TotalBlocks,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            deviceId: _deviceId
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
    private DeviceId _deviceId = DeviceId.CPU;

    [Params(100, 1000, 10000)]
    public int SequenceLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _deviceId = DeviceId.CPU;
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 10000,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            deviceId: _deviceId
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
    private DeviceId _deviceId = DeviceId.CPU;

    [Params(10, 50, 100)]
    public int NumSequences { get; set; }

    [Params(64, 128, 256)]
    public int AvgSequenceLength { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _deviceId = DeviceId.CPU;
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 5000,
            blockSize: 16,
            headDim: 128,
            numLayers: 32,
            numAttentionHeads: 32,
            deviceId: _deviceId
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
        var data = new float[Product(shape)];
        var random = new Random(42);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)random.NextDouble();
        }
        return new Tensor(data, shape);
    }

    private static int Product(int[] array)
    {
        int product = 1;
        foreach (int value in array)
        {
            product *= value;
        }
        return product;
    }
}
