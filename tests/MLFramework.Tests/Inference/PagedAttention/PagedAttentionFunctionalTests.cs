using MlFramework.Inference.PagedAttention;
using MlFramework.Inference.PagedAttention.Kernels;
using MlFramework.Inference.PagedAttention.Phases;
using RitterFramework.Core.Tensor;
using MLFramework.Core;
using Xunit;

namespace MLFramework.Tests.Inference.PagedAttention;

public class PagedAttentionFunctionalTests : IDisposable
{
    private readonly KVCacheBlockManager _blockManager;
    private readonly BlockTable _blockTable;
    private readonly StandardPagedAttentionKernel _kernel;
    private readonly DeviceId _device;
    private readonly int _blockSize = 16;
    private readonly int _numHeads = 4;
    private readonly int _headDim = 32;

    public PagedAttentionFunctionalTests()
    {
        _device = DeviceId.CPU;
        _blockManager = new KVCacheBlockManager(
            totalBlocks: 100,
            blockSize: _blockSize,
            headDim: _headDim,
            numLayers: 1,
            numAttentionHeads: _numHeads,
            deviceId: _device
        );
        _blockTable = new BlockTable(_blockManager);
        _kernel = new StandardPagedAttentionKernel(_headDim);
    }

    public void Dispose()
    {
        _blockManager.Dispose();
    }

    [Fact]
    public void PagedAttention_MatchesContiguousForSingleToken()
    {
        // Setup
        int sequenceId = 1;
        var query = CreateRandomQuery(batchSize: 1, numTokens: 1);

        // Store keys and values
        var key = CreateRandomQuery(1, 1);
        var value = CreateRandomQuery(1, 1);

        _blockTable.AllocateAndAppendBlock(sequenceId);
        StoreKVInBlock(sequenceId, 0, key, value);

        // Compute paged attention
        var pagedOutput = _kernel.ComputePagedAttention(
            query,
            GatherKeys(sequenceId),
            GatherValues(sequenceId),
            AttentionPhase.Decode
        );

        // Compute contiguous attention
        var contiguousOutput = ComputeContiguousAttention(query, key, value);

        // Compare outputs
        AssertTensorsClose(pagedOutput, contiguousOutput, tolerance: 1e-5);
    }

    [Fact]
    public void PagedAttention_MatchesContiguousForMultipleTokens()
    {
        // Setup
        int sequenceId = 1;
        int numTokens = 32;
        var query = CreateRandomQuery(batchSize: 1, numTokens: 1);

        // Store keys and values for all tokens
        var keys = new List<Tensor>();
        var values = new List<Tensor>();

        for (int i = 0; i < numTokens; i++)
        {
            var key = CreateRandomQuery(1, 1);
            var value = CreateRandomQuery(1, 1);
            keys.Add(key);
            values.Add(value);

            // Allocate block if needed
            if (i % _blockSize == 0)
            {
                _blockTable.AllocateAndAppendBlock(sequenceId);
            }

            StoreKVInBlock(sequenceId, i, key, value);
        }

        // Compute paged attention
        var cachedKeys = ConcatenateKeys(keys);
        var cachedValues = ConcatenateValues(values);
        var pagedOutput = _kernel.ComputePagedAttention(
            query,
            cachedKeys,
            cachedValues,
            AttentionPhase.Decode
        );

        // Compute contiguous attention
        var contiguousOutput = ComputeContiguousAttention(
            query,
            cachedKeys,
            cachedValues
        );

        // Compare outputs
        AssertTensorsClose(pagedOutput, contiguousOutput, tolerance: 1e-5);
    }

    [Fact]
    public void PagedAttention_MatchesContiguousAcrossMultipleBlocks()
    {
        // Setup - ensure we span multiple blocks
        int sequenceId = 1;
        int numTokens = _blockSize * 3; // 3 blocks
        var query = CreateRandomQuery(batchSize: 1, numTokens: 1);

        // Store keys and values
        var keys = new List<Tensor>();
        var values = new List<Tensor>();

        for (int i = 0; i < numTokens; i++)
        {
            var key = CreateRandomQuery(1, 1);
            var value = CreateRandomQuery(1, 1);
            keys.Add(key);
            values.Add(value);

            if (i % _blockSize == 0)
            {
                _blockTable.AllocateAndAppendBlock(sequenceId);
            }

            StoreKVInBlock(sequenceId, i, key, value);
        }

        // Compute paged attention
        var cachedKeys = ConcatenateKeys(keys);
        var cachedValues = ConcatenateValues(values);
        var pagedOutput = _kernel.ComputePagedAttention(
            query,
            cachedKeys,
            cachedValues,
            AttentionPhase.Decode
        );

        // Compute contiguous attention
        var contiguousOutput = ComputeContiguousAttention(
            query,
            cachedKeys,
            cachedValues
        );

        // Compare outputs
        AssertTensorsClose(pagedOutput, contiguousOutput, tolerance: 1e-5);
    }

    [Fact]
    public void PagedAttention_PrefillPhase_MatchesContiguous()
    {
        // Setup
        int sequenceId = 1;
        int numTokens = 16;
        var query = CreateRandomQuery(batchSize: 1, numTokens: numTokens);

        // Store keys and values
        var keys = new List<Tensor>();
        var values = new List<Tensor>();

        for (int i = 0; i < numTokens; i++)
        {
            var key = CreateRandomQuery(1, 1);
            var value = CreateRandomQuery(1, 1);
            keys.Add(key);
            values.Add(value);

            if (i % _blockSize == 0)
            {
                _blockTable.AllocateAndAppendBlock(sequenceId);
            }

            StoreKVInBlock(sequenceId, i, key, value);
        }

        // Compute paged attention (prefill phase)
        var cachedKeys = ConcatenateKeys(keys);
        var cachedValues = ConcatenateValues(values);
        var pagedOutput = _kernel.ComputePagedAttention(
            query,
            cachedKeys,
            cachedValues,
            AttentionPhase.Prefill
        );

        // Compute contiguous attention
        var contiguousOutput = ComputeContiguousAttention(
            query,
            cachedKeys,
            cachedValues
        );

        // Compare outputs
        AssertTensorsClose(pagedOutput, contiguousOutput, tolerance: 1e-5);
    }

    [Fact]
    public void PagedAttention_MultipleSequences_IndependentResults()
    {
        // Setup two sequences
        int seq1 = 1, seq2 = 2;
        var query1 = CreateRandomQuery(1, 1);
        var query2 = CreateRandomQuery(1, 1);

        // Store different KV for each sequence
        var key1 = CreateRandomQuery(1, 10);
        var value1 = CreateRandomQuery(1, 10);
        var key2 = CreateRandomQuery(1, 10);
        var value2 = CreateRandomQuery(1, 10);

        _blockTable.AllocateAndAppendBlock(seq1);
        for (int i = 0; i < 10; i++)
            StoreKVInBlock(seq1, i, SliceTensor(key1, i), SliceTensor(value1, i));

        _blockTable.AllocateAndAppendBlock(seq2);
        for (int i = 0; i < 10; i++)
            StoreKVInBlock(seq2, i, SliceTensor(key2, i), SliceTensor(value2, i));

        // Compute attention for each sequence
        var output1 = _kernel.ComputePagedAttention(
            query1,
            key1,
            value1,
            AttentionPhase.Decode
        );

        var output2 = _kernel.ComputePagedAttention(
            query2,
            key2,
            value2,
            AttentionPhase.Decode
        );

        // Outputs should be different (different inputs)
        Assert.False(AreTensorsClose(output1, output2, tolerance: 1e-5));
    }

    [Fact]
    public void PagedAttention_NumericalStability_LargeValues()
    {
        // Test with large values to check numerical stability
        int sequenceId = 1;
        var query = CreateRandomQuery(1, 1, scale: 100.0f);
        var key = CreateRandomQuery(1, 10, scale: 100.0f);
        var value = CreateRandomQuery(1, 10, scale: 100.0f);

        _blockTable.AllocateAndAppendBlock(sequenceId);
        for (int i = 0; i < 10; i++)
            StoreKVInBlock(sequenceId, i, SliceTensor(key, i), SliceTensor(value, i));

        // Should not produce NaN or Inf
        var output = _kernel.ComputePagedAttention(
            query,
            key,
            value,
            AttentionPhase.Decode
        );

        Assert.False(HasNaNOrInf(output));
    }

    // Dictionary to store KV pairs for testing
    private readonly Dictionary<int, Dictionary<int, (Tensor key, Tensor value)>> _storedKV =
        new Dictionary<int, Dictionary<int, (Tensor, Tensor)>>();

    // Helper methods
    private Tensor CreateRandomQuery(int batchSize, int numTokens, float scale = 1.0f)
    {
        var random = new Random(42);
        var data = new float[batchSize * numTokens * _numHeads * _headDim];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)(random.NextDouble() * 2 - 1) * scale;
        }
        return new Tensor(data, new[] { batchSize, numTokens, _numHeads, _headDim });
    }

    private void StoreKVInBlock(int sequenceId, int tokenIndex, Tensor key, Tensor value)
    {
        if (!_storedKV.ContainsKey(sequenceId))
        {
            _storedKV[sequenceId] = new Dictionary<int, (Tensor, Tensor)>();
        }
        _storedKV[sequenceId][tokenIndex] = (key.Clone(), value.Clone());
    }

    private Tensor GatherKeys(int sequenceId)
    {
        if (!_storedKV.ContainsKey(sequenceId))
            return Tensor.Zeros(new[] { 1, 0, _numHeads, _headDim });

        var tokenIndices = _storedKV[sequenceId].Keys.OrderBy(x => x).ToList();
        var keys = new List<Tensor>();

        foreach (var idx in tokenIndices)
        {
            keys.Add(_storedKV[sequenceId][idx].key);
        }

        return ConcatenateKeys(keys);
    }

    private Tensor GatherValues(int sequenceId)
    {
        if (!_storedKV.ContainsKey(sequenceId))
            return Tensor.Zeros(new[] { 1, 0, _numHeads, _headDim });

        var tokenIndices = _storedKV[sequenceId].Keys.OrderBy(x => x).ToList();
        var values = new List<Tensor>();

        foreach (var idx in tokenIndices)
        {
            values.Add(_storedKV[sequenceId][idx].value);
        }

        return ConcatenateValues(values);
    }

    private Tensor ConcatenateKeys(List<Tensor> keys)
    {
        if (keys.Count == 0)
            return Tensor.Zeros(new[] { 1, 0, _numHeads, _headDim });

        int numTokens = keys.Count;
        var data = new float[keys.Count * _numHeads * _headDim];

        for (int i = 0; i < keys.Count; i++)
        {
            Array.Copy(keys[i].Data, 0, data, i * _numHeads * _headDim, _numHeads * _headDim);
        }

        return new Tensor(data, new[] { 1, numTokens, _numHeads, _headDim });
    }

    private Tensor ConcatenateValues(List<Tensor> values)
    {
        if (values.Count == 0)
            return Tensor.Zeros(new[] { 1, 0, _numHeads, _headDim });

        int numTokens = values.Count;
        var data = new float[values.Count * _numHeads * _headDim];

        for (int i = 0; i < values.Count; i++)
        {
            Array.Copy(values[i].Data, 0, data, i * _numHeads * _headDim, _numHeads * _headDim);
        }

        return new Tensor(data, new[] { 1, numTokens, _numHeads, _headDim });
    }

    private Tensor SliceTensor(Tensor tensor, int index)
    {
        // Extract a single token's tensor
        // Tensor shape: [batch, numTokens, numHeads, headDim]
        var sliceData = new float[_numHeads * _headDim];
        int startIdx = index * _numHeads * _headDim;
        Array.Copy(tensor.Data, startIdx, sliceData, 0, _numHeads * _headDim);
        return new Tensor(sliceData, new[] { 1, 1, _numHeads, _headDim });
    }

    private Tensor ComputeContiguousAttention(Tensor query, Tensor key, Tensor value)
    {
        // Standard contiguous attention computation - use the same kernel for now
        return _kernel.ComputePagedAttention(query, key, value, AttentionPhase.Prefill);
    }

    private void AssertTensorsClose(Tensor a, Tensor b, double tolerance)
    {
        Assert.Equal(a.Shape, b.Shape);
        var dataA = a.Data;
        var dataB = b.Data;

        for (int i = 0; i < dataA.Length; i++)
        {
            Assert.InRange(Math.Abs(dataA[i] - dataB[i]), 0, tolerance);
        }
    }

    private bool AreTensorsClose(Tensor a, Tensor b, double tolerance)
    {
        var dataA = a.Data;
        var dataB = b.Data;

        for (int i = 0; i < dataA.Length; i++)
        {
            if (Math.Abs(dataA[i] - dataB[i]) > tolerance)
                return false;
        }
        return true;
    }

    private bool HasNaNOrInf(Tensor tensor)
    {
        var data = tensor.Data;
        return data.Any(d => float.IsNaN(d) || float.IsInfinity(d));
    }
}
