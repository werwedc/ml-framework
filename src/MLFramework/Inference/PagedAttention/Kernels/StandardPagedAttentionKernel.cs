using RitterFramework.Core.Tensor;
using MLFramework.Core;
using MlFramework.Inference.PagedAttention.Phases;

namespace MlFramework.Inference.PagedAttention.Kernels;

/// <summary>
/// Reference implementation of paged attention using standard tensor operations.
/// This implementation is not optimized but provides correct results for testing.
/// For production, use optimized CUDA/HIP implementations.
/// </summary>
public class StandardPagedAttentionKernel : IPagedAttentionKernel
{
    private readonly float _scale;

    public StandardPagedAttentionKernel(int headDim)
    {
        _scale = 1.0f / MathF.Sqrt(headDim);
    }

    public Tensor ComputePagedAttention(
        Tensor query,
        Tensor cachedKeys,
        Tensor cachedValues,
        AttentionPhase phase,
        float scale = -1.0f)
    {
        var actualScale = scale > 0 ? scale : _scale;

        // Input shapes:
        // query:      [batch, numQueries, numHeads, headDim]
        // cachedKeys: [batch, numCached, numHeads, headDim]
        // cachedValues: [batch, numCached, numHeads, headDim]

        int batchSize = query.Shape[0];
        int numQueries = query.Shape[1];
        int numHeads = query.Shape[2];
        int headDim = query.Shape[3];
        int numCached = cachedKeys.Shape[1];

        // Compute attention scores: Q * K^T
        // Result shape: [batch, numQueries, numHeads, numCached]
        var attentionScores = MatmulQueryKey(
            query,
            cachedKeys,
            batchSize,
            numQueries,
            numHeads,
            headDim,
            numCached
        );

        // Scale scores
        attentionScores = ScaleTensor(attentionScores, actualScale);

        // Create causal mask for decode phase
        if (phase == AttentionPhase.Decode && numCached > numQueries)
        {
            attentionScores = ApplyCausalMask(
                attentionScores,
                batchSize,
                numQueries,
                numHeads,
                numCached
            );
        }

        // Compute attention weights using softmax
        var attentionWeights = Softmax(attentionScores, dim: -1);

        // Compute output: weights * V
        // Result shape: [batch, numQueries, numHeads, headDim]
        var output = MatmulWeightValue(
            attentionWeights,
            cachedValues,
            batchSize,
            numQueries,
            numHeads,
            numCached,
            headDim
        );

        return output;
    }

    private Tensor MatmulQueryKey(
        Tensor query,
        Tensor keys,
        int batchSize,
        int numQueries,
        int numHeads,
        int headDim,
        int numCached)
    {
        // Query: [batch, numQueries, numHeads, headDim]
        // Keys:  [batch, numCached, numHeads, headDim]
        // Output: [batch, numQueries, numHeads, numCached]

        // Simple implementation: loop over batches and heads
        int outputSize = batchSize * numQueries * numHeads * numCached;
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

    private Tensor MatmulWeightValue(
        Tensor weights,
        Tensor values,
        int batchSize,
        int numQueries,
        int numHeads,
        int numCached,
        int headDim)
    {
        // Weights: [batch, numQueries, numHeads, numCached]
        // Values:  [batch, numCached, numHeads, headDim]
        // Output: [batch, numQueries, numHeads, headDim]

        int outputSize = batchSize * numQueries * numHeads * headDim;
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

        // Softmax along the last dimension (dim = -1)
        // Shape: [batch, numQueries, numHeads, numCached]
        // Apply softmax along numCached dimension

        int outerDim = shape[0] * shape[1] * shape[2];  // batch * numQueries * numHeads
        int innerDim = shape[3];  // numCached

        for (int i = 0; i < outerDim; i++)
        {
            // Find max for numerical stability
            float max = float.NegativeInfinity;
            for (int j = 0; j < innerDim; j++)
            {
                int idx = i * innerDim + j;
                if (inputData[idx] > max)
                    max = inputData[idx];
            }

            // Exponentiate and sum
            float sum = 0.0f;
            for (int j = 0; j < innerDim; j++)
            {
                int idx = i * innerDim + j;
                outputData[idx] = MathF.Exp(inputData[idx] - max);
                sum += outputData[idx];
            }

            // Normalize
            for (int j = 0; j < innerDim; j++)
            {
                int idx = i * innerDim + j;
                outputData[idx] /= sum;
            }
        }

        return new Tensor(outputData, shape);
    }

    private Tensor ApplyCausalMask(
        Tensor scores,
        int batchSize,
        int numQueries,
        int numHeads,
        int numCached)
    {
        var inputData = scores.Data;
        var outputData = new float[inputData.Length];

        // Copy data
        Array.Copy(inputData, outputData, inputData.Length);

        // Apply causal mask: mask out future tokens
        // For decode phase, we only attend to past tokens
        int startPos = numCached - numQueries;

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int q = 0; q < numQueries; q++)
                {
                    for (int c = startPos + q + 1; c < numCached; c++)
                    {
                        int idx = b * (numQueries * numHeads * numCached) +
                                  q * (numHeads * numCached) +
                                  h * numCached +
                                  c;
                        outputData[idx] = float.NegativeInfinity;
                    }
                }
            }
        }

        return new Tensor(outputData, scores.Shape);
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

    public bool SupportsDevice(Device device)
    {
        // Standard kernel works on CPU and GPU (though not optimized)
        return true;
    }
}
