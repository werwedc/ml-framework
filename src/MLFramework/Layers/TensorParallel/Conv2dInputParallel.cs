using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.Modules;
using MLFramework.Distributed;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Input-channel parallel 2D convolutional layer.
    /// Splits the input channels across multiple devices.
    /// </summary>
    public class Conv2dInputParallel : IModule
    {
        private readonly int _inChannels;
        private readonly int _outChannels;
        private readonly int _worldSize;
        private readonly int _rank;
        private readonly TensorParallelGroup? _processGroup;

        // Sharded weight: [out_channels, in_channels / world_size, kernel_h, kernel_w]
        private Tensor _weight;
        private Tensor? _bias; // Bias: [out_channels] (not sharded)

        public Tensor Weight => _weight;
        public Tensor? Bias => _bias;

        public int InChannels => _inChannels;
        public int OutChannels => _outChannels;
        public int KernelSize { get; }
        public int Stride { get; }
        public int Padding { get; }
        public int Dilation { get; }
        public int Groups { get; }

        public string ModuleType => "Conv2dInputParallel";

        public bool IsTraining { get; set; } = false;

        /// <summary>
        /// Creates a new input-channel parallel Conv2d layer
        /// </summary>
        public Conv2dInputParallel(
            int inChannels,
            int outChannels,
            int kernelSize,
            TensorParallelGroup? processGroup = null,
            int stride = 1,
            int padding = 0,
            int dilation = 1,
            int groups = 1,
            bool bias = true)
        {
            if (inChannels <= 0)
                throw new ArgumentException("inChannels must be positive", nameof(inChannels));
            if (outChannels <= 0)
                throw new ArgumentException("outChannels must be positive", nameof(outChannels));
            if (kernelSize <= 0)
                throw new ArgumentException("kernelSize must be positive", nameof(kernelSize));

            _inChannels = inChannels;
            _outChannels = outChannels;
            _worldSize = TensorParallel.GetWorldSize();
            _rank = TensorParallel.GetRank();
            _processGroup = processGroup;

            KernelSize = kernelSize;
            Stride = stride;
            Padding = padding;
            Dilation = dilation;
            Groups = groups;

            // Validate dimensions
            if (inChannels % _worldSize != 0)
            {
                throw new ArgumentException(
                    $"inChannels ({inChannels}) must be divisible by worldSize ({_worldSize})");
            }

            // Calculate sharded dimensions
            int shardInChannels = inChannels / _worldSize;
            int kernelH = kernelSize;
            int kernelW = kernelSize;

            // Initialize sharded weight
            // Shape: [out_channels, in_channels / world_size, kernel_h, kernel_w]
            _weight = InitializeWeight(outChannels, shardInChannels, kernelH, kernelW);

            if (bias)
            {
                // Bias is NOT sharded: [out_channels]
                _bias = Tensor.Zeros(new[] { outChannels });
            }
        }

        /// <summary>
        /// Gets all parameters in the module
        /// </summary>
        public IEnumerable<Tensor> Parameters
        {
            get
            {
                yield return _weight;
                if (_bias != null)
                {
                    yield return _bias;
                }
            }
        }

        /// <summary>
        /// Forward pass for input-channel parallel Conv2d
        /// Input:  [batch, in_channels / world_size, h, w] (sharded)
        /// Weight: [out_channels, in_channels / world_size, kernel_h, kernel_w]
        /// Output: [batch, out_channels, h_out, w_out] (after all-reduce)
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            // Verify input is sharded
            int expectedInChannels = _inChannels / _worldSize;
            if (input.Shape[1] != expectedInChannels)
            {
                throw new InvalidOperationException(
                    $"Expected sharded input with {expectedInChannels} channels, got {input.Shape[1]}");
            }

            // Perform local convolution
            // input: [batch, in_shard, h, w]
            // weight: [out_channels, in_shard, kernel_h, kernel_w]
            // output_local: [batch, out_channels, h_out, w_out]
            var outputLocal = Conv2dHelper(input, _weight, Stride, Padding, Dilation);

            // All-reduce to sum results from all ranks
            var comm = _processGroup?.Communicator ?? TensorParallel.GetCommunicator();
            comm.AllReduceAsync(outputLocal, ReduceOp.Sum).Wait();

            // Add bias if present (after all-reduce)
            if (_bias != null)
            {
                outputLocal = AddBiasToOutput(outputLocal, _bias);
            }

            return outputLocal;
        }

        /// <summary>
        /// Applies a function to all parameters
        /// </summary>
        public void ApplyToParameters(Action<Tensor> action)
        {
            action(_weight);
            if (_bias != null)
            {
                action(_bias);
            }
        }

        /// <summary>
        /// Sets the requires_grad flag for all parameters
        /// </summary>
        public void SetRequiresGrad(bool requiresGrad)
        {
            _weight.RequiresGrad = requiresGrad;
            if (_bias != null)
            {
                _bias.RequiresGrad = requiresGrad;
            }
        }

        private Tensor InitializeWeight(int outFeat, int inFeat, int kernelH, int kernelW)
        {
            // Use Kaiming/He initialization for ReLU-friendly networks
            double std = Math.Sqrt(2.0 / (inFeat * kernelH * kernelW));
            var tensor = TensorMathExtensions.RandomNormal(new[] { outFeat, inFeat, kernelH, kernelW }, mean: 0.0, std: std, seed: 42);
            tensor.RequiresGrad = true;
            return tensor;
        }

        /// <summary>
        /// Helper method to perform 2D convolution (simplified version of Conv2d.Forward)
        /// </summary>
        private Tensor Conv2dHelper(Tensor input, Tensor weight, int stride, int padding, int dilation)
        {
            // Input shape: [batch_size, in_channels, height, width]
            // Weight shape: [out_channels, in_channels, kernel_size, kernel_size]
            // Output shape: [batch_size, out_channels, out_height, out_width]

            int batchSize = input.Shape[0];
            int inChannels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];

            int outChannels = weight.Shape[0];
            int kernelSize = weight.Shape[2];

            int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
            int outWidth = (width + 2 * padding - kernelSize) / stride + 1;

            // Extract patches using im2col
            var patches = Im2Col(input, kernelSize, padding, stride);

            // Reshape weight: [outC, inC, kH, kW] -> [outC, inC*kH*kW]
            var weightFlat = weight.Reshape(new[] { outChannels, inChannels * kernelSize * kernelSize });

            // Matrix multiplication: patches @ weightFlat.T
            var weightFlatTransposed = weightFlat.Transpose();
            var outputFlat = MatMul(patches, weightFlatTransposed);

            // Reshape output: [batch * out_h*out_w, outC] -> [batch, out_h, out_w, outC] -> [batch, outC, out_h, out_w]
            var output = outputFlat.Reshape(new[] { batchSize, outHeight, outWidth, outChannels })
                                  .Reshape(new[] { batchSize, outChannels, outHeight, outWidth });

            return output;
        }

        private Tensor Im2Col(Tensor input, int kernelSize, int padding, int stride)
        {
            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];

            int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
            int outWidth = (width + 2 * padding - kernelSize) / stride + 1;

            int numPositions = outHeight * outWidth;
            int patchSize = channels * kernelSize * kernelSize;

            // Allocate output tensor as 2D: [batch * out_h * out_w, in_c * k * k]
            var output = Tensor.Zeros(new[] { batchSize * numPositions, patchSize });

            // Extract patches directly into 2D format
            for (int b = 0; b < batchSize; b++)
            {
                for (int oy = 0; oy < outHeight; oy++)
                {
                    for (int ox = 0; ox < outWidth; ox++)
                    {
                        int positionIndex = b * numPositions + oy * outWidth + ox;

                        for (int c = 0; c < channels; c++)
                        {
                            for (int ky = 0; ky < kernelSize; ky++)
                            {
                                for (int kx = 0; kx < kernelSize; kx++)
                                {
                                    int iy = oy * stride - padding + ky;
                                    int ix = ox * stride - padding + kx;

                                    int patchIndex = (c * kernelSize + ky) * kernelSize + kx;

                                    if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                                    {
                                        output[new[] { positionIndex, patchIndex }] = input[new[] { b, c, iy, ix }];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
        }

        private Tensor MatMul(Tensor a, Tensor b)
        {
            // Standard matrix multiplication: a @ b
            // a: [m, n]
            // b: [n, p]
            // result: [m, p]

            int m = a.Shape[0];
            int n = a.Shape[1];
            int p = b.Shape[1];

            if (b.Shape[0] != n)
                throw new ArgumentException($"Inner dimensions must match: a.Shape[1]={n}, b.Shape[0]={b.Shape[0]}");

            var resultData = new float[m * p];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += a[new[] { i, k }] * b[new[] { k, j }];
                    }
                    resultData[i * p + j] = sum;
                }
            }

            return new Tensor(resultData, new[] { m, p }, a.RequiresGrad || b.RequiresGrad);
        }

        private Tensor AddBiasToOutput(Tensor input, Tensor bias)
        {
            // input: [batch_size, out_channels, out_height, out_width]
            // bias: [out_channels]
            // Add bias to each channel

            var resultData = new float[input.Size];
            var outChannels = bias.Shape[0];
            int batchSize = input.Shape[0];
            int outHeight = input.Shape[2];
            int outWidth = input.Shape[3];

            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < outChannels; c++)
                {
                    for (int h = 0; h < outHeight; h++)
                    {
                        for (int w = 0; w < outWidth; w++)
                        {
                            resultData[((b * outChannels + c) * outHeight + h) * outWidth + w] =
                                input[new[] { b, c, h, w }] + bias[new[] { c }];
                        }
                    }
                }
            }

            return new Tensor(resultData, input.Shape, input.RequiresGrad || bias.RequiresGrad);
        }
    }
}
