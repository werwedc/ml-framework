using RitterFramework.Core.Tensor;
using MLFramework.Modules;

namespace MLFramework.LoRA
{
    /// <summary>
    /// LoRA adapter for Conv2d layers
    /// Wraps a standard Conv2d layer and injects low-rank adapter matrices for vision models
    /// </summary>
    public class LoRAConv2d : LoRAAdapterBase, IModule
    {
        private readonly Conv2d _convLayer;
        private Tensor _loraA = null!; // Rank x (InChannels * KernelSize * KernelSize)
        private Tensor _loraB = null!; // OutChannels x Rank
        private readonly float _dropoutRate;
        private readonly bool _useBias;
        private readonly Tensor? _loraBias;
        private readonly Random? _dropoutRandom;

        public int InChannels => _convLayer.InChannels;
        public int OutChannels => _convLayer.OutChannels;
        public int KernelSize => _convLayer.KernelSize;

        public string ModuleType => "LoRAConv2d";

        public bool IsTraining { get; set; } = false;

        /// <summary>
        /// Creates a new LoRA adapter for a Conv2d layer
        /// </summary>
        public LoRAConv2d(Conv2d convLayer, int rank, float alpha,
                          LoRAInitializationStrategy initialization = LoRAInitializationStrategy.Standard,
                          float dropout = 0.0f, bool useBias = false)
            : base(convLayer, rank, alpha)
        {
            _convLayer = convLayer ?? throw new ArgumentNullException(nameof(convLayer));
            _dropoutRate = dropout;
            _useBias = useBias;

            // Initialize LoRA matrices
            InitializeLoRAMatrices(initialization);

            if (_useBias)
            {
                _loraBias = Tensor.Zeros(new[] { OutChannels });
            }

            if (_dropoutRate > 0.0f)
            {
                _dropoutRandom = new Random(42);
            }
        }

        private void InitializeLoRAMatrices(LoRAInitializationStrategy strategy)
        {
            int inChannels = _convLayer.InChannels;
            int outChannels = _convLayer.OutChannels;
            int kernelSize = _convLayer.KernelSize;

            // Flatten spatial dimensions for LoRA
            // For Conv2d: [Out, In, K, K] -> [Out, In*K*K]
            // LoRA operates on [Out, In*K*K] dimension
            int flattenedInDim = inChannels * kernelSize * kernelSize;

            switch (strategy)
            {
                case LoRAInitializationStrategy.Standard:
                    // A: Kaiming normal, B: Zeros
                    _loraA = KaimingNormal(new[] { Rank, flattenedInDim });
                    _loraB = Tensor.Zeros(new[] { outChannels, Rank });
                    break;

                case LoRAInitializationStrategy.Xavier:
                    // Both: Xavier uniform
                    _loraA = XavierUniform(new[] { Rank, flattenedInDim });
                    _loraB = XavierUniform(new[] { outChannels, Rank });
                    break;

                case LoRAInitializationStrategy.Zero:
                    // Both: Zeros
                    _loraA = Tensor.Zeros(new[] { Rank, flattenedInDim });
                    _loraB = Tensor.Zeros(new[] { outChannels, Rank });
                    break;

                default:
                    throw new ArgumentException($"Unknown initialization strategy: {strategy}");
            }
        }

        /// <summary>
        /// Forward pass through the LoRA-adapted Conv2d layer
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            // Standard forward pass through base layer
            var output = _convLayer.Forward(input);

            if (!IsEnabled)
                return output;

            // LoRA forward pass for Conv2d
            // Need to flatten spatial dims for LoRA computation
            int batchSize = input.Shape[0];
            int height = input.Shape[2];
            int width = input.Shape[3];
            int kernelSize = _convLayer.KernelSize;
            int padding = _convLayer.Padding;
            int stride = _convLayer.Stride;

            // Calculate output dimensions
            int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
            int outWidth = (width + 2 * padding - kernelSize) / stride + 1;

            // Extract patches using im2col - returns [batch * out_h * out_w, in_c * k * k]
            var patches = Im2Col(input, kernelSize, padding, stride);

            // Compute LoRA: W_patch + (alpha/r) * B * A * patch
            // patches: [batch * out_h * out_w, in_c * k * k]
            // _loraA: [rank, in_c * k * k]
            // _loraB: [out_c, rank]

            // Apply LoRA: A * patch [rank, in*k*k] x [batch*positions, in*k*k]^T
            var loraInput = MatMul(patches, _loraA.Transpose());

            // Apply dropout if enabled
            if (_dropoutRate > 0.0f && IsTraining)
            {
                loraInput = ApplyDropout(loraInput);
            }

            // Apply B: [out, rank] x [batch*positions, rank]^T
            var loraOutput = MatMul(loraInput, _loraB.Transpose());

            // Scale by alpha/r
            loraOutput = loraOutput * ScalingFactor;

            // Reshape and add to output
            // loraOutput: [batch * out_h * out_w, out_c]
            // Need: [batch, out_c, out_h, out_w]
            loraOutput = loraOutput.Reshape(new[] { batchSize, outHeight, outWidth, OutChannels })
                                  .Reshape(new[] { batchSize, OutChannels, outHeight, outWidth });

            // Add bias if present
            if (_loraBias != null)
            {
                output = output + loraOutput;
                output = AddBias(output, _loraBias);
            }
            else
            {
                output = output + loraOutput;
            }

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

        private Tensor ApplyDropout(Tensor tensor)
        {
            var mask = RandomUniform(tensor.Shape, _dropoutRandom!);
            mask = Where(mask.GreaterThan(_dropoutRate), 1.0f / (1.0f - _dropoutRate), 0.0f);
            return tensor.Mul(mask);
        }

        public override void FreezeBaseLayer()
        {
            _convLayer.Weight.RequiresGrad = false;
            if (_convLayer.Bias != null)
            {
                _convLayer.Bias.RequiresGrad = false;
            }
            _isBaseLayerFrozen = true;
        }

        public override void UnfreezeBaseLayer()
        {
            _convLayer.Weight.RequiresGrad = true;
            if (_convLayer.Bias != null)
            {
                _convLayer.Bias.RequiresGrad = true;
            }
            _isBaseLayerFrozen = false;
        }

        public override IEnumerable<Tensor> TrainableParameters
        {
            get
            {
                if (!_isBaseLayerFrozen)
                {
                    yield return _convLayer.Weight;
                    if (_convLayer.Bias != null)
                        yield return _convLayer.Bias;
                }
                // Adapter parameters are always trainable
                yield return _loraA;
                yield return _loraB;
                if (_loraBias != null)
                    yield return _loraBias;
            }
        }

        public override IEnumerable<Tensor> FrozenParameters
        {
            get
            {
                if (_isBaseLayerFrozen)
                {
                    yield return _convLayer.Weight;
                    if (_convLayer.Bias != null)
                        yield return _convLayer.Bias;
                }
            }
        }

        public override void MergeAdapter()
        {
            // Backup original weights
            _baseLayerWeightsBackup = _convLayer.Weight.Clone();

            // Flatten and reshape for LoRA merge
            // Base weight: [out, in, k, k] -> [out, in*k*k]
            int outChannels = _convLayer.OutChannels;
            int kernelSize = _convLayer.KernelSize;
            int flattenedInDim = _convLayer.InChannels * kernelSize * kernelSize;

            var weightFlat = _convLayer.Weight.Reshape(new[] { outChannels, flattenedInDim });

            // W_new = W + (alpha/r) * B * A
            var deltaW = MatMul(_loraB, _loraA);
            deltaW = deltaW * ScalingFactor;

            var newWeight = weightFlat + deltaW;
            _convLayer.Weight = newWeight.Reshape(_convLayer.Weight.Shape);
        }

        public override void ResetBaseLayer()
        {
            if (_baseLayerWeightsBackup == null)
                throw new InvalidOperationException("No backup available. Cannot reset.");

            _convLayer.Weight = _baseLayerWeightsBackup;
            _baseLayerWeightsBackup = null;
        }

        public override (Tensor? MatrixA, Tensor? MatrixB) GetAdapterWeights()
        {
            return (_loraA, _loraB);
        }

        public override void SetAdapterWeights(Tensor? matrixA, Tensor? matrixB)
        {
            if (matrixA == null || matrixB == null)
                throw new ArgumentNullException("Adapter weights cannot be null");

            int flattenedInDim = _convLayer.InChannels * _convLayer.KernelSize * _convLayer.KernelSize;

            // Validate shapes
            if (matrixA.Shape.Length != 2 || matrixA.Shape[0] != Rank || matrixA.Shape[1] != flattenedInDim)
                throw new ArgumentException($"Matrix A shape must be [{Rank}, {flattenedInDim}]");

            if (matrixB.Shape.Length != 2 || matrixB.Shape[0] != OutChannels || matrixB.Shape[1] != Rank)
                throw new ArgumentException($"Matrix B shape must be [{OutChannels}, {Rank}]");

            _loraA.CopyFrom(matrixA);
            _loraB.CopyFrom(matrixB);
        }

        public Tensor? GetBias()
        {
            return _loraBias;
        }

        public void SetBias(Tensor? bias)
        {
            if (bias != null)
            {
                if (bias.Shape.Length != 1 || bias.Shape[0] != OutChannels)
                    throw new ArgumentException($"Bias shape must be [{OutChannels}]");
                _loraBias?.CopyFrom(bias);
            }
        }

        // IModule implementation
        public IEnumerable<Tensor> Parameters => TrainableParameters;

        public void ApplyToParameters(Action<Tensor> action)
        {
            foreach (var param in Parameters)
            {
                action(param);
            }
        }

        public void SetRequiresGrad(bool requiresGrad)
        {
            foreach (var param in Parameters)
            {
                param.RequiresGrad = requiresGrad;
            }
        }

        // Helper methods for tensor operations
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

        private Tensor KaimingNormal(int[] shape)
        {
            // Kaiming (He) normal initialization
            // std = sqrt(2 / fan_in)
            int fanIn = shape[1];
            float std = (float)Math.Sqrt(2.0 / fanIn);

            var random = new Random(42);
            var data = new float[shape[0] * shape[1]];

            // Box-Muller transform for normal distribution
            for (int i = 0; i < data.Length; i += 2)
            {
                float u1 = (float)random.NextDouble();
                float u2 = (float)random.NextDouble();

                float z0 = (float)(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                float z1 = (float)(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));

                data[i] = z0;
                if (i + 1 < data.Length)
                {
                    data[i + 1] = z1;
                }
            }

            return new Tensor(data, shape, requiresGrad: true);
        }

        private Tensor XavierUniform(int[] shape)
        {
            // Xavier/Glorot uniform initialization
            // limit = sqrt(6 / (fan_in + fan_out))
            int fanIn = shape[1];
            int fanOut = shape[0];
            float limit = (float)Math.Sqrt(6.0 / (fanIn + fanOut));

            var random = new Random(42);
            var data = new float[shape[0] * shape[1]];

            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)(random.NextDouble() * 2 * limit - limit);
            }

            return new Tensor(data, shape, requiresGrad: true);
        }

        private Tensor RandomUniform(int[] shape, Random random)
        {
            var data = new float[shape.Aggregate(1, (x, y) => x * y)];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)random.NextDouble();
            }
            return new Tensor(data, shape);
        }

        private Tensor AddBias(Tensor input, Tensor bias)
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

        private Tensor Where(Tensor condition, float trueValue, float falseValue)
        {
            var data = new float[condition.Size];
            for (int i = 0; i < data.Length; i++)
            {
                int[] indices = GetIndices(condition.Shape, i);
                data[i] = condition[indices] > 0 ? trueValue : falseValue;
            }
            return new Tensor(data, condition.Shape);
        }

        private int[] GetIndices(int[] shape, int flatIndex)
        {
            var indices = new int[shape.Length];
            var strides = new int[shape.Length];
            int stride = 1;

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= shape[i];
            }

            int remaining = flatIndex;
            for (int i = 0; i < shape.Length; i++)
            {
                indices[i] = remaining / strides[i];
                remaining %= strides[i];
            }

            return indices;
        }
    }
}
