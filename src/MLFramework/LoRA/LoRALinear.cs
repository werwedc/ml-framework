using RitterFramework.Core.Tensor;
using MLFramework.Modules;

namespace MLFramework.LoRA
{
    /// <summary>
    /// LoRA adapter for Linear layers
    /// Wraps a standard Linear layer and injects low-rank adapter matrices
    /// </summary>
    public class LoRALinear : LoRAAdapterBase, IModule
    {
        private readonly Linear _linearLayer;
        private Tensor _loraA = null!; // Rank x InDim
        private Tensor _loraB = null!; // OutDim x Rank
        private readonly float _dropoutRate;
        private readonly bool _useBias;
        private readonly Tensor? _loraBias; // Optional bias adapter
        private readonly Random? _dropoutRandom;

        public int InDim => _linearLayer.InFeatures;
        public int OutDim => _linearLayer.OutFeatures;

        public string ModuleType => "LoRALinear";

        public bool IsTraining { get; set; } = false;

        /// <summary>
        /// Creates a new LoRA adapter for a Linear layer
        /// </summary>
        public LoRALinear(Linear linearLayer, int rank, float alpha,
                          LoRAInitializationStrategy initialization = LoRAInitializationStrategy.Standard,
                          float dropout = 0.0f, bool useBias = false)
            : base(linearLayer, rank, alpha)
        {
            _linearLayer = linearLayer ?? throw new ArgumentNullException(nameof(linearLayer));
            _dropoutRate = dropout;
            _useBias = useBias;

            // Initialize LoRA matrices
            InitializeLoRAMatrices(initialization);

            if (_useBias)
            {
                _loraBias = Tensor.Zeros(new[] { OutDim });
            }

            if (_dropoutRate > 0.0f)
            {
                _dropoutRandom = new Random(42); // Fixed seed for reproducibility
            }
        }

        private void InitializeLoRAMatrices(LoRAInitializationStrategy strategy)
        {
            int inDim = _linearLayer.InFeatures;
            int outDim = _linearLayer.OutFeatures;

            switch (strategy)
            {
                case LoRAInitializationStrategy.Standard:
                    // A: Kaiming normal, B: Zeros
                    _loraA = KaimingNormal(new[] { Rank, inDim });
                    _loraB = Tensor.Zeros(new[] { outDim, Rank });
                    break;

                case LoRAInitializationStrategy.Xavier:
                    // Both: Xavier uniform
                    _loraA = XavierUniform(new[] { Rank, inDim });
                    _loraB = XavierUniform(new[] { outDim, Rank });
                    break;

                case LoRAInitializationStrategy.Zero:
                    // Both: Zeros
                    _loraA = Tensor.Zeros(new[] { Rank, inDim });
                    _loraB = Tensor.Zeros(new[] { outDim, Rank });
                    break;

                default:
                    throw new ArgumentException($"Unknown initialization strategy: {strategy}");
            }
        }

        /// <summary>
        /// Forward pass through the LoRA-adapted linear layer
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            // Standard forward pass through base layer
            var output = _linearLayer.Forward(input);

            if (!IsEnabled)
                return output;

            // LoRA forward pass: Wx + (alpha/r) * B(Ax)
            // Compute Ax (InDim -> Rank)
            var loraInput = MatMul(input, _loraA.Transpose()); // [batch, in] x [in, rank] = [batch, rank]

            // Apply dropout if enabled (training mode)
            if (_dropoutRate > 0.0f && IsTraining)
            {
                loraInput = ApplyDropout(loraInput);
            }

            // Compute B(Ax) (Rank -> OutDim)
            var loraOutput = MatMul(loraInput, _loraB.Transpose()); // [batch, rank] x [rank, out] = [batch, out]

        // Scale by alpha/r
        loraOutput = loraOutput * ScalingFactor; // Tensor * float (scalar)

            // Add bias adapter if present
            if (_loraBias != null)
            {
                output = output + loraOutput + _loraBias;
            }
            else
            {
                output = output + loraOutput;
            }

            return output;
        }

        private Tensor ApplyDropout(Tensor tensor)
        {
            var mask = RandomUniform(tensor.Shape, _dropoutRandom!);
            mask = Where(mask.GreaterThan(_dropoutRate), 1.0f / (1.0f - _dropoutRate), 0.0f);
            return tensor.Mul(mask); // Element-wise multiplication
        }

        public override void FreezeBaseLayer()
        {
            // Mark base layer weights as frozen
            _linearLayer.Weight.RequiresGrad = false;
            if (_linearLayer.Bias != null)
            {
                _linearLayer.Bias.RequiresGrad = false;
            }
            _isBaseLayerFrozen = true;
        }

        public override void UnfreezeBaseLayer()
        {
            // Mark base layer weights as trainable
            _linearLayer.Weight.RequiresGrad = true;
            if (_linearLayer.Bias != null)
            {
                _linearLayer.Bias.RequiresGrad = true;
            }
            _isBaseLayerFrozen = false;
        }

        public override IEnumerable<Tensor> TrainableParameters
        {
            get
            {
                if (!_isBaseLayerFrozen)
                {
                    yield return _linearLayer.Weight;
                    if (_linearLayer.Bias != null)
                        yield return _linearLayer.Bias;
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
                    yield return _linearLayer.Weight;
                    if (_linearLayer.Bias != null)
                        yield return _linearLayer.Bias;
                }
            }
        }

        public override void MergeAdapter()
        {
            // Backup original weights
            _baseLayerWeightsBackup = _linearLayer.Weight.Clone();

            // W_new = W + (alpha/r) * B * A
            var deltaW = MatMul(_loraB, _loraA); // [out, rank] x [rank, in] = [out, in]
            deltaW = deltaW * ScalingFactor; // Tensor * float (scalar)

            _linearLayer.Weight = _linearLayer.Weight + deltaW;
        }

        public override void ResetBaseLayer()
        {
            if (_baseLayerWeightsBackup == null)
                throw new InvalidOperationException("No backup available. Cannot reset.");

            _linearLayer.Weight = _baseLayerWeightsBackup;
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

            // Validate shapes
            if (matrixA.Shape.Length != 2 || matrixA.Shape[0] != Rank || matrixA.Shape[1] != InDim)
                throw new ArgumentException($"Matrix A shape must be [{Rank}, {InDim}]");

            if (matrixB.Shape.Length != 2 || matrixB.Shape[0] != OutDim || matrixB.Shape[1] != Rank)
                throw new ArgumentException($"Matrix B shape must be [{OutDim}, {Rank}]");

            // Copy weights
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
                if (bias.Shape.Length != 1 || bias.Shape[0] != OutDim)
                    throw new ArgumentException($"Bias shape must be [{OutDim}]");
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
            // a: [m, n]
            // b: [p, n]
            // result: [m, p]

            int m = a.Shape[0];
            int n = a.Shape[1];
            int p = b.Shape[0];

            var resultData = new float[m * p];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += a[new[] { i, k }] * b[new[] { j, k }];
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

        private Tensor Where(Tensor condition, float trueValue, float falseValue)
        {
            var data = new float[condition.Size];
            for (int i = 0; i < data.Length; i++)
            {
                // This is a simplified implementation
                // In a real implementation, we'd access the tensor data directly
                data[i] = falseValue; // Placeholder
            }
            return new Tensor(data, condition.Shape);
        }
    }
}
