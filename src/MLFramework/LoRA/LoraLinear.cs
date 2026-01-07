using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Modules;
using MLFramework.NN;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.LoRA
{
    /// <summary>
    /// LoRA (Low-Rank Adaptation) adapter for Linear layers.
    /// Wraps a standard Linear layer and injects low-rank adapter matrices for efficient fine-tuning.
    /// </summary>
    public class LoraLinear : LoRAAdapterBase, IModule
    {
        private readonly Linear _baseLinear;
        private readonly Parameter _loraA;     // [out_features, rank] - trainable
        private readonly Parameter _loraB;     // [rank, in_features] - trainable
        private readonly float _dropout;
        private bool _merged;
        private readonly Random? _dropoutRandom;

        public int InFeatures => _baseLinear.InFeatures;
        public int OutFeatures => _baseLinear.OutFeatures;

        public new string ModuleType => "LoraLinear";

        public bool IsEnabled { get; set; } = true;

        /// <summary>
        /// Creates a new LoRA adapter for a Linear layer
        /// </summary>
        public LoraLinear(Linear baseLinear, int rank, int alpha, float dropout = 0.0f)
            : base(baseLinear, rank, alpha)
        {
            _baseLinear = baseLinear ?? throw new ArgumentNullException(nameof(baseLinear));
            _dropout = dropout;

            // Freeze base layer
            _baseLinear.SetRequiresGrad(false);

            // Initialize LoRA matrices
            // A: [out_features, rank] - initialized with small random values
            // B: [rank, in_features] - initialized with zeros (start with no effect)
            _loraA = InitializeLoraA(baseLinear.OutFeatures, rank);
            _loraB = InitializeLoraB(rank, baseLinear.InFeatures);

            _merged = false;

            if (_dropout > 0.0f)
            {
                _dropoutRandom = new Random(42);
            }
        }

        /// <summary>
        /// Forward pass through the LoRA-adapted Linear layer
        /// </summary>
        public Tensor Forward(Tensor x)
        {
            if (!IsEnabled || _merged)
            {
                return _baseLinear.Forward(x);
            }

            // Base computation (frozen)
            var baseOutput = _baseLinear.Forward(x);

            // LoRA computation: x @ B.T @ A.T * scaling
            // x: [batch, in_features]
            // B: [rank, in_features], B.T: [in_features, rank]
            // A: [out_features, rank], A.T: [rank, out_features]
            var loraOutput = MatMul(x, _loraB.Transpose());

            if (_dropout > 0.0f)
            {
                loraOutput = ApplyDropout(loraOutput, _dropout);
            }

            loraOutput = MatMul(loraOutput, _loraA.Transpose()) * ScalingFactor;

            return AddTensors(baseOutput, loraOutput);
        }

        /// <summary>
        /// Merge LoRA weights into base weights for inference
        /// </summary>
        public void Merge()
        {
            if (_merged)
                return;

            // W_merged = W_base + (B.T @ A.T) * scaling
            var deltaWeight = MatMul(_loraB.Transpose(), _loraA.Transpose()) * ScalingFactor;

            // Add delta to base weight
            AddTensorsInPlace(_baseLinear.Weight, deltaWeight);

            _merged = true;
        }

        /// <summary>
        /// Unmerge LoRA weights (restore base weights)
        /// </summary>
        public void Unmerge()
        {
            if (!_merged)
                return;

            // Subtract delta from base weight
            var deltaWeight = MatMul(_loraB.Transpose(), _loraA.Transpose()) * ScalingFactor;
            SubtractTensorsInPlace(_baseLinear.Weight, deltaWeight);

            _merged = false;
        }

        /// <summary>
        /// Get trainable LoRA parameters
        /// </summary>
        public override IEnumerable<Tensor> TrainableParameters
        {
            get
            {
                if (_merged)
                    return Enumerable.Empty<Tensor>();

                return new[] { _loraA, _loraB };
            }
        }

        /// <summary>
        /// Get frozen parameters
        /// </summary>
        public override IEnumerable<Tensor> FrozenParameters
        {
            get { return _baseLinear.Parameters; }
        }

        /// <summary>
        /// Freeze the base layer
        /// </summary>
        public override void FreezeBaseLayer()
        {
            _baseLinear.SetRequiresGrad(false);
        }

        /// <summary>
        /// Unfreeze the base layer
        /// </summary>
        public override void UnfreezeBaseLayer()
        {
            _baseLinear.SetRequiresGrad(true);
        }

        /// <summary>
        /// Merge the adapter weights into the base layer
        /// </summary>
        public override void MergeAdapter()
        {
            Merge();
        }

        /// <summary>
        /// Reset the base layer to original weights
        /// </summary>
        public override void ResetBaseLayer()
        {
            Unmerge();
        }

        /// <summary>
        /// Get adapter weights
        /// </summary>
        public override (Tensor? MatrixA, Tensor? MatrixB) GetAdapterWeights()
        {
            return (_loraA, _loraB);
        }

        /// <summary>
        /// Set adapter weights
        /// </summary>
        public override void SetAdapterWeights(Tensor? matrixA, Tensor? matrixB)
        {
            if (matrixA != null)
            {
                if (matrixA.Shape.Length != 2 || matrixA.Shape[0] != OutFeatures || matrixA.Shape[1] != Rank)
                    throw new ArgumentException("MatrixA must have shape [out_features, rank]");

                _loraA.CopyFrom(matrixA);
            }

            if (matrixB != null)
            {
                if (matrixB.Shape.Length != 2 || matrixB.Shape[0] != Rank || matrixB.Shape[1] != InFeatures)
                    throw new ArgumentException("MatrixB must have shape [rank, in_features]");

                _loraB.CopyFrom(matrixB);
            }
        }

        private Parameter InitializeLoraA(int outFeatures, int rank)
        {
            // Initialize A with small random values (std=0.01)
            float std = 0.01f;
            var random = new Random(42);
            var data = new float[outFeatures * rank];

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

            return new Parameter(data, new[] { outFeatures, rank }, "loraA", requiresGrad: true);
        }

        private Parameter InitializeLoraB(int rank, int inFeatures)
        {
            // Initialize B with zeros (start with no effect)
            return new Parameter(new float[rank * inFeatures], new[] { rank, inFeatures }, "loraB", requiresGrad: true);
        }

        private Tensor MatMul(Tensor a, Tensor b)
        {
            // Standard matrix multiplication: a @ b
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

        private Tensor AddTensors(Tensor a, Tensor b)
        {
            if (!a.Shape.SequenceEqual(b.Shape))
                throw new ArgumentException("Shapes must match for addition");

            var resultData = new float[a.Size];

            for (int i = 0; i < a.Size; i++)
            {
                resultData[i] = a.Data[i] + b.Data[i];
            }

            return new Tensor(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad);
        }

        private void AddTensorsInPlace(Tensor target, Tensor source)
        {
            if (!target.Shape.SequenceEqual(source.Shape))
                throw new ArgumentException("Shapes must match for in-place addition");

            for (int i = 0; i < target.Size; i++)
            {
                target.Data[i] += source.Data[i];
            }
        }

        private void SubtractTensorsInPlace(Tensor target, Tensor source)
        {
            if (!target.Shape.SequenceEqual(source.Shape))
                throw new ArgumentException("Shapes must match for in-place subtraction");

            for (int i = 0; i < target.Size; i++)
            {
                target.Data[i] -= source.Data[i];
            }
        }

        private Tensor ApplyDropout(Tensor input, float dropoutRate)
        {
            var resultData = new float[input.Size];
            var random = _dropoutRandom ?? new Random(42);

            float scale = 1.0f / (1.0f - dropoutRate);

            for (int i = 0; i < input.Size; i++)
            {
                if (random.NextDouble() >= dropoutRate)
                {
                    resultData[i] = input.Data[i] * scale;
                }
                else
                {
                    resultData[i] = 0.0f;
                }
            }

            return new Tensor(resultData, input.Shape, input.RequiresGrad);
        }
    }
}
