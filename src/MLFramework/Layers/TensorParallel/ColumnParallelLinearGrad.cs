using MLFramework.Distributed;
using MLFramework.NN;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Column-parallel linear layer with gradient support for tensor parallelism.
    /// The weight matrix is split along the output dimension (columns).
    /// </summary>
    public class ColumnParallelLinearGrad : TPGradientLayer
    {
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly bool _gatherOutput;

        private Tensor _weight; // [out/WorldSize, in]
        private Parameter? _weightParam;
        private Parameter? _biasParam;

        // Saved for backward
        private Tensor? _lastInput;
        private bool _outputWasGathered;

        /// <summary>
        /// Gets the input feature dimension.
        /// </summary>
        public int InputSize => _inputSize;

        /// <summary>
        /// Gets the total output feature dimension (across all ranks).
        /// </summary>
        public int OutputSize => _outputSize;

        /// <summary>
        /// Gets the local output feature dimension (for this rank).
        /// </summary>
        public int LocalOutputSize => _outputSize / _worldSize;

        /// <summary>
        /// Gets whether output is gathered in forward pass.
        /// </summary>
        public bool GatherOutput => _gatherOutput;

        /// <summary>
        /// Creates a new ColumnParallelLinearGrad layer.
        /// </summary>
        /// <param name="inputSize">Total input feature dimension</param>
        /// <param name="outputSize">Total output feature dimension (split across ranks)</param>
        /// <param name="bias">Whether to include bias</param>
        /// <param name="gatherOutput">Whether to gather output across ranks in forward pass</param>
        /// <param name="processGroup">Optional process group for TP operations</param>
        public ColumnParallelLinearGrad(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool gatherOutput = false,
            TensorParallelGroup? processGroup = null)
            : base(processGroup)
        {
            if (inputSize <= 0)
                throw new ArgumentException("inputSize must be positive", nameof(inputSize));
            if (outputSize <= 0)
                throw new ArgumentException("outputSize must be positive", nameof(outputSize));
            if (outputSize % _worldSize != 0)
                throw new ArgumentException($"outputSize ({outputSize}) must be divisible by world size ({_worldSize})",
                    nameof(outputSize));

            _inputSize = inputSize;
            _outputSize = outputSize;
            _gatherOutput = gatherOutput;

            int shardOutSize = outputSize / _worldSize;
            _weight = InitializeWeight(shardOutSize, inputSize);
            _weightParam = new Parameter(_weight, "weight", requiresGrad: true);

            if (bias)
            {
                var biasTensor = Tensor.Zeros(new[] { shardOutSize });
                _biasParam = new Parameter(biasTensor, "bias", requiresGrad: true);
            }
        }

        /// <summary>
        /// Forward pass implementation.
        /// </summary>
        protected override Tensor ForwardInternal(Tensor input)
        {
            _lastInput = input; // Save for backward

            // output_local = x @ W_local^T
            // input: [batch, ..., in]
            // _weight: [out_shard, in]
            // output: [batch, ..., out_shard]
            var outputLocal = TensorMathExtensions.MatMul(input, _weight, transposeB: true);

            if (_biasParam != null)
            {
                outputLocal = outputLocal + _biasParam;
            }

            if (_gatherOutput)
            {
                _outputWasGathered = true;
                var comm = _processGroup ?? new TensorParallelGroup(TensorParallel.GetCommunicator());
                return comm.AllGatherAsync(outputLocal, dim: -1).Result;
            }

            _outputWasGathered = false;
            return outputLocal;
        }

        /// <summary>
        /// Backward pass implementation.
        /// </summary>
        protected override Tensor BackwardInternal(Tensor gradOutput)
        {
            Tensor gradOutputLocal = gradOutput;

            // If output was gathered in forward, slice gradient back to local shard
            if (_outputWasGathered)
            {
                int shardOutSize = _outputSize / _worldSize;
                int startIdx = _rank * shardOutSize;
                int endIdx = startIdx + shardOutSize;
                gradOutputLocal = gradOutputLocal.Slice(-1, startIdx, endIdx);
            }

            // Compute gradient w.r.t. weight: dW = x^T * dy_local
            // _lastInput: [batch, ..., in]
            // gradOutputLocal: [batch, ..., out_shard]
            // Result: [out_shard, in]
            var gradWeight = TensorMathExtensions.MatMul(gradOutputLocal, _lastInput, transposeA: true);

            // Compute gradient w.r.t. input: dx = dy_local * W^T
            // gradOutputLocal: [batch, ..., out_shard]
            // _weight: [out_shard, in]
            // Result: [batch, ..., in]
            var gradInput = TensorMathExtensions.MatMul(gradOutputLocal, _weight);

            // Accumulate gradient w.r.t. weight (no sync needed - each rank has its shard)
            if (_weightParam.Gradient == null)
            {
                _weightParam.Gradient = Tensor.Zeros(_weightParam.Shape);
            }
            _weightParam.Gradient = _weightParam.Gradient + gradWeight;

            // Compute gradient w.r.t. bias (if present)
            if (_biasParam != null)
            {
                // Sum over all dimensions except the last one
                var gradBias = gradOutputLocal.Sum();
                if (_biasParam.Gradient == null)
                {
                    _biasParam.Gradient = Tensor.Zeros(_biasParam.Shape);
                }
                _biasParam.Gradient = _biasParam.Gradient + gradBias;
            }

            // Return gradient w.r.t. input (full, no sync needed)
            return gradInput;
        }

        /// <summary>
        /// Gets trainable parameters.
        /// </summary>
        protected override IEnumerable<Parameter> GetTrainableParameters()
        {
            yield return _weightParam;
            if (_biasParam != null)
            {
                yield return _biasParam;
            }
        }

        /// <summary>
        /// Gets the weight parameter.
        /// </summary>
        public Parameter Weight => _weightParam;

        /// <summary>
        /// Gets the bias parameter (null if bias is disabled).
        /// </summary>
        public Parameter? Bias => _biasParam;

        private Tensor InitializeWeight(int outFeat, int inFeat)
        {
            // Kaiming initialization: std = sqrt(2 / fan_in)
            double std = Math.Sqrt(2.0 / (inFeat + outFeat));
            return TensorMathExtensions.RandomNormal(outFeat, inFeat, mean: 0.0, std: std);
        }
    }
}
