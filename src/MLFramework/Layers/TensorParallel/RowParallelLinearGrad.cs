using MLFramework.Distributed;
using MLFramework.NN;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Row-parallel linear layer with gradient support for tensor parallelism.
    /// The weight matrix is split along the input dimension (rows).
    /// </summary>
    public class RowParallelLinearGrad : TPGradientLayer
    {
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly bool _inputIsSharded;

        private Tensor _weight; // [out, in/WorldSize]
        private Parameter? _weightParam;
        private Parameter? _biasParam;

        // Saved for backward
        private Tensor? _lastInputSharded;

        /// <summary>
        /// Gets the total input feature dimension (across all ranks).
        /// </summary>
        public int InputSize => _inputSize;

        /// <summary>
        /// Gets the output feature dimension.
        /// </summary>
        public int OutputSize => _outputSize;

        /// <summary>
        /// Gets the local input feature dimension (for this rank).
        /// </summary>
        public int LocalInputSize => _inputSize / _worldSize;

        /// <summary>
        /// Gets whether input is expected to be sharded.
        /// </summary>
        public bool InputIsSharded => _inputIsSharded;

        /// <summary>
        /// Creates a new RowParallelLinearGrad layer.
        /// </summary>
        /// <param name="inputSize">Total input feature dimension (split across ranks)</param>
        /// <param name="outputSize">Output feature dimension</param>
        /// <param name="bias">Whether to include bias</param>
        /// <param name="inputIsSharded">Whether input is sharded across ranks</param>
        /// <param name="processGroup">Optional process group for TP operations</param>
        public RowParallelLinearGrad(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool inputIsSharded = true,
            TensorParallelGroup? processGroup = null)
            : base(processGroup)
        {
            if (inputSize <= 0)
                throw new ArgumentException("inputSize must be positive", nameof(inputSize));
            if (outputSize <= 0)
                throw new ArgumentException("outputSize must be positive", nameof(outputSize));
            if (inputSize % _worldSize != 0)
                throw new ArgumentException($"inputSize ({inputSize}) must be divisible by world size ({_worldSize})",
                    nameof(inputSize));

            _inputSize = inputSize;
            _outputSize = outputSize;
            _inputIsSharded = inputIsSharded;

            int shardInSize = inputSize / _worldSize;
            _weight = InitializeWeight(outputSize, shardInSize);
            _weightParam = new Parameter(_weight, "weight", requiresGrad: true);

            if (bias)
            {
                var biasTensor = Tensor.Zeros(new[] { outputSize });
                _biasParam = new Parameter(biasTensor, "bias", requiresGrad: true);
            }
        }

        /// <summary>
        /// Forward pass implementation.
        /// </summary>
        protected override Tensor ForwardInternal(Tensor input)
        {
            if (_inputIsSharded)
            {
                _lastInputSharded = input; // Save for backward
            }
            else
            {
                // Slice input to our shard
                int shardInSize = _inputSize / _worldSize;
                int startIdx = _rank * shardInSize;
                int endIdx = startIdx + shardInSize;
                _lastInputSharded = input.Slice(-1, startIdx, endIdx);
            }

            // output_local = x_sharded @ W_local^T
            // _lastInputSharded: [batch, ..., in_shard]
            // _weight: [out, in_shard]
            // outputLocal: [batch, ..., out]
            var outputLocal = TensorMathExtensions.MatMul(_lastInputSharded, _weight, transposeB: true);

            // All-reduce to sum results from all ranks
            var comm = _processGroup ?? new TensorParallelGroup(TensorParallel.GetCommunicator());
            comm.AllReduceAsync(outputLocal, ReduceOp.Sum).Wait();
            // AllReduce modifies tensor in-place, so outputLocal is now the reduced tensor
            var output = outputLocal;

            if (_biasParam != null)
            {
                output = output + _biasParam;
            }

            return output;
        }

        /// <summary>
        /// Backward pass implementation.
        /// </summary>
        protected override Tensor BackwardInternal(Tensor gradOutput)
        {
            // Gradient w.r.t. output is full (no slicing needed)
            // It was all-reduced in forward, so gradient is already summed

            // Compute gradient w.r.t. weight: dW = x_sharded^T * dy_full
            // _lastInputSharded: [batch, ..., in_shard]
            // gradOutput: [batch, ..., out]
            // Result: [out, in_shard]
            var gradWeight = TensorMathExtensions.MatMul(gradOutput, _lastInputSharded, transposeA: true);

            // Compute gradient w.r.t. input: dx_sharded = dy_full * W^T
            // gradOutput: [batch, ..., out]
            // _weight: [out, in_shard]
            // Result: [batch, ..., in_shard]
            var gradInputSharded = TensorMathExtensions.MatMul(gradOutput, _weight);

            // Accumulate gradient w.r.t. weight (no sync needed - each rank has its shard)
            if (_weightParam.Gradient == null)
            {
                _weightParam.Gradient = Tensor.Zeros(_weightParam.Shape);
            }
            _weightParam.Gradient = _weightParam.Gradient + gradWeight;

            // Compute gradient w.r.t. bias (if present)
            if (_biasParam != null)
            {
                var gradBias = gradOutput.Sum();
                if (_biasParam.Gradient == null)
                {
                    _biasParam.Gradient = Tensor.Zeros(_biasParam.Shape);
                }
                _biasParam.Gradient = _biasParam.Gradient + gradBias;
            }

            // Return gradient w.r.t. input (sharded, no sync needed)
            return gradInputSharded;
        }

        /// <summary>
        /// Gets trainable parameters.
        /// </summary>
        protected override System.Collections.Generic.IEnumerable<Parameter> GetTrainableParameters()
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
