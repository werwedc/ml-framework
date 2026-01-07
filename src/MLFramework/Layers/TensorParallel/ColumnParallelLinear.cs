using RitterFramework.Core.Tensor;
using System;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Column Parallel Linear layer that splits the weight matrix along the column dimension across multiple devices.
    /// This enables training models with output dimensions larger than a single device's memory capacity.
    /// In column parallelism, the output dimension is split across devices. Each device computes a portion of the output.
    /// </summary>
    public class ColumnParallelLinear
    {
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly int _worldSize;
        private readonly int _rank;
        private readonly bool _gatherOutput;
        private readonly TensorParallelGroup? _processGroup;

        // The sharded weight matrix: [output_size / world_size, input_size]
        // Each rank holds a different column slice of the full weight matrix
        private Tensor _weight;

        private Tensor? _bias; // Optional bias: [output_size / world_size]

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
        /// Creates a new ColumnParallelLinear layer.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="outputSize">Total output feature dimension (split across ranks)</param>
        /// <param name="bias">Whether to include bias</param>
        /// <param name="gatherOutput">Whether to gather output across ranks in forward pass</param>
        /// <param name="processGroup">Optional process group for TP operations</param>
        public ColumnParallelLinear(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool gatherOutput = false,
            TensorParallelGroup? processGroup = null)
        {
            if (inputSize <= 0)
                throw new ArgumentException("inputSize must be positive", nameof(inputSize));
            if (outputSize <= 0)
                throw new ArgumentException("outputSize must be positive", nameof(outputSize));

            _inputSize = inputSize;
            _outputSize = outputSize;
            _gatherOutput = gatherOutput;
            _processGroup = processGroup;

            if (processGroup != null)
            {
                _worldSize = processGroup.WorldSize;
                _rank = processGroup.Rank;
            }
            else if (TensorParallel.IsInitialized())
            {
                _worldSize = TensorParallel.GetWorldSize();
                _rank = TensorParallel.GetRank();
            }
            else
            {
                // Default to single-process mode
                _worldSize = 1;
                _rank = 0;
            }

            // Calculate sharded dimensions
            int shardOutputSize = outputSize / _worldSize;
            if (outputSize % _worldSize != 0)
            {
                throw new ArgumentException(
                    $"outputSize ({outputSize}) must be divisible by worldSize ({_worldSize})");
            }

            // Initialize sharded weight
            // Shape: [output_size / world_size, input_size]
            _weight = InitializeWeight(shardOutputSize, inputSize);

            if (bias)
            {
                // Sharded bias: [output_size / world_size]
                _bias = Tensor.Zeros(new[] { shardOutputSize });
            }
        }

        /// <summary>
        /// Forward pass for column parallel linear layer
        /// Input:  [batch_size, ..., input_size]
        /// Weight: [output_size / world_size, input_size]
        /// Output: [batch_size, ..., output_size / world_size] (if !gatherOutput)
        ///         [batch_size, ..., output_size] (if gatherOutput)
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <returns>Output tensor</returns>
        public Tensor Forward(Tensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            // Perform local matrix multiplication
            // input: [batch_size, ..., input_size]
            // weight: [output_shard, input_size]
            // output_local: [batch_size, ..., output_shard]
            var outputLocal = TensorMathExtensions.MatMul(input, _weight, transposeB: true);

            // Add bias if present
            if (_bias != null)
            {
                outputLocal = AddBias(outputLocal, _bias);
            }

            // Optionally gather output across ranks
            if (_gatherOutput)
            {
                // Gather along the output dimension (last dimension)
                // Result: [batch_size, ..., output_size]
                var comm = _processGroup ?? new TensorParallelGroup(TensorParallel.GetCommunicator());
                return comm.AllGatherAsync(outputLocal, dim: -1).Result;
            }

            return outputLocal;
        }

        /// <summary>
        /// Get the local weight shard (for inspection, not modification)
        /// </summary>
        public Tensor GetLocalWeight() => _weight;

        /// <summary>
        /// Get the local bias shard (for inspection, not modification)
        /// </summary>
        public Tensor? GetLocalBias() => _bias;

        /// <summary>
        /// Get the shape of the local weight shard
        /// </summary>
        public (int rows, int cols) GetLocalWeightShape()
        {
            return (_outputSize / _worldSize, _inputSize);
        }

        /// <summary>
        /// Initialize weight with appropriate scaling
        /// </summary>
        private Tensor InitializeWeight(int outFeatures, int inFeatures)
        {
            // Use Xavier/Glorot initialization
            double std = Math.Sqrt(2.0 / (inFeatures + outFeatures));
            return TensorMathExtensions.RandomNormal(outFeatures, inFeatures, mean: 0.0, std: std, seed: 42);
        }

        /// <summary>
        /// Add bias to output tensor
        /// </summary>
        private Tensor AddBias(Tensor input, Tensor bias)
        {
            // input: [batch_size, ..., out_shard]
            // bias: [out_shard]
            // Add bias to each output feature

            var resultData = new float[input.Size];
            var outFeatures = bias.Shape[0];
            int batchSize = input.Size / outFeatures;

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < outFeatures; f++)
                {
                    resultData[b * outFeatures + f] = input[new[] { b, f }] + bias[new[] { f }];
                }
            }

            return new Tensor(resultData, input.Shape, input.RequiresGrad || bias.RequiresGrad);
        }
    }
}
