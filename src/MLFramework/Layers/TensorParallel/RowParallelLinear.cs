using MLFramework.Distributed;
using RitterFramework.Core.Tensor;
using System;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Row Parallel Linear layer that splits the weight matrix along the row dimension across multiple devices.
    /// Each device computes with a portion of the input, and results are aggregated via all-reduce.
    /// This is typically paired with column-parallel layers in MLP/attention blocks.
    /// </summary>
    public class RowParallelLinear
    {
        protected readonly int _inputSize;
        protected readonly int _outputSize;
        protected readonly int _worldSize;
        protected readonly int _rank;
        protected readonly bool _inputIsSharded;
        protected readonly TensorParallelGroup? _processGroup;

        // The sharded weight matrix: [output_size, input_size / world_size]
        // Each rank holds a different row slice of the full weight matrix
        protected Tensor _weight;

        protected Tensor? _bias; // Optional bias: [output_size] (not sharded, shared)

        /// <summary>
        /// Gets the input feature dimension.
        /// </summary>
        public int InputSize => _inputSize;

        /// <summary>
        /// Gets the output feature dimension.
        /// </summary>
        public int OutputSize => _outputSize;

        /// <summary>
        /// Gets whether input is expected to be sharded.
        /// </summary>
        public bool InputIsSharded => _inputIsSharded;

        /// <summary>
        /// Gets the local input feature dimension (for this rank).
        /// </summary>
        public int LocalInputSize => _inputSize / _worldSize;

        /// <summary>
        /// Creates a new RowParallelLinear layer.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="outputSize">Output feature dimension</param>
        /// <param name="bias">Whether to include bias</param>
        /// <param name="inputIsSharded">Whether input is already sharded from previous column-parallel layer</param>
        /// <param name="processGroup">Optional process group for TP operations</param>
        public RowParallelLinear(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool inputIsSharded = true,
            TensorParallelGroup? processGroup = null)
        {
            if (inputSize <= 0)
                throw new ArgumentException("inputSize must be positive", nameof(inputSize));
            if (outputSize <= 0)
                throw new ArgumentException("outputSize must be positive", nameof(outputSize));

            _inputSize = inputSize;
            _outputSize = outputSize;
            _inputIsSharded = inputIsSharded;
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
            int shardInputSize = inputSize / _worldSize;
            if (inputSize % _worldSize != 0)
            {
                throw new ArgumentException(
                    $"inputSize ({inputSize}) must be divisible by worldSize ({_worldSize})");
            }

            // Initialize sharded weight
            // Shape: [output_size, input_size / world_size]
            _weight = InitializeWeight(outputSize, shardInputSize);

            if (bias)
            {
                // Bias is NOT sharded: [output_size]
                // All ranks use the same bias values
                _bias = Tensor.Zeros(new[] { outputSize });
            }
        }

        /// <summary>
        /// Forward pass for row parallel linear layer.
        /// Input:  [batch_size, ..., input_size] (full)
        ///         [batch_size, ..., input_size / world_size] (if sharded)
        /// Weight: [output_size, input_size / world_size]
        /// Output: [batch_size, ..., output_size] (after all-reduce)
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <returns>Output tensor</returns>
        public Tensor Forward(Tensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            Tensor readyInput = input;

            // If input is supposed to be sharded but isn't, we can still work
            // But typically, input comes from previous column-parallel layer
            if (_inputIsSharded)
            {
                // Input should already be sharded from previous column-parallel layer
                // Verify shape matches expected shard size
                int expectedInputSize = _inputSize / _worldSize;
                var inputShape = input.Shape;
                int lastDim = inputShape[^1];

                if (lastDim != expectedInputSize)
                {
                    throw new InvalidOperationException(
                        $"Expected sharded input with last dim={expectedInputSize}, " +
                        $"but got {lastDim}. If input is not sharded, set inputIsSharded=false.");
                }
            }
            else
            {
                // Input is not sharded, we need to slice it to our shard
                int startIdx = _rank * (_inputSize / _worldSize);
                int endIdx = startIdx + (_inputSize / _worldSize);
                readyInput = input.Slice(-1, startIdx, endIdx);
            }

            // Perform local matrix multiplication
            // readyInput: [batch_size, ..., input_shard]
            // weight: [output_size, input_shard]
            // output_local: [batch_size, ..., output_size]
            var outputLocal = TensorMathExtensions.MatMul(readyInput, _weight, transposeB: true);

            // All-reduce to sum results from all ranks (modifies outputLocal in-place)
            var comm = _processGroup ?? new TensorParallelGroup(TensorParallel.GetCommunicator());
            comm.AllReduce(outputLocal, ReduceOp.Sum);

            // Add bias if present (after all-reduce, so bias added once)
            if (_bias != null)
            {
                outputLocal = AddBias(outputLocal, _bias);
            }

            return outputLocal;
        }

        /// <summary>
        /// Get the local weight shard (for inspection, not modification).
        /// </summary>
        public Tensor GetLocalWeight() => _weight;

        /// <summary>
        /// Get the local bias (shared across all ranks).
        /// </summary>
        public Tensor? GetLocalBias() => _bias;

        /// <summary>
        /// Get the shape of the local weight shard.
        /// </summary>
        public (int rows, int cols) GetLocalWeightShape()
        {
            return (_outputSize, _inputSize / _worldSize);
        }

        /// <summary>
        /// Initialize weight with appropriate scaling.
        /// </summary>
        private Tensor InitializeWeight(int outFeatures, int inFeatures)
        {
            // Use Xavier/Glorot initialization
            double std = Math.Sqrt(2.0 / (inFeatures + outFeatures));
            return TensorMathExtensions.RandomNormal(outFeatures, inFeatures, mean: 0.0, std: std, seed: 42);
        }

        /// <summary>
        /// Add bias to output tensor.
        /// </summary>
        protected Tensor AddBias(Tensor input, Tensor bias)
        {
            // input: [batch_size, ..., output_size]
            // bias: [output_size]
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
