using MLFramework.Distributed;
using RitterFramework.Core.Tensor;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Row Parallel Linear layer that automatically gathers sharded input if needed.
    /// This is less efficient than RowParallelLinear but more flexible when input
    /// might come from different sources.
    /// </summary>
    public class RowParallelLinearWithInputGather : RowParallelLinear
    {
        /// <summary>
        /// Creates a new RowParallelLinearWithInputGather layer.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="outputSize">Output feature dimension</param>
        /// <param name="bias">Whether to include bias</param>
        /// <param name="inputIsSharded">Whether input is already sharded</param>
        /// <param name="processGroup">Optional process group for TP operations</param>
        public RowParallelLinearWithInputGather(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool inputIsSharded = false,
            TensorParallelGroup? processGroup = null)
            : base(inputSize, outputSize, bias, inputIsSharded, processGroup)
        {
        }

        /// <summary>
        /// Forward pass that automatically gathers sharded input if needed.
        /// This is less efficient but more flexible for various input scenarios.
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <returns>Output tensor</returns>
        public new Tensor Forward(Tensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            var readyInput = input;

            // If input is sharded (from previous column-parallel), gather it first
            // This is less efficient but more flexible
            if (InputIsSharded)
            {
                var gatherComm = _processGroup ?? new TensorParallelGroup(TensorParallel.GetCommunicator());
                readyInput = gatherComm.AllGatherAsync(input, dim: -1).Result;
            }

            // Now proceed with non-sharded input path
            // Slice our portion of the input
            int startIdx = _rank * (_inputSize / _worldSize);
            int endIdx = startIdx + (_inputSize / _worldSize);
            var slicedInput = readyInput.Slice(-1, startIdx, endIdx);

            // Local matmul
            var outputLocal = TensorMathExtensions.MatMul(slicedInput, _weight, transposeB: true);

            // All-reduce (modifies outputLocal in-place)
            var reduceComm = _processGroup ?? new TensorParallelGroup(TensorParallel.GetCommunicator());
            reduceComm.AllReduce(outputLocal, ReduceOp.Sum);

            // Add bias
            if (_bias != null)
            {
                outputLocal = AddBias(outputLocal, _bias);
            }

            return outputLocal;
        }
    }
}
