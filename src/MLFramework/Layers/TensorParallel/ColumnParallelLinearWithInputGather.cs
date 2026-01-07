using RitterFramework.Core.Tensor;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Column Parallel Linear layer with optional input gathering.
    /// Extends ColumnParallelLinear to support scenarios where the input might be sharded
    /// from a previous row-parallel layer and needs to be gathered first.
    /// </summary>
    public class ColumnParallelLinearWithInputGather : ColumnParallelLinear
    {
        private readonly bool _inputIsSharded;
        private readonly TensorParallelGroup? _processGroup;

        /// <summary>
        /// Gets whether input is expected to be sharded.
        /// </summary>
        public bool InputIsSharded => _inputIsSharded;

        /// <summary>
        /// Creates a new ColumnParallelLinearWithInputGather layer.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="outputSize">Total output feature dimension (split across ranks)</param>
        /// <param name="bias">Whether to include bias</param>
        /// <param name="gatherOutput">Whether to gather output across ranks in forward pass</param>
        /// <param name="inputIsSharded">Whether input is expected to be sharded from a previous row-parallel layer</param>
        /// <param name="processGroup">Optional process group for TP operations</param>
        public ColumnParallelLinearWithInputGather(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool gatherOutput = false,
            bool inputIsSharded = false,
            TensorParallelGroup? processGroup = null)
            : base(inputSize, outputSize, bias, gatherOutput, processGroup)
        {
            _inputIsSharded = inputIsSharded;
            _processGroup = processGroup;
        }

        /// <summary>
        /// Forward pass that optionally gathers input first.
        /// </summary>
        /// <param name="input">Input tensor (may be sharded if inputIsSharded=true)</param>
        /// <returns>Output tensor</returns>
        public new Tensor Forward(Tensor input)
        {
            Tensor readyInput = input;

            // If input is sharded across previous row-parallel layer, gather it
            if (_inputIsSharded)
            {
                var comm = _processGroup ?? new TensorParallelGroup(TensorParallel.GetCommunicator());
                readyInput = comm.AllGatherAsync(input, dim: -1).Result;
            }

            return base.Forward(readyInput);
        }
    }
}
