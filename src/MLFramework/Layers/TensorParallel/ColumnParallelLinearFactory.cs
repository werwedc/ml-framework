namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Factory class for creating Column Parallel Linear layers with sensible defaults.
    /// </summary>
    public static class ColumnParallelLinearFactory
    {
        /// <summary>
        /// Create a column parallel linear layer with sensible defaults.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="outputSize">Total output feature dimension (split across ranks)</param>
        /// <param name="bias">Whether to include bias</param>
        /// <param name="gatherOutput">Whether to gather output across ranks in forward pass</param>
        /// <returns>Configured ColumnParallelLinear layer</returns>
        public static ColumnParallelLinear Create(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool gatherOutput = false)
        {
            return new ColumnParallelLinear(inputSize, outputSize, bias, gatherOutput);
        }

        /// <summary>
        /// Create for transformer attention projection (no bias, gather output).
        /// This is typical for Q, K, V projections in multi-head attention.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="outputSize">Total output feature dimension (split across ranks)</param>
        /// <returns>Configured ColumnParallelLinear layer</returns>
        public static ColumnParallelLinear CreateForAttention(
            int inputSize,
            int outputSize)
        {
            return new ColumnParallelLinear(
                inputSize,
                outputSize,
                bias: false,
                gatherOutput: true);
        }

        /// <summary>
        /// Create for MLP hidden layer (with bias, don't gather - feed to row parallel).
        /// This is typical for the first linear layer in an MLP block where output
        /// is sharded and passed to a row-parallel linear layer.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="hiddenSize">Hidden feature dimension (split across ranks)</param>
        /// <returns>Configured ColumnParallelLinear layer</returns>
        public static ColumnParallelLinear CreateForMLPHidden(
            int inputSize,
            int hiddenSize)
        {
            return new ColumnParallelLinear(
                inputSize,
                hiddenSize,
                bias: true,
                gatherOutput: false);
        }

        /// <summary>
        /// Create with input gathering for scenarios where input might be sharded.
        /// This is useful when the input comes from a previous row-parallel layer.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="outputSize">Total output feature dimension (split across ranks)</param>
        /// <param name="bias">Whether to include bias</param>
        /// <param name="gatherOutput">Whether to gather output across ranks in forward pass</param>
        /// <param name="inputIsSharded">Whether input is expected to be sharded</param>
        /// <returns>Configured ColumnParallelLinearWithInputGather layer</returns>
        public static ColumnParallelLinearWithInputGather CreateWithInputGather(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool gatherOutput = false,
            bool inputIsSharded = true)
        {
            return new ColumnParallelLinearWithInputGather(
                inputSize,
                outputSize,
                bias,
                gatherOutput,
                inputIsSharded);
        }
    }
}
