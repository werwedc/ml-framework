namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Factory for creating row parallel linear layers with sensible defaults.
    /// </summary>
    public static class RowParallelLinearFactory
    {
        /// <summary>
        /// Create a row parallel linear layer with sensible defaults.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="outputSize">Output feature dimension</param>
        /// <param name="bias">Whether to include bias</param>
        /// <param name="inputIsSharded">Whether input is already sharded from previous column-parallel layer</param>
        /// <returns>Row parallel linear layer</returns>
        public static RowParallelLinear Create(
            int inputSize,
            int outputSize,
            bool bias = true,
            bool inputIsSharded = true)
        {
            return new RowParallelLinear(inputSize, outputSize, bias, inputIsSharded);
        }

        /// <summary>
        /// Create a row parallel linear layer for MLP output layer.
        /// This is typically used after a column-parallel hidden layer.
        /// </summary>
        /// <param name="hiddenSize">Hidden feature dimension</param>
        /// <param name="outputSize">Output feature dimension</param>
        /// <param name="bias">Whether to include bias</param>
        /// <returns>Row parallel linear layer configured for MLP output</returns>
        public static RowParallelLinear CreateForMLPOutput(
            int hiddenSize,
            int outputSize,
            bool bias = true)
        {
            return new RowParallelLinear(
                hiddenSize,
                outputSize,
                bias: bias,
                inputIsSharded: true);
        }

        /// <summary>
        /// Create a row parallel linear layer with input gathering.
        /// This is less efficient but more flexible for various input scenarios.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="outputSize">Output feature dimension</param>
        /// <param name="bias">Whether to include bias</param>
        /// <returns>Row parallel linear layer with input gathering</returns>
        public static RowParallelLinearWithInputGather CreateWithGather(
            int inputSize,
            int outputSize,
            bool bias = true)
        {
            return new RowParallelLinearWithInputGather(inputSize, outputSize, bias, inputIsSharded: false);
        }
    }
}
