using RitterFramework.Core.Tensor;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Helper to create the standard column-then-row parallel MLP pattern.
    /// This is the most common pattern in Transformer models.
    /// </summary>
    public static class TPMLPFactory
    {
        /// <summary>
        /// Create an MLP block with column-parallel first layer and row-parallel second layer.
        /// This is the standard pattern for MLP layers in transformer models.
        /// </summary>
        /// <param name="inputSize">Input feature dimension</param>
        /// <param name="hiddenSize">Hidden feature dimension</param>
        /// <param name="outputSize">Output feature dimension</param>
        /// <param name="bias">Whether to include bias</param>
        /// <returns>Tuple of (column layer, row layer)</returns>
        public static (ColumnParallelLinear, RowParallelLinear) CreateMLPBlock(
            int inputSize,
            int hiddenSize,
            int outputSize,
            bool bias = true)
        {
            var columnLayer = new ColumnParallelLinear(
                inputSize,
                hiddenSize,
                bias: bias,
                gatherOutput: false);

            var rowLayer = new RowParallelLinear(
                hiddenSize,
                outputSize,
                bias: bias,
                inputIsSharded: true);

            return (columnLayer, rowLayer);
        }

        /// <summary>
        /// Forward pass through the combined MLP block.
        /// Input -> ColumnParallel (hidden, sharded) -> Activation -> RowParallel -> Output
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <param name="columnLayer">Column-parallel layer</param>
        /// <param name="rowLayer">Row-parallel layer</param>
        /// <param name="activation">Activation function (e.g., ReLU, GELU)</param>
        /// <returns>Output tensor</returns>
        public static Tensor ForwardMLP(
            Tensor input,
            ColumnParallelLinear columnLayer,
            RowParallelLinear rowLayer,
            Func<Tensor, Tensor> activation)
        {
            var hidden = columnLayer.Forward(input);
            var activated = activation(hidden);
            var output = rowLayer.Forward(activated);
            return output;
        }

        /// <summary>
        /// Create an MLP block with a ReLU activation.
        /// </summary>
        public static (ColumnParallelLinear, RowParallelLinear) CreateMLPBlockWithReLU(
            int inputSize,
            int hiddenSize,
            int outputSize,
            bool bias = true)
        {
            return CreateMLPBlock(inputSize, hiddenSize, outputSize, bias);
        }

        /// <summary>
        /// Create an MLP block with a GELU activation.
        /// </summary>
        public static (ColumnParallelLinear, RowParallelLinear) CreateMLPBlockWithGELU(
            int inputSize,
            int hiddenSize,
            int outputSize,
            bool bias = true)
        {
            return CreateMLPBlock(inputSize, hiddenSize, outputSize, bias);
        }

        /// <summary>
        /// Forward pass through MLP with ReLU activation.
        /// </summary>
        public static Tensor ForwardMLPWithReLU(
            Tensor input,
            ColumnParallelLinear columnLayer,
            RowParallelLinear rowLayer)
        {
            return ForwardMLP(input, columnLayer, rowLayer, ReLU);
        }

        /// <summary>
        /// Forward pass through MLP with GELU activation.
        /// </summary>
        public static Tensor ForwardMLPWithGELU(
            Tensor input,
            ColumnParallelLinear columnLayer,
            RowParallelLinear rowLayer)
        {
            return ForwardMLP(input, columnLayer, rowLayer, GELU);
        }

        #region Activation Functions

        /// <summary>
        /// ReLU activation function: max(0, x)
        /// </summary>
        private static Tensor ReLU(Tensor x)
        {
            var data = new float[x.Size];
            for (int i = 0; i < x.Size; i++)
            {
                data[i] = Math.Max(0, x.Data[i]);
            }
            return new Tensor(data, x.Shape, x.RequiresGrad, x.Dtype);
        }

        /// <summary>
        /// GELU activation function.
        /// Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        /// </summary>
        private static Tensor GELU(Tensor x)
        {
            var data = new float[x.Size];
            const float sqrt2OverPi = 0.7978845608f; // sqrt(2/pi)
            const float coeff = 0.044715f;

            for (int i = 0; i < x.Size; i++)
            {
                float xi = x.Data[i];
                float tanhArg = sqrt2OverPi * (xi + coeff * xi * xi * xi);
                data[i] = 0.5f * xi * (1.0f + (float)Math.Tanh(tanhArg));
            }
            return new Tensor(data, x.Shape, x.RequiresGrad, x.Dtype);
        }

        #endregion
    }
}
