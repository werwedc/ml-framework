using RitterFramework.Core.Tensor;

namespace MLFramework.Modules
{
    /// <summary>
    /// A simple linear (fully connected) layer
    /// </summary>
    public class Linear : IModule
    {
        public Tensor Weight { get; set; }
        public Tensor? Bias { get; set; }

        public int InFeatures { get; }
        public int OutFeatures { get; }

        public string ModuleType => "Linear";

        public bool IsTraining { get; set; } = false;

        /// <summary>
        /// Creates a new linear layer
        /// </summary>
        /// <param name="inFeatures">Number of input features</param>
        /// <param name="outFeatures">Number of output features</param>
        /// <param name="useBias">Whether to use bias</param>
        public Linear(int inFeatures, int outFeatures, bool useBias = true)
        {
            if (inFeatures <= 0)
                throw new ArgumentException("inFeatures must be positive", nameof(inFeatures));
            if (outFeatures <= 0)
                throw new ArgumentException("outFeatures must be positive", nameof(outFeatures));

            InFeatures = inFeatures;
            OutFeatures = outFeatures;

            // Initialize weights with Xavier uniform initialization
            Weight = InitializeWeight(new[] { outFeatures, inFeatures });

            if (useBias)
            {
                Bias = Tensor.Zeros(new[] { outFeatures });
            }
        }

        /// <summary>
        /// Creates a linear layer from existing weights
        /// </summary>
        public Linear(Tensor weight, Tensor? bias = null)
        {
            Weight = weight ?? throw new ArgumentNullException(nameof(weight));

            if (weight.Shape.Length != 2)
                throw new ArgumentException("Weight must be 2-dimensional", nameof(weight));

            InFeatures = weight.Shape[1];
            OutFeatures = weight.Shape[0];
            Bias = bias;
        }

        /// <summary>
        /// Gets all parameters in the module
        /// </summary>
        public IEnumerable<Tensor> Parameters
        {
            get
            {
                yield return Weight;
                if (Bias != null)
                    yield return Bias;
            }
        }

        /// <summary>
        /// Forward pass: y = xW^T + b
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            // input shape: [batch_size, in_features]
            // weight shape: [out_features, in_features]
            // output shape: [batch_size, out_features]

            // Matrix multiplication: input @ weight.T
            var output = MatMul(input, Weight);

            // Add bias if present
            if (Bias != null)
            {
                output = AddBias(output, Bias);
            }

            return output;
        }

        /// <summary>
        /// Applies a function to all parameters
        /// </summary>
        public void ApplyToParameters(Action<Tensor> action)
        {
            action(Weight);
            if (Bias != null)
            {
                action(Bias);
            }
        }

        /// <summary>
        /// Sets the requires_grad flag for all parameters
        /// </summary>
        public void SetRequiresGrad(bool requiresGrad)
        {
            Weight.RequiresGrad = requiresGrad;
            if (Bias != null)
            {
                Bias.RequiresGrad = requiresGrad;
            }
        }

        private Tensor InitializeWeight(int[] shape)
        {
            // Xavier/Glorot uniform initialization
            // limit = sqrt(6 / (fan_in + fan_out))
            int fanIn = shape[1];
            int fanOut = shape[0];
            float limit = (float)Math.Sqrt(6.0 / (fanIn + fanOut));

            var random = new Random(42);
            var data = new float[shape[0] * shape[1]];

            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)(random.NextDouble() * 2 * limit - limit);
            }

            return new Tensor(data, shape, requiresGrad: true);
        }

        // Simple matrix multiplication implementation
        private Tensor MatMul(Tensor a, Tensor b)
        {
            // a: [batch_size, in_features]
            // b: [out_features, in_features]
            // result: [batch_size, out_features]

            int batchSize = a.Shape[0];
            int inFeatures = a.Shape[1];
            int outFeatures = b.Shape[0];

            var resultData = new float[batchSize * outFeatures];

            // result[i, j] = sum_k a[i, k] * b[j, k]
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outFeatures; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < inFeatures; k++)
                    {
                        sum += a[new[] { i, k }] * b[new[] { j, k }];
                    }
                    resultData[i * outFeatures + j] = sum;
                }
            }

            return new Tensor(resultData, new[] { batchSize, outFeatures }, a.RequiresGrad || b.RequiresGrad);
        }

        private Tensor AddBias(Tensor input, Tensor bias)
        {
            // input: [batch_size, out_features]
            // bias: [out_features]
            // Add bias to each row

            var resultData = new float[input.Size];
            var outFeatures = bias.Shape[0];
            int batchSize = input.Shape[0];

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < outFeatures; j++)
                {
                    resultData[i * outFeatures + j] = input[new[] { i, j }] + bias[new[] { j }];
                }
            }

            return new Tensor(resultData, input.Shape, input.RequiresGrad || bias.RequiresGrad);
        }
    }
}
