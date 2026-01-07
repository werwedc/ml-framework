using RitterFramework.Core.Tensor;
using System;

namespace MLFramework.Modules
{
    /// <summary>
    /// A fully connected (dense) linear layer
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
        /// Creates a new Linear layer
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

            // Initialize weights: [out_features, in_features]
            Weight = InitializeWeight(new[] { outFeatures, inFeatures });

            if (useBias)
            {
                Bias = Tensor.Zeros(new[] { outFeatures });
            }
        }

        /// <summary>
        /// Creates a Linear layer from existing weights
        /// </summary>
        /// <param name="weight">Weight tensor [out_features, in_features]</param>
        /// <param name="bias">Optional bias tensor [out_features]</param>
        public Linear(Tensor weight, Tensor? bias = null)
        {
            Weight = weight ?? throw new ArgumentNullException(nameof(weight));

            if (weight.Shape.Length != 2)
                throw new ArgumentException("Weight must be 2-dimensional [outF, inF]", nameof(weight));

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
        /// Forward pass: linear transformation y = xW^T + b
        /// </summary>
        /// <param name="input">Input tensor [batch_size, in_features]</param>
        /// <returns>Output tensor [batch_size, out_features]</returns>
        public Tensor Forward(Tensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            // Input shape: [batch_size, in_features]
            // Weight shape: [out_features, in_features]
            // Output shape: [batch_size, out_features]

            int batchSize = input.Shape[0];
            int inFeatures = input.Shape[1];

            if (inFeatures != InFeatures)
                throw new ArgumentException(
                    $"Input feature dimension {inFeatures} does not match layer's InFeatures {InFeatures}");

            // Compute output = input @ Weight.T
            // input: [batch, inF]
            // Weight.T: [inF, outF]
            // output: [batch, outF]
            var weightTransposed = Weight.Transpose();
            var output = MatMul(input, weightTransposed);

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
            // Kaiming initialization for linear layers
            // std = sqrt(2 / fan_in)
            int fanIn = shape[1]; // inFeatures
            float std = (float)Math.Sqrt(2.0 / fanIn);

            var random = new Random(42);
            var data = new float[shape[0] * shape[1]];

            // Box-Muller transform for normal distribution
            for (int i = 0; i < data.Length; i += 2)
            {
                float u1 = (float)random.NextDouble();
                float u2 = (float)random.NextDouble();

                float z0 = (float)(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                float z1 = (float)(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));

                data[i] = z0;
                if (i + 1 < data.Length)
                {
                    data[i + 1] = z1;
                }
            }

            return new Tensor(data, shape, requiresGrad: true);
        }

        private Tensor MatMul(Tensor a, Tensor b)
        {
            // Standard matrix multiplication: a @ b
            // a: [m, n]
            // b: [n, p]
            // result: [m, p]

            int m = a.Shape[0];
            int n = a.Shape[1];
            int p = b.Shape[1];

            if (b.Shape[0] != n)
                throw new ArgumentException($"Inner dimensions must match: a.Shape[1]={n}, b.Shape[0]={b.Shape[0]}");

            var resultData = new float[m * p];

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < n; k++)
                    {
                        sum += a[new[] { i, k }] * b[new[] { k, j }];
                    }
                    resultData[i * p + j] = sum;
                }
            }

            return new Tensor(resultData, new[] { m, p }, a.RequiresGrad || b.RequiresGrad);
        }

        private Tensor AddBias(Tensor input, Tensor bias)
        {
            // input: [batch_size, out_features]
            // bias: [out_features]
            // Add bias to each output feature

            var resultData = new float[input.Size];
            var outFeatures = bias.Shape[0];
            int batchSize = input.Shape[0];

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
