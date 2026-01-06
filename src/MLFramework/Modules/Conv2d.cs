using RitterFramework.Core.Tensor;
using MLFramework.LoRA;

namespace MLFramework.Modules
{
    /// <summary>
    /// A 2D convolutional layer
    /// </summary>
    public class Conv2d : IModule
    {
        public Tensor Weight { get; set; }
        public Tensor? Bias { get; set; }

        public int InChannels { get; }
        public int OutChannels { get; }
        public int KernelSize { get; }
        public int Stride { get; }
        public int Padding { get; }

        public string ModuleType => "Conv2d";

        public bool IsTraining { get; set; } = false;

        /// <summary>
        /// Creates a new Conv2d layer
        /// </summary>
        /// <param name="inChannels">Number of input channels</param>
        /// <param name="outChannels">Number of output channels</param>
        /// <param name="kernelSize">Size of the convolutional kernel</param>
        /// <param name="stride">Stride of the convolution</param>
        /// <param name="padding">Padding added to all sides</param>
        /// <param name="useBias">Whether to use bias</param>
        public Conv2d(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, bool useBias = true)
        {
            if (inChannels <= 0)
                throw new ArgumentException("inChannels must be positive", nameof(inChannels));
            if (outChannels <= 0)
                throw new ArgumentException("outChannels must be positive", nameof(outChannels));
            if (kernelSize <= 0)
                throw new ArgumentException("kernelSize must be positive", nameof(kernelSize));

            InChannels = inChannels;
            OutChannels = outChannels;
            KernelSize = kernelSize;
            Stride = stride;
            Padding = padding;

            // Initialize weights: [outChannels, inChannels, kernelSize, kernelSize]
            Weight = InitializeWeight(new[] { outChannels, inChannels, kernelSize, kernelSize });

            if (useBias)
            {
                Bias = Tensor.Zeros(new[] { outChannels });
            }
        }

        /// <summary>
        /// Creates a Conv2d layer from existing weights
        /// </summary>
        public Conv2d(Tensor weight, Tensor? bias = null, int stride = 1, int padding = 0)
        {
            Weight = weight ?? throw new ArgumentNullException(nameof(weight));

            if (weight.Shape.Length != 4)
                throw new ArgumentException("Weight must be 4-dimensional [outC, inC, kH, kW]", nameof(weight));

            InChannels = weight.Shape[1];
            OutChannels = weight.Shape[0];
            KernelSize = weight.Shape[2];
            if (weight.Shape[3] != KernelSize)
                throw new ArgumentException("Kernel height and width must be equal");
            Stride = stride;
            Padding = padding;
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
        /// Forward pass: 2D convolution
        /// </summary>
        public Tensor Forward(Tensor input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));

            // Input shape: [batch_size, in_channels, height, width]
            // Weight shape: [out_channels, in_channels, kernel_size, kernel_size]
            // Output shape: [batch_size, out_channels, out_height, out_width]

            int batchSize = input.Shape[0];
            int inChannels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];

            int outHeight = (height + 2 * Padding - KernelSize) / Stride + 1;
            int outWidth = (width + 2 * Padding - KernelSize) / Stride + 1;

            // Extract patches using im2col
            var patches = Im2Col(input, KernelSize, Padding, Stride);

            // Reshape weight: [outC, inC, kH, kW] -> [outC, inC*kH*kW]
            var weightFlat = Weight.Reshape(new[] { OutChannels, InChannels * KernelSize * KernelSize });

            // Matrix multiplication: patches @ weightFlat.T
            // patches: [batch * out_h*out_w, inC*kH*kW]
            // weightFlat: [outC, inC*kH*kW]
            // outputFlat: [batch * out_h*out_w, outC]
            var weightFlatTransposed = weightFlat.Transpose();
            var outputFlat = MatMul(patches, weightFlatTransposed);

            // Reshape output: [batch * out_h*out_w, outC] -> [batch, out_h, out_w, outC] -> [batch, outC, out_h, out_w]
            var output = outputFlat.Reshape(new[] { batchSize, outHeight, outWidth, OutChannels })
                                  .Reshape(new[] { batchSize, OutChannels, outHeight, outWidth });

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
            // Kaiming initialization for convolutional layers
            // std = sqrt(2 / (fan_in))
            int fanIn = shape[1] * shape[2] * shape[3]; // inChannels * kH * kW
            float std = (float)Math.Sqrt(2.0 / fanIn);

            var random = new Random(42);
            var data = new float[shape.Aggregate(1, (x, y) => x * y)];

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

        private Tensor Im2Col(Tensor input, int kernelSize, int padding, int stride)
        {
            int batchSize = input.Shape[0];
            int channels = input.Shape[1];
            int height = input.Shape[2];
            int width = input.Shape[3];

            int outHeight = (height + 2 * padding - kernelSize) / stride + 1;
            int outWidth = (width + 2 * padding - kernelSize) / stride + 1;

            int numPositions = outHeight * outWidth;
            int patchSize = channels * kernelSize * kernelSize;

            // Allocate output tensor as 2D: [batch * out_h * out_w, in_c * k * k]
            // This allows efficient matrix multiplication with the weight matrix
            var output = Tensor.Zeros(new[] { batchSize * numPositions, patchSize });

            // Extract patches directly into 2D format
            for (int b = 0; b < batchSize; b++)
            {
                for (int oy = 0; oy < outHeight; oy++)
                {
                    for (int ox = 0; ox < outWidth; ox++)
                    {
                        int positionIndex = b * numPositions + oy * outWidth + ox;

                        for (int c = 0; c < channels; c++)
                        {
                            for (int ky = 0; ky < kernelSize; ky++)
                            {
                                for (int kx = 0; kx < kernelSize; kx++)
                                {
                                    int iy = oy * stride - padding + ky;
                                    int ix = ox * stride - padding + kx;

                                    int patchIndex = (c * kernelSize + ky) * kernelSize + kx;

                                    if (iy >= 0 && iy < height && ix >= 0 && ix < width)
                                    {
                                        output[new[] { positionIndex, patchIndex }] = input[new[] { b, c, iy, ix }];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return output;
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
            // input: [batch_size, out_channels, out_height, out_width]
            // bias: [out_channels]
            // Add bias to each channel

            var resultData = new float[input.Size];
            var outChannels = bias.Shape[0];
            int batchSize = input.Shape[0];
            int outHeight = input.Shape[2];
            int outWidth = input.Shape[3];

            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < outChannels; c++)
                {
                    for (int h = 0; h < outHeight; h++)
                    {
                        for (int w = 0; w < outWidth; w++)
                        {
                            resultData[((b * outChannels + c) * outHeight + h) * outWidth + w] =
                                input[new[] { b, c, h, w }] + bias[new[] { c }];
                        }
                    }
                }
            }

            return new Tensor(resultData, input.Shape, input.RequiresGrad || bias.RequiresGrad);
        }
    }
}
