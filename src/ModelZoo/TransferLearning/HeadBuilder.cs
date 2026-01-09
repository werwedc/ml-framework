using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;
using MLFramework.NN;

namespace MLFramework.ModelZoo.TransferLearning
{
    /// <summary>
    /// Simple linear layer for head construction.
    /// This is a placeholder implementation that can be replaced with a full Linear layer implementation.
    /// </summary>
    public class LinearHead : Module
    {
        private readonly Parameter _weight;
        private readonly Parameter _bias;
        private readonly int _inputDim;
        private readonly int _outputDim;

        /// <summary>
        /// Gets the weight parameter.
        /// </summary>
        public Parameter Weight => _weight;

        /// <summary>
        /// Gets the bias parameter.
        /// </summary>
        public Parameter Bias => _bias;

        /// <summary>
        /// Creates a new linear head.
        /// </summary>
        public LinearHead(int inputDim, int outputDim, bool useBias = true, string name = "linear")
            : base(name)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;

            // Initialize weights using Xavier initialization
            float std = (float)Math.Sqrt(2.0 / (inputDim + outputDim));
            var weightData = new float[inputDim * outputDim];
            for (int i = 0; i < weightData.Length; i++)
            {
                weightData[i] = NormalRandom() * std;
            }

            _weight = new Parameter(weightData, new[] { outputDim, inputDim }, "weight");

            if (useBias)
            {
                var biasData = new float[outputDim];
                _bias = new Parameter(biasData, new[] { outputDim }, "bias");
            }
            else
            {
                _bias = null;
            }
        }

        public override Tensor Forward(Tensor input)
        {
            // Placeholder implementation - actual implementation would perform matrix multiplication
            // For now, return a dummy tensor
            var outputData = new float[_outputDim];
            for (int i = 0; i < outputData.Length; i++)
            {
                outputData[i] = 0.0f;
            }

            return new Tensor(outputData, new[] { _outputDim });
        }

        public override IEnumerable<Parameter> GetParameters()
        {
            yield return _weight;
            if (_bias != null)
                yield return _bias;
        }

        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            yield return ("weight", _weight);
            if (_bias != null)
                yield return ("bias", _bias);
        }

        private static float NormalRandom()
        {
            // Box-Muller transform for generating normal distribution
            var rand = new Random();
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return (float)z0;
        }

        /// <summary>
        /// Reinitializes the weights with the given strategy.
        /// </summary>
        public void InitializeWeights(WeightInitializationStrategy strategy)
        {
            var (gain, std) = strategy.GetDefaultParameters();
            float scale = gain * std;

            var weightData = _weight.Data;
            for (int i = 0; i < weightData.Length; i++)
            {
                weightData[i] = NormalRandom() * scale;
            }

            if (_bias != null)
            {
                var biasData = _bias.Data;
                for (int i = 0; i < biasData.Length; i++)
                {
                    biasData[i] = 0.0f; // Bias typically initialized to zero
                }
            }
        }
    }

    /// <summary>
    /// Multi-layer perceptron head.
    /// </summary>
    public class MLPHead : Module
    {
        private readonly List<Module> _layers;
        private readonly SequentialModule _sequential;

        /// <summary>
        /// Creates a new MLP head.
        /// </summary>
        public MLPHead(int inputDim, int[] hiddenDims, int outputDim, string name = "mlp_head")
            : base(name)
        {
            _layers = new List<Module>();
            _sequential = new SequentialModule(name);

            int currentDim = inputDim;
            for (int i = 0; i < hiddenDims.Length; i++)
            {
                var layer = new LinearHead(currentDim, hiddenDims[i], true, $"layer_{i}");
                _layers.Add(layer);
                _sequential.Add(layer);
                currentDim = hiddenDims[i];
            }

            var outputLayer = new LinearHead(currentDim, outputDim, true, "output");
            _layers.Add(outputLayer);
            _sequential.Add(outputLayer);
        }

        public override Tensor Forward(Tensor input)
        {
            return _sequential.Forward(input);
        }

        public override IEnumerable<Parameter> GetParameters()
        {
            return _sequential.GetParameters();
        }

        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            return _sequential.GetNamedParameters();
        }

        /// <summary>
        /// Reinitializes all weights in the MLP.
        /// </summary>
        public void InitializeWeights(WeightInitializationStrategy strategy)
        {
            foreach (var layer in _layers)
            {
                if (layer is LinearHead linear)
                {
                    linear.InitializeWeights(strategy);
                }
            }
        }
    }

    /// <summary>
    /// Builder class for creating common head architectures.
    /// </summary>
    public static class HeadBuilder
    {
        /// <summary>
        /// Creates a linear classification head with optional dropout.
        /// </summary>
        /// <param name="inputDim">Input dimension.</param>
        /// <param name="numClasses">Number of output classes.</param>
        /// <param name="includeDropout">Whether to include dropout.</param>
        /// <param name="dropoutRate">Dropout rate.</param>
        /// <returns>A linear head module.</returns>
        public static Module LinearHead(int inputDim, int numClasses, bool includeDropout = false, float dropoutRate = 0.5f)
        {
            var linear = new LinearHead(inputDim, numClasses, true, "linear_head");

            if (includeDropout)
            {
                // For now, dropout is not implemented as a module
                // This is a placeholder for future dropout implementation
                return linear;
            }

            return linear;
        }

        /// <summary>
        /// Creates a multi-layer perceptron head.
        /// </summary>
        /// <param name="inputDim">Input dimension.</param>
        /// <param name="hiddenDims">Array of hidden layer dimensions.</param>
        /// <param name="outputDim">Output dimension.</param>
        /// <returns>An MLP head module.</returns>
        public static Module MLPHead(int inputDim, int[] hiddenDims, int outputDim)
        {
            return new MLPHead(inputDim, hiddenDims, outputDim);
        }

        /// <summary>
        /// Creates a convolutional head.
        /// </summary>
        /// <param name="inputChannels">Number of input channels.</param>
        /// <param name="numClasses">Number of output classes.</param>
        /// <param name="kernelSizes">Array of kernel sizes for conv layers.</param>
        /// <returns>A placeholder convolutional head module.</returns>
        /// <remarks>
        /// This is a placeholder implementation that returns a linear head.
        /// Full implementation requires Conv2d, BatchNorm, and Pooling modules.
        /// </remarks>
        public static Module ConvHead(int inputChannels, int numClasses, int[] kernelSizes = null)
        {
            // Placeholder: return a linear head for now
            // A full implementation would create a sequence of Conv2d, BatchNorm, ReLU, Pooling layers
            var hiddenSize = inputChannels * 16; // Arbitrary scaling
            return new LinearHead(hiddenSize, numClasses, true, "conv_head_placeholder");
        }

        /// <summary>
        /// Creates a head with global average pooling followed by a linear layer.
        /// </summary>
        /// <param name="outputSize">Size after pooling.</param>
        /// <param name="numClasses">Number of output classes.</param>
        /// <returns>A head module with adaptive pooling.</returns>
        /// <remarks>
        /// This is a placeholder implementation.
        /// Full implementation requires AdaptiveAvgPool2d module.
        /// </remarks>
        public static Module AdaptiveAvgPoolHead(int outputSize, int numClasses)
        {
            // Placeholder: return a linear head for now
            // A full implementation would have AdaptiveAvgPool2d followed by Linear
            return new LinearHead(outputSize, numClasses, true, "adaptive_pool_head");
        }

        /// <summary>
        /// Creates an attention-based head.
        /// </summary>
        /// <param name="inputDim">Input dimension.</param>
        /// <param name="numClasses">Number of output classes.</param>
        /// <param name="numHeads">Number of attention heads.</param>
        /// <returns>An attention head module.</returns>
        /// <remarks>
        /// This is a placeholder implementation.
        /// Full implementation requires MultiHeadAttention module.
        /// </remarks>
        public static Module AttentionHead(int inputDim, int numClasses, int numHeads = 8)
        {
            // Placeholder: return an MLP head for now
            // A full implementation would use MultiHeadAttention followed by projection layers
            return MLPHead(inputDim, new[] { inputDim / 2 }, numClasses);
        }

        /// <summary>
        /// Creates a head and initializes weights with the given strategy.
        /// </summary>
        /// <param name="head">The head module to initialize.</param>
        /// <param name="strategy">Initialization strategy.</param>
        public static void InitializeHead(Module head, WeightInitializationStrategy strategy)
        {
            if (head == null)
                throw new ArgumentNullException(nameof(head));

            if (head is LinearHead linear)
            {
                linear.InitializeWeights(strategy);
            }
            else if (head is MLPHead mlp)
            {
                mlp.InitializeWeights(strategy);
            }
            else
            {
                // For other module types, recursively initialize all parameters
                foreach (var param in head.GetParameters())
                {
                    // Reinitialize parameter data using the strategy
                    var (gain, std) = strategy.GetDefaultParameters();
                    float scale = gain * std;

                    for (int i = 0; i < param.Data.Length; i++)
                    {
                        param.Data[i] = NormalRandom() * scale;
                    }
                }
            }
        }

        private static float NormalRandom()
        {
            var rand = new Random();
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return (float)z0;
        }

        /// <summary>
        /// Creates a default head for common transfer learning scenarios.
        /// </summary>
        /// <param name="inputDim">Input dimension.</param>
        /// <param name="numClasses">Number of target classes.</param>
        /// <param name="headType">Type of head to create ("linear", "mlp", "conv", "attention").</param>
        /// <returns>A head module.</returns>
        public static Module CreateDefaultHead(int inputDim, int numClasses, string headType = "linear")
        {
            return headType.ToLowerInvariant() switch
            {
                "linear" => LinearHead(inputDim, numClasses),
                "mlp" => MLPHead(inputDim, new[] { inputDim / 2, inputDim / 4 }, numClasses),
                "conv" => ConvHead(inputDim, numClasses),
                "attention" => AttentionHead(inputDim, numClasses),
                _ => throw new ArgumentException($"Unknown head type: {headType}")
            };
        }
    }
}
