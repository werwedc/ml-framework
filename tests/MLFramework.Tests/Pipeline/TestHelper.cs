using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.NN;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    /// <summary>
    /// Helper class with utility methods for pipeline testing
    /// </summary>
    public static class TestHelper
    {
        private static readonly Random _random = new Random(42); // Fixed seed for reproducibility

        /// <summary>
        /// Create a simple MLP model for testing
        /// </summary>
        public static Module CreateSimpleMLP(int inputSize, int hiddenSize, int outputSize)
        {
            var model = new SequentialModule("SimpleMLP");

            // Hidden layer 1
            model.Add(new Linear(inputSize, hiddenSize, "linear1"));

            // Hidden layer 2
            model.Add(new Linear(hiddenSize, hiddenSize, "linear2"));

            // Output layer
            model.Add(new Linear(hiddenSize, outputSize, "linear3"));

            return model;
        }

        /// <summary>
        /// Create a simple CNN model for testing
        /// </summary>
        public static Module CreateSimpleCNN(int inputChannels, int numClasses)
        {
            var model = new SequentialModule("SimpleCNN");

            // Conv layer 1
            model.Add(new TestConv2d(inputChannels, 32, kernelSize: 3, name: "conv1"));

            // Conv layer 2
            model.Add(new TestConv2d(32, 64, kernelSize: 3, name: "conv2"));

            // Flatten and FC layer
            model.Add(new Linear(64 * 28 * 28, numClasses, "fc"));

            return model;
        }

        /// <summary>
        /// Create dummy input tensor with random data
        /// </summary>
        public static Tensor CreateDummyInput(int batchSize, int inputSize, IDevice device)
        {
            var data = new float[batchSize * inputSize];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)_random.NextDouble() * 2.0f - 1.0f; // Random in [-1, 1]
            }
            return new Tensor(data, new[] { (long)batchSize, (long)inputSize });
        }

        /// <summary>
        /// Create dummy 4D input tensor for CNN (NCHW format)
        /// </summary>
        public static Tensor CreateDummyInput4D(int batchSize, int channels, int height, int width, IDevice device)
        {
            var data = new float[batchSize * channels * height * width];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)_random.NextDouble() * 2.0f - 1.0f;
            }
            return new Tensor(data, new[] { (long)batchSize, (long)channels, (long)height, (long)width });
        }

        /// <summary>
        /// Compare two tensors (assert they match within tolerance)
        /// </summary>
        public static void AssertTensorClose(Tensor a, Tensor b, float tolerance = 1e-4f)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            Assert.Equal(a.Shape, b.Shape);

            var aData = a.Data;
            var bData = b.Data;

            Assert.Equal(aData.Length, bData.Length);

            for (int i = 0; i < aData.Length; i++)
            {
                float diff = Math.Abs(aData[i] - bData[i]);
                Assert.True(diff < tolerance,
                    $"Tensors differ at index {i}: {aData[i]} vs {bData[i]}, diff = {diff}");
            }
        }

        /// <summary>
        /// Measure execution time of an action
        /// </summary>
        public static long MeasureExecutionTime(Action action)
        {
            var stopwatch = Stopwatch.StartNew();
            action();
            stopwatch.Stop();
            return stopwatch.ElapsedMilliseconds;
        }

        /// <summary>
        /// Measure execution time of an async action
        /// </summary>
        public static async Task<long> MeasureExecutionTimeAsync(Func<Task> action)
        {
            var stopwatch = Stopwatch.StartNew();
            await action();
            stopwatch.Stop();
            return stopwatch.ElapsedMilliseconds;
        }

        /// <summary>
        /// Get all parameters from a module as a flat list
        /// </summary>
        public static List<Parameter> GetAllParameters(Module module)
        {
            return module.GetParameters().ToList();
        }

        /// <summary>
        /// Check if a tensor contains NaN or infinity
        /// </summary>
        public static bool ContainsNaNOrInf(Tensor tensor)
        {
            var data = tensor.Data;
            foreach (float val in data)
            {
                if (float.IsNaN(val) || float.IsInfinity(val))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Set a fixed random seed for reproducibility
        /// </summary>
        public static void SetRandomSeed(int seed)
        {
            _random = new Random(seed);
        }
    }

    /// <summary>
    /// Test implementation of Linear layer
    /// </summary>
    public class Linear : Module
    {
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly Parameter _weight;
        private readonly Parameter _bias;

        public int InputSize => _inputSize;
        public int OutputSize => _outputSize;

        public Linear(int inputSize, int outputSize, string name = "Linear") : base(name)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;

            // Create parameters with zeros for testing
            _weight = new Parameter(Tensor.Zeros(new long[] { outputSize, inputSize }), "weight");
            _bias = new Parameter(Tensor.Zeros(new long[] { outputSize }), "bias");
        }

        public override Tensor Forward(Tensor input)
        {
            // Simplified forward pass for testing (just return input)
            return input;
        }

        public override IEnumerable<Parameter> GetParameters()
        {
            yield return _weight;
            yield return _bias;
        }

        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            yield return ("weight", _weight);
            yield return ("bias", _bias);
        }
    }

    /// <summary>
    /// Test implementation of Conv2d layer
    /// </summary>
    public class TestConv2d : Module
    {
        private readonly int _inChannels;
        private readonly int _outChannels;
        private readonly int _kernelSize;
        private readonly Parameter _weight;
        private readonly Parameter _bias;

        public int InChannels => _inChannels;
        public int OutChannels => _outChannels;
        public int KernelSize => _kernelSize;

        public TestConv2d(int inChannels, int outChannels, int kernelSize, string name = "Conv2d") : base(name)
        {
            _inChannels = inChannels;
            _outChannels = outChannels;
            _kernelSize = kernelSize;

            // Create parameters
            long weightShape[] = { outChannels, inChannels, kernelSize, kernelSize };
            _weight = new Parameter(Tensor.Zeros(weightShape), "weight");
            _bias = new Parameter(Tensor.Zeros(new long[] { outChannels }), "bias");
        }

        public override Tensor Forward(Tensor input)
        {
            // Simplified forward pass for testing
            return input;
        }

        public override IEnumerable<Parameter> GetParameters()
        {
            yield return _weight;
            yield return _bias;
        }

        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            yield return ("weight", _weight);
            yield return ("bias", _bias);
        }
    }
}
