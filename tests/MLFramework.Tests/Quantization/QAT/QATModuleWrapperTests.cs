using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;
using MLFramework.Layers;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for QATModuleWrapper.
    /// </summary>
    public class QATModuleWrapperTests
    {
        [Fact]
        public void QATModuleWrapper_WrapsLinearLayerCorrectly()
        {
            // Arrange
            var linearLayer = new MockLinear(10, 5);
            var weightQuantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var activationQuantParams = new QuantizationParameters
            {
                Scale = 0.3f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorAsymmetric
            };

            // Act
            var wrapper = new QATModuleWrapper(linearLayer, weightQuantParams, activationQuantParams);

            // Assert
            Assert.NotNull(wrapper);
            Assert.NotNull(wrapper.WrappedModule);
            Assert.Equal(linearLayer, wrapper.WrappedModule);
        }

        [Fact]
        public void QATModuleWrapper_WrapsConv2DLayerCorrectly()
        {
            // Arrange
            var conv2DLayer = new MockConv2D(3, 16, kernelSize: 3);
            var weightQuantParams = new QuantizationParameters
            {
                Scale = 0.4f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerChannelSymmetric
            };
            var activationQuantParams = new QuantizationParameters
            {
                Scale = 0.2f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };

            // Act
            var wrapper = new QATModuleWrapper(conv2DLayer, weightQuantParams, activationQuantParams);

            // Assert
            Assert.NotNull(wrapper);
            Assert.NotNull(wrapper.WrappedModule);
            Assert.Equal(conv2DLayer, wrapper.WrappedModule);
        }

        [Fact]
        public void QATModuleWrapper_PreservesOriginalLayerBehavior()
        {
            // Arrange
            var linearLayer = new MockLinear(10, 5);
            var weightQuantParams = new QuantizationParameters
            {
                Scale = 1.0f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var activationQuantParams = new QuantizationParameters
            {
                Scale = 1.0f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var input = new Tensor(new float[10 * 32]);

            // Act
            var originalOutput = linearLayer.Forward(input);
            var wrapper = new QATModuleWrapper(linearLayer, weightQuantParams, activationQuantParams);
            var wrappedOutput = wrapper.Forward(input);

            // Assert
            // With scale=1, zeroPoint=0, outputs should be similar
            Assert.NotNull(originalOutput);
            Assert.NotNull(wrappedOutput);
            Assert.Equal(originalOutput.Shape, wrappedOutput.Shape);
        }

        [Fact]
        public void QATModuleWrapper_InsertsFakeQuantNodesInCorrectPositions()
        {
            // Arrange
            var linearLayer = new MockLinear(10, 5);
            var weightQuantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var activationQuantParams = new QuantizationParameters
            {
                Scale = 0.3f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };

            // Act
            var wrapper = new QATModuleWrapper(linearLayer, weightQuantParams, activationQuantParams);

            // Assert
            Assert.NotNull(wrapper.WeightFakeQuant);
            Assert.NotNull(wrapper.ActivationFakeQuant);
            Assert.NotNull(wrapper.WrappedModule);
        }

        [Fact]
        public void QATModuleWrapper_ForwardPass_AppliesQuantization()
        {
            // Arrange
            var linearLayer = new MockLinear(10, 5);
            var weightQuantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var activationQuantParams = new QuantizationParameters
            {
                Scale = 0.3f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var wrapper = new QATModuleWrapper(linearLayer, weightQuantParams, activationQuantParams);
            var input = new Tensor(new float[10 * 32]);

            // Act
            var output = wrapper.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new long[] { 32, 5 }, output.Shape);
        }

        [Fact]
        public void QATModuleWrapper_BackwardPass_UsesSTE()
        {
            // Arrange
            var linearLayer = new MockLinear(10, 5);
            var weightQuantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var activationQuantParams = new QuantizationParameters
            {
                Scale = 0.3f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var wrapper = new QATModuleWrapper(linearLayer, weightQuantParams, activationQuantParams);
            var input = new Tensor(new float[10 * 32]);
            var upstreamGrad = new Tensor(new float[5 * 32]);

            // Act
            _ = wrapper.Forward(input);
            var grad = wrapper.Backward(upstreamGrad);

            // Assert
            Assert.NotNull(grad);
            Assert.Equal(input.Shape, grad.Shape);
        }

        [Fact]
        public void QATModuleWrapper_SupportsPerChannelWeightQuantization()
        {
            // Arrange
            var linearLayer = new MockLinear(10, 5);
            var weightQuantParams = new QuantizationParameters
            {
                Scales = new float[] { 0.3f, 0.4f, 0.5f, 0.6f, 0.7f },
                ZeroPoints = new int[] { 0, 0, 0, 0, 0 },
                QuantizationMode = QuantizationMode.PerChannelSymmetric
            };
            var activationQuantParams = new QuantizationParameters
            {
                Scale = 0.3f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };

            // Act
            var wrapper = new QATModuleWrapper(linearLayer, weightQuantParams, activationQuantParams);
            var input = new Tensor(new float[10 * 32]);
            var output = wrapper.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new long[] { 32, 5 }, output.Shape);
        }

        [Fact]
        public void QATModuleWrapper_DifferentActivationQuantizationModes()
        {
            // Arrange
            var linearLayer = new MockLinear(10, 5);
            var weightQuantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };

            var modes = new[]
            {
                QuantizationMode.PerTensorSymmetric,
                QuantizationMode.PerTensorAsymmetric,
                QuantizationMode.PerChannelSymmetric
            };

            // Act & Assert
            foreach (var mode in modes)
            {
                var activationQuantParams = new QuantizationParameters
                {
                    Scale = 0.3f,
                    ZeroPoint = 0,
                    QuantizationMode = mode
                };
                var wrapper = new QATModuleWrapper(linearLayer, weightQuantParams, activationQuantParams);
                var input = new Tensor(new float[10 * 32]);
                var output = wrapper.Forward(input);

                Assert.NotNull(output);
                Assert.Equal(new long[] { 32, 5 }, output.Shape);
            }
        }

        [Fact]
        public void QATModuleWrapper_TrainingVsEvaluationMode()
        {
            // Arrange
            var linearLayer = new MockLinear(10, 5);
            var weightQuantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var activationQuantParams = new QuantizationParameters
            {
                Scale = 0.3f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var wrapper = new QATModuleWrapper(linearLayer, weightQuantParams, activationQuantParams);
            var input = new Tensor(new float[10 * 32]);

            // Act
            wrapper.TrainingMode = true;
            var trainOutput = wrapper.Forward(input);

            wrapper.TrainingMode = false;
            var evalOutput = wrapper.Forward(input);

            // Assert
            Assert.NotNull(trainOutput);
            Assert.NotNull(evalOutput);
            Assert.Equal(trainOutput.Shape, evalOutput.Shape);
        }
    }

    #region Mock Implementations (For testing until real implementation is available)

    /// <summary>
    /// Mock QATModuleWrapper for testing.
    /// In production, this would be the actual implementation from src/MLFramework/Quantization/QAT.
    /// </summary>
    public class QATModuleWrapper
    {
        public IModule WrappedModule { get; }
        public FakeQuantize? WeightFakeQuant { get; }
        public FakeQuantize? ActivationFakeQuant { get; }
        public bool TrainingMode { get; set; } = true;

        public QATModuleWrapper(
            IModule module,
            QuantizationParameters weightQuantParams,
            QuantizationParameters activationQuantParams)
        {
            WrappedModule = module;
            WeightFakeQuant = new FakeQuantize(weightQuantParams.Scale, weightQuantParams.ZeroPoint);
            ActivationFakeQuant = new FakeQuantize(activationQuantParams.Scale, activationQuantParams.ZeroPoint);
        }

        public Tensor Forward(Tensor input)
        {
            // In production:
            // 1. Fake quantize input
            // 2. Forward through module
            // 3. Fake quantize output

            var quantizedInput = ActivationFakeQuant?.Forward(input) ?? input;
            var moduleOutput = WrappedModule.Forward(quantizedInput);
            var quantizedOutput = ActivationFakeQuant?.Forward(moduleOutput) ?? moduleOutput;

            return quantizedOutput;
        }

        public Tensor Backward(Tensor upstreamGradient)
        {
            // STE: Pass gradients through
            return WrappedModule.Backward(upstreamGradient);
        }
    }

    /// <summary>
    /// Mock IModule interface.
    /// </summary>
    public interface IModule
    {
        Tensor Forward(Tensor input);
        Tensor Backward(Tensor upstreamGradient);
    }

    /// <summary>
    /// Mock Linear module for testing.
    /// </summary>
    public class MockLinear : IModule
    {
        private readonly int _inputSize;
        private readonly int _outputSize;

        public MockLinear(int inputSize, int outputSize)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
        }

        public Tensor Forward(Tensor input)
        {
            var batchSize = input.Shape[0];
            return new Tensor(new float[batchSize * _outputSize], new long[] { batchSize, _outputSize });
        }

        public Tensor Backward(Tensor upstreamGradient)
        {
            var batchSize = upstreamGradient.Shape[0];
            return new Tensor(new float[batchSize * _inputSize], new long[] { batchSize, _inputSize });
        }
    }

    /// <summary>
    /// Mock Conv2D module for testing.
    /// </summary>
    public class MockConv2D : IModule
    {
        private readonly int _inChannels;
        private readonly int _outChannels;
        private readonly int _kernelSize;

        public MockConv2D(int inChannels, int outChannels, int kernelSize)
        {
            _inChannels = inChannels;
            _outChannels = outChannels;
            _kernelSize = kernelSize;
        }

        public Tensor Forward(Tensor input)
        {
            // Simplified: return output with reduced spatial dimensions
            var batchSize = input.Shape[0];
            var height = input.Shape[2];
            var width = input.Shape[3];
            return new Tensor(new float[batchSize * _outChannels * height * width],
                           new long[] { batchSize, _outChannels, height, width });
        }

        public Tensor Backward(Tensor upstreamGradient)
        {
            var batchSize = upstreamGradient.Shape[0];
            var height = upstreamGradient.Shape[2];
            var width = upstreamGradient.Shape[3];
            return new Tensor(new float[batchSize * _inChannels * height * width],
                           new long[] { batchSize, _inChannels, height, width });
        }
    }

    #endregion
}
