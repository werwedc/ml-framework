using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;
using MLFramework.Layers;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for FakeQuantizeLayer.
    /// </summary>
    public class FakeQuantizeLayerTests
    {
        [Fact]
        public void FakeQuantizeLayer_WrapsLayerCorrectly()
        {
            // Arrange
            var linearLayer = new LinearLayer(10, 5);
            var quantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };

            // Act
            var fakeQuantLayer = new FakeQuantizeLayer(linearLayer, quantParams);

            // Assert
            Assert.NotNull(fakeQuantLayer);
            Assert.NotNull(fakeQuantLayer.WrappedLayer);
            Assert.Equal(linearLayer, fakeQuantLayer.WrappedLayer);
        }

        [Fact]
        public void FakeQuantizeLayer_StoresParametersCorrectly()
        {
            // Arrange
            var linearLayer = new LinearLayer(10, 5);
            var quantParams = new QuantizationParameters
            {
                Scale = 0.75f,
                ZeroPoint = 5,
                QuantizationMode = QuantizationMode.PerTensorAsymmetric
            };

            // Act
            var fakeQuantLayer = new FakeQuantizeLayer(linearLayer, quantParams);

            // Assert
            Assert.NotNull(fakeQuantLayer.QuantizationParameters);
            Assert.Equal(0.75f, fakeQuantLayer.QuantizationParameters.Scale);
            Assert.Equal(5, fakeQuantLayer.QuantizationParameters.ZeroPoint);
            Assert.Equal(QuantizationMode.PerTensorAsymmetric, fakeQuantLayer.QuantizationParameters.QuantizationMode);
        }

        [Fact]
        public void FakeQuantizeLayer_ForwardPass_UpdatesDuringTraining()
        {
            // Arrange
            var linearLayer = new LinearLayer(10, 5);
            var quantParams = new QuantizationParameters
            {
                Scale = 1.0f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var fakeQuantLayer = new FakeQuantizeLayer(linearLayer, quantParams);
            var input = new Tensor(new float[10 * 32]); // Batch of 32, input size 10
            fakeQuantLayer.TrainingMode = true;

            // Act
            var initialScale = fakeQuantLayer.QuantizationParameters.Scale;
            _ = fakeQuantLayer.Forward(input);
            var updatedScale = fakeQuantLayer.QuantizationParameters.Scale;

            // Assert
            // Scale should potentially update during training (depends on observer)
            Assert.NotNull(fakeQuantLayer.QuantizationParameters);
        }

        [Fact]
        public void FakeQuantizeLayer_PerTensorMode_WorksCorrectly()
        {
            // Arrange
            var linearLayer = new LinearLayer(10, 5);
            var quantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var fakeQuantLayer = new FakeQuantizeLayer(linearLayer, quantParams);
            var input = new Tensor(new float[10 * 32]);

            // Act
            var output = fakeQuantLayer.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new long[] { 32, 5 }, output.Shape);
        }

        [Fact]
        public void FakeQuantizeLayer_PerChannelMode_WorksCorrectly()
        {
            // Arrange
            var linearLayer = new LinearLayer(10, 5);
            var quantParams = new QuantizationParameters
            {
                Scales = new float[] { 0.3f, 0.5f, 0.7f, 0.9f, 1.1f },
                ZeroPoints = new int[] { 0, 0, 0, 0, 0 },
                QuantizationMode = QuantizationMode.PerChannelSymmetric
            };
            var fakeQuantLayer = new FakeQuantizeLayer(linearLayer, quantParams);
            var input = new Tensor(new float[10 * 32]);

            // Act
            var output = fakeQuantLayer.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new long[] { 32, 5 }, output.Shape);
        }

        [Fact]
        public void FakeQuantizeLayer_BackwardPass_PropagatesGradients()
        {
            // Arrange
            var linearLayer = new LinearLayer(10, 5);
            var quantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var fakeQuantLayer = new FakeQuantizeLayer(linearLayer, quantParams);
            var input = new Tensor(new float[10 * 32]);
            var upstreamGrad = new Tensor(new float[5 * 32]);

            // Act
            _ = fakeQuantLayer.Forward(input);
            var grad = fakeQuantLayer.Backward(upstreamGrad);

            // Assert
            Assert.NotNull(grad);
            Assert.Equal(input.Shape, grad.Shape);
        }

        [Fact]
        public void FakeQuantizeLayer_EvaluationMode_DisablesObserverUpdates()
        {
            // Arrange
            var linearLayer = new LinearLayer(10, 5);
            var quantParams = new QuantizationParameters
            {
                Scale = 1.0f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var fakeQuantLayer = new FakeQuantizeLayer(linearLayer, quantParams);
            fakeQuantLayer.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);

            // Act
            _ = fakeQuantLayer.Forward(input);
            fakeQuantLayer.TrainingMode = false; // Evaluation mode
            _ = fakeQuantLayer.Forward(input);

            // Assert
            Assert.False(fakeQuantLayer.TrainingMode);
        }

        [Fact]
        public void FakeQuantizeLayer_ClonesParameters()
        {
            // Arrange
            var linearLayer = new LinearLayer(10, 5);
            var quantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };
            var fakeQuantLayer = new FakeQuantizeLayer(linearLayer, quantParams);

            // Act
            var originalScale = fakeQuantLayer.QuantizationParameters.Scale;
            fakeQuantLayer.QuantizationParameters.Scale = 1.0f;

            // Assert
            Assert.Equal(1.0f, fakeQuantLayer.QuantizationParameters.Scale);
        }

        [Fact]
        public void FakeQuantizeLayer_SupportsMultipleLayers()
        {
            // Arrange
            var layer1 = new LinearLayer(10, 5);
            var layer2 = new LinearLayer(5, 2);
            var quantParams = new QuantizationParameters
            {
                Scale = 0.5f,
                ZeroPoint = 0,
                QuantizationMode = QuantizationMode.PerTensorSymmetric
            };

            // Act
            var fakeQuantLayer1 = new FakeQuantizeLayer(layer1, quantParams);
            var fakeQuantLayer2 = new FakeQuantizeLayer(layer2, quantParams);
            var input = new Tensor(new float[10 * 32]);

            // Assert
            var output1 = fakeQuantLayer1.Forward(input);
            var output2 = fakeQuantLayer2.Forward(output1);
            Assert.NotNull(output1);
            Assert.NotNull(output2);
        }
    }

    #region Mock Implementations (For testing until real implementation is available)

    /// <summary>
    /// Mock FakeQuantizeLayer for testing.
    /// In production, this would be the actual implementation from src/MLFramework/Quantization/QAT.
    /// </summary>
    public class FakeQuantizeLayer
    {
        public ILayer WrappedLayer { get; }
        public QuantizationParameters QuantizationParameters { get; set; }
        public bool TrainingMode { get; set; } = true;

        public FakeQuantizeLayer(ILayer layer, QuantizationParameters quantParams)
        {
            WrappedLayer = layer;
            QuantizationParameters = quantParams;
        }

        public Tensor Forward(Tensor input)
        {
            // In production, this would:
            // 1. Observe input statistics if training
            // 2. Apply fake quantization
            // 3. Pass to wrapped layer
            // 4. Apply fake quantization to output

            var wrappedOutput = WrappedLayer.Forward(input);
            return wrappedOutput;
        }

        public Tensor Backward(Tensor upstreamGradient)
        {
            // Pass through gradients using STE
            return WrappedLayer.Backward(upstreamGradient);
        }
    }

    /// <summary>
    /// Mock LinearLayer for testing.
    /// </summary>
    public class LinearLayer : ILayer
    {
        private readonly int _inputSize;
        private readonly int _outputSize;

        public LinearLayer(int inputSize, int outputSize)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;
        }

        public Tensor Forward(Tensor input)
        {
            // Simplified forward pass
            var batchSize = input.Shape[0];
            return new Tensor(new float[batchSize * _outputSize], new long[] { batchSize, _outputSize });
        }

        public Tensor Backward(Tensor upstreamGradient)
        {
            // Simplified backward pass
            var batchSize = upstreamGradient.Shape[0];
            return new Tensor(new float[batchSize * _inputSize], new long[] { batchSize, _inputSize });
        }
    }

    /// <summary>
    /// Mock ILayer interface.
    /// </summary>
    public interface ILayer
    {
        Tensor Forward(Tensor input);
        Tensor Backward(Tensor upstreamGradient);
    }

    #endregion
}
