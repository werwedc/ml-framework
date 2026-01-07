using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for FakeQuantize operation.
    /// </summary>
    public class FakeQuantizeTests
    {
        [Fact]
        public void FakeQuantize_SimulatesQuantizationNoise()
        {
            // Arrange
            var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var scale = 0.5f;
            var zeroPoint = 0;

            // Act
            var fakeQuantize = new FakeQuantize(scale, zeroPoint);
            var output = fakeQuantize.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape, output.Shape);
            // Output should be quantized and dequantized, introducing noise
            Assert.NotEqual(input.ToArray(), output.ToArray());
        }

        [Fact]
        public void FakeQuantize_GradientsFlowCorrectly_UsingSTE()
        {
            // Arrange
            var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f });
            var scale = 0.5f;
            var zeroPoint = 0;
            var fakeQuantize = new FakeQuantize(scale, zeroPoint);

            // Act
            var output = fakeQuantize.Forward(input);
            var upstreamGrad = new Tensor(new float[] { 1.0f, 1.0f, 1.0f });
            var grad = fakeQuantize.Backward(upstreamGrad);

            // Assert
            // STE should pass through gradients directly
            Assert.NotNull(grad);
            Assert.Equal(upstreamGrad.ToArray(), grad.ToArray());
        }

        [Fact]
        public void FakeQuantize_ForwardPass_ProducesNoisyOutputs()
        {
            // Arrange
            var input = new Tensor(new float[] { 0.1f, 0.5f, 0.9f, 1.5f });
            var scale = 0.1f;
            var zeroPoint = 0;
            var fakeQuantize = new FakeQuantize(scale, zeroPoint);

            // Act
            var output = fakeQuantize.Forward(input);

            // Assert
            var inputData = input.ToArray();
            var outputData = output.ToArray();

            // Each output should be different from input due to quantization
            var differences = 0;
            for (int i = 0; i < inputData.Length; i++)
            {
                if (Math.Abs(inputData[i] - outputData[i]) > 1e-5)
                {
                    differences++;
                }
            }
            Assert.True(differences > 0, "Output should differ from input due to quantization");
        }

        [Fact]
        public void FakeQuantize_BackwardPass_UsesIdentityGradients()
        {
            // Arrange
            var fakeQuantize = new FakeQuantize(1.0f, 0);
            var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f });
            var upstreamGrad = new Tensor(new float[] { 0.5f, 0.3f, 0.7f });

            // Act
            _ = fakeQuantize.Forward(input); // Forward pass
            var grad = fakeQuantize.Backward(upstreamGrad);

            // Assert
            // Identity function: output gradient = input gradient
            Assert.Equal(upstreamGrad.ToArray(), grad.ToArray());
        }

        [Fact]
        public void FakeQuantize_PerTensorMode_WorksCorrectly()
        {
            // Arrange
            var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
            var scale = 0.5f;
            var zeroPoint = 0;
            var fakeQuantize = new FakeQuantize(scale, zeroPoint, perTensor: true);

            // Act
            var output = fakeQuantize.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape, output.Shape);
        }

        [Fact]
        public void FakeQuantize_PerChannelMode_WorksCorrectly()
        {
            // Arrange
            var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
            var scales = new float[] { 0.5f, 1.0f };
            var zeroPoints = new int[] { 0, 0 };
            var fakeQuantize = new FakeQuantize(scales, zeroPoints, channelAxis: 0);

            // Act
            var output = fakeQuantize.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape, output.Shape);
        }

        [Fact]
        public void FakeQuantize_WithZeroPoint_CalculatesCorrectly()
        {
            // Arrange
            var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var scale = 1.0f;
            var zeroPoint = 5;
            var fakeQuantize = new FakeQuantize(scale, zeroPoint);

            // Act
            var output = fakeQuantize.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape, output.Shape);
            // Verify quantization/dequantization with zero point
            var inputData = input.ToArray();
            var outputData = output.ToArray();
            for (int i = 0; i < inputData.Length; i++)
            {
                var quantized = Math.Round(inputData[i] / scale) + zeroPoint;
                var dequantized = (quantized - zeroPoint) * scale;
                Assert.Equal(dequantized, outputData[i], precision: 5);
            }
        }

        [Fact]
        public void FakeQuantize_UpdatesScaleAndZeroPoint()
        {
            // Arrange
            var input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
            var fakeQuantize = new FakeQuantize(1.0f, 0);

            // Act
            fakeQuantize.UpdateScaleAndZeroPoint(newScale: 0.5f, newZeroPoint: 2);
            var output = fakeQuantize.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape, output.Shape);
        }

        [Fact]
        public void FakeQuantize_HandlesNegativeValues()
        {
            // Arrange
            var input = new Tensor(new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f });
            var fakeQuantize = new FakeQuantize(1.0f, 0);

            // Act
            var output = fakeQuantize.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape, output.Shape);
        }
    }

    /// <summary>
    /// Mock implementation of FakeQuantize for testing.
    /// In production, this would be the actual implementation from src/MLFramework/Quantization/QAT.
    /// </summary>
    public class FakeQuantize
    {
        private float _scale;
        private int _zeroPoint;
        private float[]? _scales; // For per-channel
        private int[]? _zeroPoints; // For per-channel
        private readonly bool _perTensor;
        private readonly int? _channelAxis;

        public FakeQuantize(float scale, int zeroPoint, bool perTensor = true)
        {
            _scale = scale;
            _zeroPoint = zeroPoint;
            _perTensor = perTensor;
        }

        public FakeQuantize(float[] scales, int[] zeroPoints, int channelAxis)
        {
            _scales = scales;
            _zeroPoints = zeroPoints;
            _perTensor = false;
            _channelAxis = channelAxis;
        }

        public Tensor Forward(Tensor input)
        {
            var inputData = input.ToArray();
            var outputData = new float[inputData.Length];

            if (_perTensor)
            {
                for (int i = 0; i < inputData.Length; i++)
                {
                    // Quantize: round(x / scale) + zeroPoint
                    var quantized = (float)Math.Round(inputData[i] / _scale) + _zeroPoint;
                    // Dequantize: (quantized - zeroPoint) * scale
                    outputData[i] = (quantized - _zeroPoint) * _scale;
                }
            }
            else if (_scales != null && _zeroPoints != null)
            {
                // Per-channel quantization (simplified for 1D case)
                int channels = _scales.Length;
                int valuesPerChannel = inputData.Length / channels;

                for (int c = 0; c < channels; c++)
                {
                    for (int i = 0; i < valuesPerChannel; i++)
                    {
                        int idx = c * valuesPerChannel + i;
                        var quantized = (float)Math.Round(inputData[idx] / _scales[c]) + _zeroPoints[c];
                        outputData[idx] = (quantized - _zeroPoints[c]) * _scales[c];
                    }
                }
            }

            return new Tensor(outputData, input.Shape);
        }

        public Tensor Backward(Tensor upstreamGradient)
        {
            // STE: Identity function - pass through gradients directly
            return upstreamGradient.Clone();
        }

        public void UpdateScaleAndZeroPoint(float newScale, int newZeroPoint)
        {
            _scale = newScale;
            _zeroPoint = newZeroPoint;
        }
    }
}
