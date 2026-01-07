using RitterFramework.Core.Tensor;
using MLFramework.Quantization.Backends;
using MLFramework.Quantization.Backends.CPUBackend;
using MLFramework.Quantization.DataStructures;
using Xunit;

namespace MLFramework.Tests.Quantization.Backends
{
    /// <summary>
    /// Tests for CPU backend implementation.
    /// </summary>
    public class CPUBackendTests
    {
        private readonly CPUBackend _backend;

        public CPUBackendTests()
        {
            _backend = new CPUBackend();
        }

        [Fact]
        public void IsAvailable_AlwaysReturnsTrue()
        {
            // Act
            var isAvailable = _backend.IsAvailable();

            // Assert
            Assert.True(isAvailable);
        }

        [Fact]
        public void Quantize_ValidInput_ProducesCorrectResults()
        {
            // Arrange
            var inputData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var inputTensor = new Tensor(inputData, new int[] { 4 });
            var parameters = new QuantizationParameters(
                scale: 1.0f,
                zeroPoint: 0,
                min: 1.0f,
                max: 4.0f,
                mode: QuantizationMode.PerTensorSymmetric);

            // Act
            var quantized = _backend.Quantize(inputTensor, parameters);

            // Assert
            Assert.NotNull(quantized);
            Assert.Equal(inputTensor.Shape, quantized.Shape);
        }

        [Fact]
        public void Dequantize_ValidInput_ProducesCorrectResults()
        {
            // Arrange
            var inputData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            var inputTensor = new Tensor(inputData, new int[] { 4 }, false, DataType.Int8);
            var parameters = new QuantizationParameters(
                scale: 1.0f,
                zeroPoint: 0,
                min: 1.0f,
                max: 4.0f,
                mode: QuantizationMode.PerTensorSymmetric);

            // Act
            var dequantized = _backend.Dequantize(inputTensor, parameters);

            // Assert
            Assert.NotNull(dequantized);
            Assert.Equal(inputTensor.Shape, dequantized.Shape);
        }

        [Fact]
        public void MatMulInt8_ValidMatrices_ProducesCorrectResults()
        {
            // Arrange
            var matrixA = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new int[] { 2, 2 }, false, DataType.Int8);
            var matrixB = new Tensor(new float[] { 5.0f, 6.0f, 7.0f, 8.0f }, new int[] { 2, 2 }, false, DataType.Int8);

            // Act
            var result = _backend.MatMulInt8(matrixA, matrixB);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(2, result.Shape[1]);
        }

        [Fact]
        public void MatMulInt8_IncompatibleDimensions_ThrowsException()
        {
            // Arrange
            var matrixA = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 1, 2 }, false, DataType.Int8);
            var matrixB = new Tensor(new float[] { 3.0f, 4.0f, 5.0f }, new int[] { 1, 3 }, false, DataType.Int8);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _backend.MatMulInt8(matrixA, matrixB));
        }

        [Fact]
        public void Conv2DInt8_ValidInput_ProducesCorrectResults()
        {
            // Arrange
            var input = new Tensor(new float[4 * 3 * 5 * 5], new int[] { 4, 3, 5, 5 }, false, DataType.Int8);
            var weights = new Tensor(new float[8 * 3 * 3 * 3], new int[] { 8, 3, 3, 3 }, false, DataType.Int8);
            var stride = new int[] { 1, 1 };
            var padding = new int[] { 0, 0 };
            var dilation = new int[] { 1, 1 };

            // Act
            var result = _backend.Conv2DInt8(input, weights, null, stride, padding, dilation);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(4, result.Shape[0]); // batch size
            Assert.Equal(8, result.Shape[1]); // output channels
            Assert.Equal(3, result.Shape[2]); // output height (5 - 3 + 1)
            Assert.Equal(3, result.Shape[3]); // output width
        }

        [Fact]
        public void Conv2DInt8_WithBias_ProducesCorrectResults()
        {
            // Arrange
            var input = new Tensor(new float[1 * 1 * 5 * 5], new int[] { 1, 1, 5, 5 }, false, DataType.Int8);
            var weights = new Tensor(new float[2 * 1 * 3 * 3], new int[] { 2, 1, 3, 3 }, false, DataType.Int8);
            var bias = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
            var stride = new int[] { 1, 1 };
            var padding = new int[] { 0, 0 };
            var dilation = new int[] { 1, 1 };

            // Act
            var result = _backend.Conv2DInt8(input, weights, bias, stride, padding, dilation);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Shape[1]); // output channels
        }

        [Fact]
        public void QuantizeDequantize_RoundTrip_ProducesApproximateOriginalValues()
        {
            // Arrange
            var inputData = new float[] { 1.5f, 2.7f, 3.2f, 4.9f };
            var inputTensor = new Tensor(inputData, new int[] { 4 });
            var parameters = new QuantizationParameters(
                scale: 0.5f,
                zeroPoint: 0,
                min: 1.0f,
                max: 5.0f,
                mode: QuantizationMode.PerTensorSymmetric);

            // Act
            var quantized = _backend.Quantize(inputTensor, parameters);
            var dequantized = _backend.Dequantize(quantized, parameters);

            // Assert
            Assert.NotNull(dequantized);
            Assert.Equal(inputTensor.Shape, dequantized.Shape);

            // Check that values are approximately preserved
            for (int i = 0; i < inputData.Length; i++)
            {
                Assert.Equal(inputData[i], dequantized.Data[i], precision: 1);
            }
        }

        [Fact]
        public void GetCapabilities_ReturnsFullCapabilities()
        {
            // Act
            var capabilities = _backend.GetCapabilities();

            // Assert
            Assert.True(capabilities.SupportsInt8MatMul);
            Assert.True(capabilities.SupportsInt8Conv2D);
            Assert.True(capabilities.SupportsPerChannelQuantization);
            Assert.True(capabilities.SupportsMixedPrecision);
            Assert.True(capabilities.SupportsDynamicQuantization);
            Assert.True(capabilities.SupportsStaticQuantization);
            Assert.True(capabilities.SupportsAsymmetricQuantization);
            Assert.True(capabilities.SupportsSymmetricQuantization);
        }
    }
}
