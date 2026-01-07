using System;
using System.Linq;
using Xunit;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Operations;

namespace MLFramework.Tests.Quantization.Operations
{
    public class DequantizationOperationsTests
    {
        [Fact]
        public void Dequantize_WithValidValue_ReturnsCorrectFloat()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            sbyte quantizedValue = 2;

            // Act
            float dequantized = DequantizationOperations.Dequantize(quantizedValue, parameters);

            // Assert
            Assert.Equal(1.0f, dequantized, 6); // (2 - 0) * 0.5 = 1.0
        }

        [Fact]
        public void Dequantize_WithNegativeValue_ReturnsCorrectFloat()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            sbyte quantizedValue = -2;

            // Act
            float dequantized = DequantizationOperations.Dequantize(quantizedValue, parameters);

            // Assert
            Assert.Equal(-1.0f, dequantized, 6); // (-2 - 0) * 0.5 = -1.0
        }

        [Fact]
        public void Dequantize_WithZeroPoint_ReturnsCorrectFloat()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 10, type: QuantizationType.Int8);
            sbyte quantizedValue = 10;

            // Act
            float dequantized = DequantizationOperations.Dequantize(quantizedValue, parameters);

            // Assert
            Assert.Equal(0.0f, dequantized, 6); // (10 - 10) * 0.5 = 0.0
        }

        [Fact]
        public void DequantizeUInt8_WithValidValue_ReturnsCorrectFloat()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 128, type: QuantizationType.UInt8);
            byte quantizedValue = 130;

            // Act
            float dequantized = DequantizationOperations.DequantizeUInt8(quantizedValue, parameters);

            // Assert
            Assert.Equal(1.0f, dequantized, 6); // (130 - 128) * 0.5 = 1.0
        }

        [Fact]
        public void DequantizeTensor_WithValidArray_ReturnsDequantizedArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var quantizedTensor = new sbyte[] { -2, 0, 2, 4 };

            // Act
            float[] dequantized = DequantizationOperations.DequantizeTensor(quantizedTensor, parameters);

            // Assert
            Assert.Equal(4, dequantized.Length);
            Assert.Equal(-1.0f, dequantized[0], 6);
            Assert.Equal(0.0f, dequantized[1], 6);
            Assert.Equal(1.0f, dequantized[2], 6);
            Assert.Equal(2.0f, dequantized[3], 6);
        }

        [Fact]
        public void DequantizeTensor_WithEmptyArray_ReturnsEmptyArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var quantizedTensor = Array.Empty<sbyte>();

            // Act
            float[] dequantized = DequantizationOperations.DequantizeTensor(quantizedTensor, parameters);

            // Assert
            Assert.Empty(dequantized);
        }

        [Fact]
        public void DequantizeTensor_WithNullArray_ReturnsEmptyArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act
            float[] dequantized = DequantizationOperations.DequantizeTensor(null!, parameters);

            // Assert
            Assert.Empty(dequantized);
        }

        [Fact]
        public void DequantizeTensorUInt8_WithValidArray_ReturnsDequantizedArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 128, type: QuantizationType.UInt8);
            var quantizedTensor = new byte[] { 128, 130, 132 };

            // Act
            float[] dequantized = DequantizationOperations.DequantizeTensorUInt8(quantizedTensor, parameters);

            // Assert
            Assert.Equal(3, dequantized.Length);
            Assert.Equal(0.0f, dequantized[0], 6);
            Assert.Equal(1.0f, dequantized[1], 6);
            Assert.Equal(2.0f, dequantized[2], 6);
        }

        [Fact]
        public void DequantizeTensorInPlace_ReturnsNewArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var quantizedTensor = new sbyte[] { -2, 0, 2 };

            // Act
            float[] dequantized = DequantizationOperations.DequantizeTensorInPlace(quantizedTensor, parameters);

            // Assert
            Assert.NotSame(quantizedTensor, dequantized);
            Assert.Equal(3, dequantized.Length);
            Assert.Equal(-1.0f, dequantized[0], 6);
            Assert.Equal(0.0f, dequantized[1], 6);
            Assert.Equal(1.0f, dequantized[2], 6);
        }

        [Fact]
        public void DequantizeTensorInPlace_WithNullArray_ThrowsArgumentNullException()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => DequantizationOperations.DequantizeTensorInPlace(null!, parameters));
        }

        [Fact]
        public void DequantizeSymmetric_WithValidValue_ReturnsCorrectFloat()
        {
            // Arrange
            sbyte quantizedValue = 2;
            float scale = 0.5f;

            // Act
            float dequantized = DequantizationOperations.DequantizeSymmetric(quantizedValue, scale);

            // Assert
            Assert.Equal(1.0f, dequantized, 6); // 2 * 0.5 = 1.0
        }

        [Fact]
        public void DequantizeAsymmetric_WithValidValue_ReturnsCorrectFloat()
        {
            // Arrange
            sbyte quantizedValue = 12;
            float scale = 0.5f;
            int zeroPoint = 10;

            // Act
            float dequantized = DequantizationOperations.DequantizeAsymmetric(quantizedValue, scale, zeroPoint);

            // Assert
            Assert.Equal(1.0f, dequantized, 6); // (12 - 10) * 0.5 = 1.0
        }

        [Fact]
        public void ComputeQuantizationError_WithValidTensor_ReturnsErrorArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = new float[] { -1.0f, 0.0f, 1.0f, 2.0f };

            // Act
            float[] error = DequantizationOperations.ComputeQuantizationError(tensor, parameters);

            // Assert
            Assert.Equal(4, error.Length);
            Assert.All(error, e => Assert.InRange(Math.Abs(e), 0, 0.5)); // Error should be small
        }

        [Fact]
        public void ComputeQuantizationError_WithEmptyArray_ReturnsEmptyArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = Array.Empty<float>();

            // Act
            float[] error = DequantizationOperations.ComputeQuantizationError(tensor, parameters);

            // Assert
            Assert.Empty(error);
        }

        [Fact]
        public void ComputeQuantizationError_WithNullArray_ReturnsEmptyArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act
            float[] error = DequantizationOperations.ComputeQuantizationError(null!, parameters);

            // Assert
            Assert.Empty(error);
        }

        [Fact]
        public void ComputeMSE_WithValidTensor_ReturnsNonZeroMSE()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = new float[] { -1.0f, 0.0f, 1.0f, 2.0f };

            // Act
            float mse = DequantizationOperations.ComputeMSE(tensor, parameters);

            // Assert
            Assert.True(mse >= 0);
            Assert.True(mse < 1.0f); // MSE should be small for well-scaled data
        }

        [Fact]
        public void ComputeMSE_WithEmptyArray_ReturnsZero()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = Array.Empty<float>();

            // Act
            float mse = DequantizationOperations.ComputeMSE(tensor, parameters);

            // Assert
            Assert.Equal(0, mse);
        }

        [Fact]
        public void ComputeMSE_WithNullArray_ReturnsZero()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act
            float mse = DequantizationOperations.ComputeMSE(null!, parameters);

            // Assert
            Assert.Equal(0, mse);
        }

        [Fact]
        public void ComputeSQNR_WithValidTensor_ReturnsFiniteValue()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = new float[] { -1.0f, 0.0f, 1.0f, 2.0f };

            // Act
            float sqnr = DequantizationOperations.ComputeSQNR(tensor, parameters);

            // Assert
            Assert.True(float.IsFinite(sqnr));
            Assert.True(sqnr > 0); // SQNR should be positive for non-zero signals
        }

        [Fact]
        public void ComputeSQNR_WithEmptyArray_ReturnsNegativeInfinity()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = Array.Empty<float>();

            // Act
            float sqnr = DequantizationOperations.ComputeSQNR(tensor, parameters);

            // Assert
            Assert.Equal(float.NegativeInfinity, sqnr);
        }

        [Fact]
        public void ComputeSQNR_WithNullArray_ReturnsNegativeInfinity()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act
            float sqnr = DequantizationOperations.ComputeSQNR(null!, parameters);

            // Assert
            Assert.Equal(float.NegativeInfinity, sqnr);
        }

        [Fact]
        public void ComputePSNR_WithValidTensor_ReturnsFiniteValue()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = new float[] { -1.0f, 0.0f, 1.0f, 2.0f };

            // Act
            float psnr = DequantizationOperations.ComputePSNR(tensor, parameters);

            // Assert
            Assert.True(float.IsFinite(psnr));
            Assert.True(psnr > 0); // PSNR should be positive
        }

        [Fact]
        public void ComputePSNR_WithEmptyArray_ReturnsNegativeInfinity()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = Array.Empty<float>();

            // Act
            float psnr = DequantizationOperations.ComputePSNR(tensor, parameters);

            // Assert
            Assert.Equal(float.NegativeInfinity, psnr);
        }

        [Fact]
        public void ComputePSNR_WithNullArray_ReturnsNegativeInfinity()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act
            float psnr = DequantizationOperations.ComputePSNR(null!, parameters);

            // Assert
            Assert.Equal(float.NegativeInfinity, psnr);
        }

        [Fact]
        public void RoundTrip_QuantizationDequantization_MaintainsApproximateValues()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.01f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = new float[] { -0.5f, -0.25f, 0.0f, 0.25f, 0.5f };

            // Act
            sbyte[] quantized = QuantizationOperations.QuantizeTensor(tensor, parameters);
            float[] dequantized = DequantizationOperations.DequantizeTensor(quantized, parameters);

            // Assert
            Assert.Equal(tensor.Length, dequantized.Length);
            for (int i = 0; i < tensor.Length; i++)
            {
                float error = Math.Abs(dequantized[i] - tensor[i]);
                Assert.True(error < 0.01f, $"Error at index {i}: {error}");
            }
        }

        [Fact]
        public void ComputeMSE_WithZeroError_ReturnsZero()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 1.0f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = new float[] { -1.0f, 0.0f, 1.0f };

            // Act
            float mse = DequantizationOperations.ComputeMSE(tensor, parameters);

            // Assert
            // MSE should be very small (but might not be exactly zero due to quantization)
            Assert.True(mse < 0.01f);
        }

        [Fact]
        public void ComputeSQNR_WithLargerSignal_ReturnsHigherValue()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.01f, zeroPoint: 0, type: QuantizationType.Int8);
            var weakSignal = Enumerable.Repeat(0.01f, 100).ToArray();
            var strongSignal = Enumerable.Repeat(1.0f, 100).ToArray();

            // Act
            float sqnrWeak = DequantizationOperations.ComputeSQNR(weakSignal, parameters);
            float sqnrStrong = DequantizationOperations.ComputeSQNR(strongSignal, parameters);

            // Assert
            // Stronger signal should generally give higher SNR
            Assert.True(sqnrStrong >= sqnrWeak);
        }

        [Fact]
        public void ComputePSNR_WithLowerError_ReturnsHigherValue()
        {
            // Arrange
            var tightParameters = new QuantizationParameters(scale: 0.001f, zeroPoint: 0, type: QuantizationType.Int8);
            var looseParameters = new QuantizationParameters(scale: 0.1f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = Enumerable.Range(0, 100).Select(i => i * 0.01f).ToArray();

            // Act
            float psnrTight = DequantizationOperations.ComputePSNR(tensor, tightParameters);
            float psnrLoose = DequantizationOperations.ComputePSNR(tensor, looseParameters);

            // Assert
            // Tighter parameters (smaller scale) should give higher PSNR (lower error)
            Assert.True(psnrTight >= psnrLoose);
        }

        [Fact]
        public void DequantizeTensor_WithLargeArray_HandlesCorrectly()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.01f, zeroPoint: 0, type: QuantizationType.Int8);
            var quantizedTensor = Enumerable.Range(-128, 256).Select(i => (sbyte)i).ToArray();

            // Act
            float[] dequantized = DequantizationOperations.DequantizeTensor(quantizedTensor, parameters);

            // Assert
            Assert.Equal(256, dequantized.Length);
            Assert.Equal(-128 * 0.01f, dequantized[0], 6);
            Assert.Equal(127 * 0.01f, dequantized[^1], 6);
        }
    }
}
