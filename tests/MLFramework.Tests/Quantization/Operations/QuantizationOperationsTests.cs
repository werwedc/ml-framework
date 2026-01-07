using System;
using System.Linq;
using Xunit;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Operations;

namespace MLFramework.Tests.Quantization.Operations
{
    public class QuantizationOperationsTests
    {
        [Fact]
        public void Quantize_WithValidValue_ReturnsCorrectInt8()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            float value = 1.0f;

            // Act
            sbyte quantized = QuantizationOperations.Quantize(value, parameters);

            // Assert
            Assert.Equal(2, quantized); // 1.0 / 0.5 = 2
        }

        [Fact]
        public void Quantize_WithNegativeValue_ReturnsCorrectInt8()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            float value = -1.0f;

            // Act
            sbyte quantized = QuantizationOperations.Quantize(value, parameters);

            // Assert
            Assert.Equal(-2, quantized); // -1.0 / 0.5 = -2
        }

        [Fact]
        public void Quantize_WithZeroPoint_ReturnsCorrectInt8()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 10, type: QuantizationType.Int8);
            float value = 0.0f;

            // Act
            sbyte quantized = QuantizationOperations.Quantize(value, parameters);

            // Assert
            Assert.Equal(10, quantized); // 0 / 0.5 + 10 = 10
        }

        [Fact]
        public void Quantize_WithNaN_ReturnsZeroPoint()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 5, type: QuantizationType.Int8);

            // Act
            sbyte quantized = QuantizationOperations.Quantize(float.NaN, parameters);

            // Assert
            Assert.Equal(5, quantized);
        }

        [Fact]
        public void Quantize_WithPositiveInfinity_ReturnsMaxValue()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act
            sbyte quantized = QuantizationOperations.Quantize(float.PositiveInfinity, parameters);

            // Assert
            Assert.Equal(sbyte.MaxValue, quantized);
        }

        [Fact]
        public void Quantize_WithNegativeInfinity_ReturnsMinValue()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act
            sbyte quantized = QuantizationOperations.Quantize(float.NegativeInfinity, parameters);

            // Assert
            Assert.Equal(sbyte.MinValue, quantized);
        }

        [Fact]
        public void Quantize_WithLargeValue_ClampsToMaxValue()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            float value = 1000.0f; // Would produce 2000 if not clamped

            // Act
            sbyte quantized = QuantizationOperations.Quantize(value, parameters);

            // Assert
            Assert.Equal(sbyte.MaxValue, quantized);
        }

        [Fact]
        public void Quantize_WithVeryNegativeValue_ClampsToMinValue()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            float value = -1000.0f; // Would produce -2000 if not clamped

            // Act
            sbyte quantized = QuantizationOperations.Quantize(value, parameters);

            // Assert
            Assert.Equal(sbyte.MinValue, quantized);
        }

        [Theory]
        [InlineData(1.25f, 1.0f, 0.5f, 0)] // 1.25 / 0.5 = 2.5 -> rounds to 3 (half away from zero), but should round to 2 (banker's rounding?)
        [InlineData(1.75f, 1.0f, 0.5f, 0)] // 1.75 / 0.5 = 3.5 -> rounds to 4 (half away from zero)
        [InlineData(0.25f, 1.0f, 0.5f, 0)] // 0.25 / 0.5 = 0.5 -> rounds to 1 (half away from zero)
        [InlineData(-0.25f, 1.0f, 0.5f, 0)] // -0.25 / 0.5 = -0.5 -> rounds to -1 (half away from zero)
        public void Quantize_RoundsHalfAwayFromZero(float value, float scale, float expectedResult, int zeroPoint)
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: scale, zeroPoint: zeroPoint, type: QuantizationType.Int8);

            // Act
            sbyte quantized = QuantizationOperations.Quantize(value, parameters);

            // Assert
            Assert.Equal(expectedResult, quantized);
        }

        [Fact]
        public void Quantize_WithUInt8Type_ThrowsArgumentException()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.UInt8);
            float value = 1.0f;

            // Act & Assert
            Assert.Throws<ArgumentException>(() => QuantizationOperations.Quantize(value, parameters));
        }

        [Fact]
        public void QuantizeUInt8_WithValidValue_ReturnsCorrectUInt8()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 128, type: QuantizationType.UInt8);
            float value = 1.0f;

            // Act
            byte quantized = QuantizationOperations.QuantizeUInt8(value, parameters);

            // Assert
            Assert.Equal(130, quantized); // 1.0 / 0.5 + 128 = 2 + 128 = 130
        }

        [Fact]
        public void QuantizeUInt8_WithInt8Type_ThrowsArgumentException()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            float value = 1.0f;

            // Act & Assert
            Assert.Throws<ArgumentException>(() => QuantizationOperations.QuantizeUInt8(value, parameters));
        }

        [Fact]
        public void QuantizeTensor_WithValidArray_ReturnsQuantizedArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = new float[] { -1.0f, 0.0f, 1.0f, 2.0f };

            // Act
            sbyte[] quantized = QuantizationOperations.QuantizeTensor(tensor, parameters);

            // Assert
            Assert.Equal(4, quantized.Length);
            Assert.Equal(-2, quantized[0]);
            Assert.Equal(0, quantized[1]);
            Assert.Equal(2, quantized[2]);
            Assert.Equal(4, quantized[3]);
        }

        [Fact]
        public void QuantizeTensor_WithEmptyArray_ReturnsEmptyArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = Array.Empty<float>();

            // Act
            sbyte[] quantized = QuantizationOperations.QuantizeTensor(tensor, parameters);

            // Assert
            Assert.Empty(quantized);
        }

        [Fact]
        public void QuantizeTensor_WithNullArray_ReturnsEmptyArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act
            sbyte[] quantized = QuantizationOperations.QuantizeTensor(null!, parameters);

            // Assert
            Assert.Empty(quantized);
        }

        [Fact]
        public void QuantizeTensorInPlace_ModifiesOriginalArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = new float[] { -1.0f, 0.0f, 1.0f };

            // Act
            float[] result = QuantizationOperations.QuantizeTensorInPlace(tensor, parameters);

            // Assert
            Assert.Same(tensor, result);
            Assert.Equal(-2, (sbyte)result[0]);
            Assert.Equal(0, (sbyte)result[1]);
            Assert.Equal(2, (sbyte)result[2]);
        }

        [Fact]
        public void QuantizeTensorInPlace_WithNullArray_ThrowsArgumentNullException()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => QuantizationOperations.QuantizeTensorInPlace(null!, parameters));
        }

        [Fact]
        public void QuantizeSymmetric_WithValidValue_ReturnsCorrectValue()
        {
            // Arrange
            float value = 1.0f;
            float scale = 0.5f;

            // Act
            sbyte quantized = QuantizationOperations.QuantizeSymmetric(value, scale);

            // Assert
            Assert.Equal(2, quantized); // 1.0 / 0.5 = 2
        }

        [Fact]
        public void QuantizeSymmetric_WithNaN_ReturnsZero()
        {
            // Arrange
            float scale = 0.5f;

            // Act
            sbyte quantized = QuantizationOperations.QuantizeSymmetric(float.NaN, scale);

            // Assert
            Assert.Equal(0, quantized); // (min + max) / 2 = (-128 + 127) / 2 = -0.5 -> 0
        }

        [Fact]
        public void QuantizeAsymmetric_WithValidValue_ReturnsCorrectValue()
        {
            // Arrange
            float value = 1.0f;
            float scale = 0.5f;
            int zeroPoint = 10;

            // Act
            sbyte quantized = QuantizationOperations.QuantizeAsymmetric(value, scale, zeroPoint);

            // Assert
            Assert.Equal(12, quantized); // 1.0 / 0.5 + 10 = 2 + 10 = 12
        }

        [Fact]
        public void QuantizeAsymmetric_WithNaN_ReturnsZeroPoint()
        {
            // Arrange
            float scale = 0.5f;
            int zeroPoint = 10;

            // Act
            sbyte quantized = QuantizationOperations.QuantizeAsymmetric(float.NaN, scale, zeroPoint);

            // Assert
            Assert.Equal(10, quantized);
        }

        [Fact]
        public void QuantizeAsymmetric_WithCustomRange_ClampsToRange()
        {
            // Arrange
            float value = 1000.0f;
            float scale = 0.5f;
            int zeroPoint = 0;
            int quantMin = -64;
            int quantMax = 63;

            // Act
            sbyte quantized = QuantizationOperations.QuantizeAsymmetric(value, scale, zeroPoint, quantMin, quantMax);

            // Assert
            Assert.Equal(63, quantized);
        }

        [Theory]
        [InlineData(0.5f, 0, 0.5f, 0.0f)] // Symmetric case
        [InlineData(0.5f, 10, 1.0f, 0.0f)] // With zero-point
        [InlineData(0.1f, 0, 2.5f, 0.0f)] // Smaller scale
        public void Dequantize_ReversesQuantization(float scale, int zeroPoint, float originalValue, float tolerance)
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: scale, zeroPoint: zeroPoint, type: QuantizationType.Int8);

            // Act
            sbyte quantized = QuantizationOperations.Quantize(originalValue, parameters);
            float dequantized = QuantizationOperations.Dequantize(quantized, parameters);

            // Assert
            Assert.InRange(dequantized, originalValue - tolerance, originalValue + tolerance);
        }

        [Fact]
        public void QuantizeAndDequantize_RoundTrip_MaintainsApproximateValue()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            float originalValue = 1.23f;

            // Act
            sbyte quantized = QuantizationOperations.Quantize(originalValue, parameters);
            float dequantized = QuantizationOperations.Dequantize(quantized, parameters);

            // Assert
            float error = Math.Abs(dequantized - originalValue);
            Assert.True(error < 0.5, $"Error {error} exceeds tolerance 0.5");
        }

        [Fact]
        public void QuantizeTensorUInt8_WithValidArray_ReturnsQuantizedArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 128, type: QuantizationType.UInt8);
            var tensor = new float[] { 0.0f, 1.0f, 2.0f };

            // Act
            byte[] quantized = QuantizationOperations.QuantizeTensorUInt8(tensor, parameters);

            // Assert
            Assert.Equal(3, quantized.Length);
            Assert.Equal(128, quantized[0]); // 0 / 0.5 + 128 = 128
            Assert.Equal(130, quantized[1]); // 1.0 / 0.5 + 128 = 130
            Assert.Equal(132, quantized[2]); // 2.0 / 0.5 + 128 = 132
        }

        [Fact]
        public void QuantizeTensorUInt8_WithEmptyArray_ReturnsEmptyArray()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 128, type: QuantizationType.UInt8);
            var tensor = Array.Empty<float>();

            // Act
            byte[] quantized = QuantizationOperations.QuantizeTensorUInt8(tensor, parameters);

            // Assert
            Assert.Empty(quantized);
        }

        [Fact]
        public void Quantize_WithAllSameValues_ReturnsConsistentQuantizedValues()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            float value = 1.5f;

            // Act
            sbyte quantized1 = QuantizationOperations.Quantize(value, parameters);
            sbyte quantized2 = QuantizationOperations.Quantize(value, parameters);
            sbyte quantized3 = QuantizationOperations.Quantize(value, parameters);

            // Assert
            Assert.Equal(quantized1, quantized2);
            Assert.Equal(quantized2, quantized3);
        }

        [Fact]
        public void Quantize_WithAllSameValuesInArray_ReturnsConsistentQuantizedValues()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = Enumerable.Repeat(1.5f, 100).ToArray();

            // Act
            sbyte[] quantized = QuantizationOperations.QuantizeTensor(tensor, parameters);

            // Assert
            Assert.All(quantized, q => Assert.Equal(3, q));
        }
    }
}
