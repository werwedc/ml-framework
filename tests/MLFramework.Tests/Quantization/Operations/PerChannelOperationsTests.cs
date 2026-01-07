using System;
using System.Linq;
using Xunit;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Operations;

namespace MLFramework.Tests.Quantization.Operations
{
    public class PerChannelOperationsTests
    {
        [Fact]
        public void QuantizeTensorPerChannel_WithValidTensor_ReturnsQuantizedArray()
        {
            // Arrange
            var channelScales = new float[] { 0.5f, 0.3f, 0.7f };
            var channelZeroPoints = new int[] { 0, 0, 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric,
                type: QuantizationType.Int8);

            // Create tensor with 3 channels, each with 2 elements
            var tensor = new float[] { 1.0f, 2.0f, 0.3f, 0.6f, 1.4f, 2.1f };

            // Act
            sbyte[] quantized = PerChannelOperations.QuantizeTensorPerChannel(tensor, parameters, channelAxis: 0);

            // Assert
            Assert.Equal(6, quantized.Length);
            // Channel 0: [1.0, 2.0] / 0.5 = [2, 4]
            Assert.Equal(2, quantized[0]);
            Assert.Equal(4, quantized[1]);
            // Channel 1: [0.3, 0.6] / 0.3 = [1, 2]
            Assert.Equal(1, quantized[2]);
            Assert.Equal(2, quantized[3]);
            // Channel 2: [1.4, 2.1] / 0.7 = [2, 3]
            Assert.Equal(2, quantized[4]);
            Assert.Equal(3, quantized[5]);
        }

        [Fact]
        public void QuantizeTensorPerChannel_WithPerTensorParameters_ThrowsArgumentException()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var tensor = new float[] { 1.0f, 2.0f, 3.0f };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PerChannelOperations.QuantizeTensorPerChannel(tensor, parameters));
        }

        [Fact]
        public void QuantizeTensorPerChannel_WithEmptyArray_ReturnsEmptyArray()
        {
            // Arrange
            var channelScales = new float[] { 0.5f };
            var channelZeroPoints = new int[] { 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);
            var tensor = Array.Empty<float>();

            // Act
            sbyte[] quantized = PerChannelOperations.QuantizeTensorPerChannel(tensor, parameters);

            // Assert
            Assert.Empty(quantized);
        }

        [Fact]
        public void QuantizeTensorPerChannel_WithNullArray_ReturnsEmptyArray()
        {
            // Arrange
            var channelScales = new float[] { 0.5f };
            var channelZeroPoints = new int[] { 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);

            // Act
            sbyte[] quantized = PerChannelOperations.QuantizeTensorPerChannel(null!, parameters);

            // Assert
            Assert.Empty(quantized);
        }

        [Fact]
        public void QuantizeTensorPerChannel_WithInvalidLength_ThrowsArgumentException()
        {
            // Arrange
            var channelScales = new float[] { 0.5f, 0.3f };
            var channelZeroPoints = new int[] { 0, 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);
            var tensor = new float[] { 1.0f, 2.0f, 3.0f }; // Not divisible by 2

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PerChannelOperations.QuantizeTensorPerChannel(tensor, parameters));
        }

        [Fact]
        public void DequantizeTensorPerChannel_WithValidTensor_ReturnsDequantizedArray()
        {
            // Arrange
            var channelScales = new float[] { 0.5f, 0.3f, 0.7f };
            var channelZeroPoints = new int[] { 0, 0, 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);

            var quantizedTensor = new sbyte[] { 2, 4, 1, 2, 2, 3 };

            // Act
            float[] dequantized = PerChannelOperations.DequantizeTensorPerChannel(quantizedTensor, parameters, channelAxis: 0);

            // Assert
            Assert.Equal(6, dequantized.Length);
            // Channel 0: [2, 4] * 0.5 = [1.0, 2.0]
            Assert.Equal(1.0f, dequantized[0], 6);
            Assert.Equal(2.0f, dequantized[1], 6);
            // Channel 1: [1, 2] * 0.3 = [0.3, 0.6]
            Assert.Equal(0.3f, dequantized[2], 6);
            Assert.Equal(0.6f, dequantized[3], 6);
            // Channel 2: [2, 3] * 0.7 = [1.4, 2.1]
            Assert.Equal(1.4f, dequantized[4], 6);
            Assert.Equal(2.1f, dequantized[5], 6);
        }

        [Fact]
        public void DequantizeTensorPerChannel_WithPerTensorParameters_ThrowsArgumentException()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, type: QuantizationType.Int8);
            var quantizedTensor = new sbyte[] { 1, 2, 3 };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PerChannelOperations.DequantizeTensorPerChannel(quantizedTensor, parameters));
        }

        [Fact]
        public void DequantizeTensorPerChannel_WithEmptyArray_ReturnsEmptyArray()
        {
            // Arrange
            var channelScales = new float[] { 0.5f };
            var channelZeroPoints = new int[] { 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);
            var quantizedTensor = Array.Empty<sbyte>();

            // Act
            float[] dequantized = PerChannelOperations.DequantizeTensorPerChannel(quantizedTensor, parameters);

            // Assert
            Assert.Empty(dequantized);
        }

        [Fact]
        public void DequantizeTensorPerChannel_WithNullArray_ReturnsEmptyArray()
        {
            // Arrange
            var channelScales = new float[] { 0.5f };
            var channelZeroPoints = new int[] { 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);

            // Act
            float[] dequantized = PerChannelOperations.DequantizeTensorPerChannel(null!, parameters);

            // Assert
            Assert.Empty(dequantized);
        }

        [Fact]
        public void QuantizeTensorPerChannel_WithCustomShape_ReturnsQuantizedArray()
        {
            // Arrange
            var channelScales = new float[] { 0.5f, 0.3f };
            var channelZeroPoints = new int[] { 0, 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);
            int[] shape = new int[] { 2, 3 }; // 2 channels, 3 elements each

            // Create tensor [2, 3] with channels-first layout
            var tensor = new float[] { 0.5f, 1.0f, 1.5f, 0.3f, 0.6f, 0.9f };

            // Act
            sbyte[] quantized = PerChannelOperations.QuantizeTensorPerChannel(tensor, shape, parameters, channelAxis: 0);

            // Assert
            Assert.Equal(6, quantized.Length);
            // Channel 0: [0.5, 1.0, 1.5] / 0.5 = [1, 2, 3]
            Assert.Equal(1, quantized[0]);
            Assert.Equal(2, quantized[1]);
            Assert.Equal(3, quantized[2]);
            // Channel 1: [0.3, 0.6, 0.9] / 0.3 = [1, 2, 3]
            Assert.Equal(1, quantized[3]);
            Assert.Equal(2, quantized[4]);
            Assert.Equal(3, quantized[5]);
        }

        [Fact]
        public void QuantizeTensorPerChannel_WithInvalidShape_ThrowsArgumentException()
        {
            // Arrange
            var channelScales = new float[] { 0.5f };
            var channelZeroPoints = new int[] { 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);
            int[] shape = new int[] { 2, 3 }; // Shape[0] = 2, but channel count = 1
            var tensor = new float[] { 0.5f, 1.0f, 1.5f, 0.3f, 0.6f, 0.9f };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PerChannelOperations.QuantizeTensorPerChannel(tensor, shape, parameters, channelAxis: 0));
        }

        [Fact]
        public void DequantizeTensorPerChannel_WithCustomShape_ReturnsDequantizedArray()
        {
            // Arrange
            var channelScales = new float[] { 0.5f, 0.3f };
            var channelZeroPoints = new int[] { 0, 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);
            int[] shape = new int[] { 2, 3 };

            var quantizedTensor = new sbyte[] { 1, 2, 3, 1, 2, 3 };

            // Act
            float[] dequantized = PerChannelOperations.DequantizeTensorPerChannel(quantizedTensor, shape, parameters, channelAxis: 0);

            // Assert
            Assert.Equal(6, dequantized.Length);
            // Channel 0: [1, 2, 3] * 0.5 = [0.5, 1.0, 1.5]
            Assert.Equal(0.5f, dequantized[0], 6);
            Assert.Equal(1.0f, dequantized[1], 6);
            Assert.Equal(1.5f, dequantized[2], 6);
            // Channel 1: [1, 2, 3] * 0.3 = [0.3, 0.6, 0.9]
            Assert.Equal(0.3f, dequantized[3], 6);
            Assert.Equal(0.6f, dequantized[4], 6);
            Assert.Equal(0.9f, dequantized[5], 6);
        }

        [Fact]
        public void ComputePerChannelParameters_WithValidTensor_ReturnsCorrectParameters()
        {
            // Arrange
            int[] shape = new int[] { 2, 3 }; // 2 channels, 3 elements each
            var tensor = new float[] { 1.0f, 2.0f, 3.0f, 0.3f, 0.6f, 0.9f };

            // Act
            var parameters = PerChannelOperations.ComputePerChannelParameters(
                tensor,
                shape,
                channelAxis: 0,
                QuantizationMode.PerChannelSymmetric,
                QuantizationType.Int8);

            // Assert
            Assert.True(parameters.IsPerChannel);
            Assert.Equal(2, parameters.ChannelCount);
            Assert.NotNull(parameters.ChannelScales);
            Assert.NotNull(parameters.ChannelZeroPoints);

            // Channel 0: [1.0, 2.0, 3.0], min=1.0, max=3.0, symmetric -> [-3.0, 3.0]
            // Scale = 6.0 / 255 = 0.0235
            Assert.InRange(parameters.ChannelScales![0], 0.02f, 0.03f);
            Assert.Equal((sbyte.MinValue + sbyte.MaxValue) / 2, parameters.ChannelZeroPoints![0]);

            // Channel 1: [0.3, 0.6, 0.9], min=0.3, max=0.9, symmetric -> [-0.9, 0.9]
            // Scale = 1.8 / 255 = 0.00706
            Assert.InRange(parameters.ChannelScales![1], 0.006f, 0.008f);
        }

        [Fact]
        public void ComputePerChannelParameters_WithEmptyTensor_ThrowsArgumentException()
        {
            // Arrange
            int[] shape = new int[] { 2, 3 };
            var tensor = Array.Empty<float>();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PerChannelOperations.ComputePerChannelParameters(
                    tensor,
                    shape,
                    channelAxis: 0,
                    QuantizationMode.PerChannelSymmetric,
                    QuantizationType.Int8));
        }

        [Fact]
        public void ComputePerChannelParameters_WithNullTensor_ThrowsArgumentException()
        {
            // Arrange
            int[] shape = new int[] { 2, 3 };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PerChannelOperations.ComputePerChannelParameters(
                    null!,
                    shape,
                    channelAxis: 0,
                    QuantizationMode.PerChannelSymmetric,
                    QuantizationType.Int8));
        }

        [Fact]
        public void ComputePerChannelParameters_WithInvalidShape_ThrowsArgumentException()
        {
            // Arrange
            int[] shape = new int[] { 2 }; // Only 1 dimension
            var tensor = new float[] { 1.0f, 2.0f, 3.0f, 0.3f, 0.6f, 0.9f };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                PerChannelOperations.ComputePerChannelParameters(
                    tensor,
                    shape,
                    channelAxis: 1, // Channel axis exceeds shape length
                    QuantizationMode.PerChannelSymmetric,
                    QuantizationType.Int8));
        }

        [Fact]
        public void ComputePerChannelParameters_WithNaNAndInfValues_HandlesCorrectly()
        {
            // Arrange
            int[] shape = new int[] { 2, 3 };
            var tensor = new float[] { 1.0f, float.NaN, 3.0f, float.PositiveInfinity, 0.6f, 0.9f };

            // Act
            var parameters = PerChannelOperations.ComputePerChannelParameters(
                tensor,
                shape,
                channelAxis: 0,
                QuantizationMode.PerChannelSymmetric,
                QuantizationType.Int8);

            // Assert
            Assert.True(parameters.IsPerChannel);
            Assert.NotNull(parameters.ChannelScales);
            // Should have filtered out NaN and Inf
            Assert.All(parameters.ChannelScales!, scale => Assert.True(scale > 0));
        }

        [Fact]
        public void ComputePerChannelParameters_AsymmetricMode_CalculatesZeroPoints()
        {
            // Arrange
            int[] shape = new int[] { 2, 3 };
            var tensor = new float[] { 1.0f, 2.0f, 3.0f, 0.3f, 0.6f, 0.9f };

            // Act
            var parameters = PerChannelOperations.ComputePerChannelParameters(
                tensor,
                shape,
                channelAxis: 0,
                QuantizationMode.PerChannelAsymmetric,
                QuantizationType.Int8);

            // Assert
            Assert.True(parameters.IsPerChannel);
            Assert.NotNull(parameters.ChannelZeroPoints);

            // In asymmetric mode, zero-points should not all be the same
            bool allZeroPointsSame = parameters.ChannelZeroPoints!.Distinct().Count() == 1;
            Assert.False(allZeroPointsSame, "Zero-points should vary in asymmetric mode");
        }

        [Fact]
        public void RoundTrip_PerChannelQuantizationDequantization_MaintainsApproximateValues()
        {
            // Arrange
            var channelScales = new float[] { 0.01f, 0.02f, 0.015f };
            var channelZeroPoints = new int[] { 0, 0, 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);
            var tensor = Enumerable.Range(0, 30).Select(i => i * 0.1f - 1.5f).ToArray();

            // Act
            sbyte[] quantized = PerChannelOperations.QuantizeTensorPerChannel(tensor, parameters, channelAxis: 0);
            float[] dequantized = PerChannelOperations.DequantizeTensorPerChannel(quantized, parameters, channelAxis: 0);

            // Assert
            Assert.Equal(tensor.Length, dequantized.Length);
            for (int i = 0; i < tensor.Length; i++)
            {
                float error = Math.Abs(dequantized[i] - tensor[i]);
                Assert.True(error < 0.02f, $"Error at index {i}: {error}");
            }
        }

        [Fact]
        public void QuantizeTensorPerChannel_WithDifferentChannelAxis_QuantizesCorrectly()
        {
            // Arrange
            var channelScales = new float[] { 0.5f, 0.3f, 0.7f };
            var channelZeroPoints = new int[] { 0, 0, 0 };
            var parameters = new QuantizationParameters(
                channelScales,
                channelZeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);
            int[] shape = new int[] { 3, 2, 2 }; // Shape: [H, C, W] where C is at axis 1

            // Create tensor with shape [3, 2, 2] where axis 1 is channels
            var tensor = new float[]
            {
                // Channel 0 (elements 0,1,4,5,8,9)
                0.5f, 1.0f,
                0.3f, 0.6f,
                0.7f, 1.4f,
                // Channel 1 (elements 2,3,6,7,10,11)
                0.15f, 0.3f,
                0.2f, 0.4f,
                0.35f, 0.7f
            };

            // Act
            sbyte[] quantized = PerChannelOperations.QuantizeTensorPerChannel(tensor, shape, parameters, channelAxis: 1);

            // Assert
            Assert.Equal(12, quantized.Length);
            // Elements at indices where channel=0 should use scale 0.5
            // Elements at indices where channel=1 should use scale 0.3
            Assert.Equal(1, quantized[0]); // 0.5 / 0.5 = 1
            Assert.Equal(2, quantized[1]); // 1.0 / 0.5 = 2
            Assert.Equal(1, quantized[2]); // 0.15 / 0.3 = 0.5 -> round to 1
            Assert.Equal(1, quantized[3]); // 0.3 / 0.3 = 1
        }
    }
}
