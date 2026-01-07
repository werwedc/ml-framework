using MLFramework.Quantization.DataStructures;
using Xunit;

namespace MLFramework.Tests.Quantization
{
    public class QuantizationParametersTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesValidParameters()
        {
            // Arrange & Act
            var parameters = new QuantizationParameters(
                scale: 0.5f,
                zeroPoint: 0,
                min: -1.0f,
                max: 1.0f,
                mode: QuantizationMode.PerTensorSymmetric,
                type: QuantizationType.Int8);

            // Assert
            Assert.Equal(0.5f, parameters.Scale);
            Assert.Equal(0, parameters.ZeroPoint);
            Assert.Equal(-1.0f, parameters.Min);
            Assert.Equal(1.0f, parameters.Max);
            Assert.Equal(QuantizationMode.PerTensorSymmetric, parameters.Mode);
            Assert.Equal(QuantizationType.Int8, parameters.Type);
        }

        [Fact]
        public void IsPerChannel_WithPerTensorMode_ReturnsFalse()
        {
            // Arrange
            var parameters = new QuantizationParameters(
                scale: 0.5f,
                zeroPoint: 0,
                mode: QuantizationMode.PerTensorSymmetric);

            // Act & Assert
            Assert.False(parameters.IsPerChannel);
        }

        [Fact]
        public void IsPerChannel_WithPerChannelMode_ReturnsTrue()
        {
            // Arrange
            var scales = new[] { 0.5f, 0.3f, 0.7f };
            var zeroPoints = new[] { 0, -1, 1 };
            var parameters = new QuantizationParameters(
                channelScales: scales,
                channelZeroPoints: zeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);

            // Act & Assert
            Assert.True(parameters.IsPerChannel);
            Assert.Equal(3, parameters.ChannelCount);
        }

        [Fact]
        public void Constructor_WithPerChannelParameters_ThrowsWhenLengthsMismatch()
        {
            // Arrange
            var scales = new[] { 0.5f, 0.3f };
            var zeroPoints = new[] { 0 };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
            {
                new QuantizationParameters(
                    channelScales: scales,
                    channelZeroPoints: zeroPoints);
            });
        }

        [Fact]
        public void Validate_WithValidScale_ReturnsTrue()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.5f, zeroPoint: 0);

            // Act
            var isValid = parameters.Validate();

            // Assert
            Assert.True(isValid);
        }

        [Fact]
        public void Validate_WithInvalidScale_ReturnsFalse()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: -0.5f, zeroPoint: 0);

            // Act
            var isValid = parameters.Validate();

            // Assert
            Assert.False(isValid);
        }

        [Fact]
        public void Validate_WithZeroScale_ReturnsFalse()
        {
            // Arrange
            var parameters = new QuantizationParameters(scale: 0.0f, zeroPoint: 0);

            // Act
            var isValid = parameters.Validate();

            // Assert
            Assert.False(isValid);
        }

        [Fact]
        public void Validate_WithPerChannelAndOneInvalidScale_ReturnsFalse()
        {
            // Arrange
            var scales = new[] { 0.5f, -0.3f, 0.7f };
            var zeroPoints = new[] { 0, -1, 1 };
            var parameters = new QuantizationParameters(
                channelScales: scales,
                channelZeroPoints: zeroPoints);

            // Act
            var isValid = parameters.Validate();

            // Assert
            Assert.False(isValid);
        }

        [Fact]
        public void Equals_WithSameParameters_ReturnsTrue()
        {
            // Arrange
            var param1 = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, min: -1.0f, max: 1.0f);
            var param2 = new QuantizationParameters(scale: 0.5f, zeroPoint: 0, min: -1.0f, max: 1.0f);

            // Act & Assert
            Assert.True(param1.Equals(param2));
            Assert.True(param1 == param2);
        }

        [Fact]
        public void Equals_WithDifferentParameters_ReturnsFalse()
        {
            // Arrange
            var param1 = new QuantizationParameters(scale: 0.5f, zeroPoint: 0);
            var param2 = new QuantizationParameters(scale: 0.3f, zeroPoint: 0);

            // Act & Assert
            Assert.False(param1.Equals(param2));
            Assert.True(param1 != param2);
        }

        [Fact]
        public void GetHashCode_WithSameParameters_ReturnsSameHashCode()
        {
            // Arrange
            var param1 = new QuantizationParameters(scale: 0.5f, zeroPoint: 0);
            var param2 = new QuantizationParameters(scale: 0.5f, zeroPoint: 0);

            // Act & Assert
            Assert.Equal(param1.GetHashCode(), param2.GetHashCode());
        }

        [Fact]
        public void ToString_ReturnsFormattedString()
        {
            // Arrange
            var parameters = new QuantizationParameters(
                scale: 0.5f,
                zeroPoint: 0,
                min: -1.0f,
                max: 1.0f,
                mode: QuantizationMode.PerTensorSymmetric,
                type: QuantizationType.Int8);

            // Act
            var str = parameters.ToString();

            // Assert
            Assert.Contains("PerTensorSymmetric", str);
            Assert.Contains("Int8", str);
            Assert.Contains("Scale: 0.5", str);
            Assert.Contains("ZeroPoint: 0", str);
        }

        [Fact]
        public void ToString_WithPerChannel_ReturnsFormattedStringWithChannelCount()
        {
            // Arrange
            var scales = new[] { 0.5f, 0.3f, 0.7f };
            var zeroPoints = new[] { 0, -1, 1 };
            var parameters = new QuantizationParameters(
                channelScales: scales,
                channelZeroPoints: zeroPoints,
                mode: QuantizationMode.PerChannelSymmetric);

            // Act
            var str = parameters.ToString();

            // Assert
            Assert.Contains("PerChannelSymmetric", str);
            Assert.Contains("Channels: 3", str);
        }
    }
}
