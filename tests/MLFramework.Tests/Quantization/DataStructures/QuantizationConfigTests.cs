using MLFramework.Quantization.DataStructures;
using Xunit;

namespace MLFramework.Tests.Quantization
{
    public class QuantizationConfigTests
    {
        [Fact]
        public void Constructor_WithDefaultValues_SetsCorrectDefaults()
        {
            // Arrange & Act
            var config = new QuantizationConfig();

            // Assert
            Assert.Equal(QuantizationMode.PerTensorSymmetric, config.WeightQuantization);
            Assert.Equal(QuantizationMode.PerTensorSymmetric, config.ActivationQuantization);
            Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
            Assert.Equal(32, config.CalibrationBatchSize);
            Assert.Equal(QuantizationType.Int8, config.QuantizationType);
            Assert.True(config.FallbackToFP32);
            Assert.Equal(0.01f, config.AccuracyThreshold);
            Assert.True(config.EnablePerChannelQuantization);
            Assert.False(config.EnableMixedPrecision);
        }

        [Fact]
        public void Validate_WithValidConfiguration_ReturnsTrue()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                CalibrationBatchSize = 32,
                AccuracyThreshold = 0.01f
            };

            // Act
            var isValid = config.Validate();

            // Assert
            Assert.True(isValid);
        }

        [Fact]
        public void Validate_WithInvalidCalibrationBatchSize_ThrowsException()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                CalibrationBatchSize = 0
            };

            // Act & Assert
            var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
            Assert.Contains("CalibrationBatchSize must be greater than 0", ex.Message);
        }

        [Fact]
        public void Validate_WithNegativeCalibrationBatchSize_ThrowsException()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                CalibrationBatchSize = -10
            };

            // Act & Assert
            var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
            Assert.Contains("CalibrationBatchSize must be greater than 0", ex.Message);
        }

        [Fact]
        public void Validate_WithAccuracyThresholdBelowZero_ThrowsException()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                AccuracyThreshold = -0.01f
            };

            // Act & Assert
            var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
            Assert.Contains("AccuracyThreshold must be between 0.0 and 1.0", ex.Message);
        }

        [Fact]
        public void Validate_WithAccuracyThresholdAboveOne_ThrowsException()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                AccuracyThreshold = 1.5f
            };

            // Act & Assert
            var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
            Assert.Contains("AccuracyThreshold must be between 0.0 and 1.0", ex.Message);
        }

        [Fact]
        public void Validate_WithPerChannelEnabledButPerTensorModes_ThrowsException()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                EnablePerChannelQuantization = true,
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorAsymmetric
            };

            // Act & Assert
            var ex = Assert.Throws<InvalidOperationException>(() => config.Validate());
            Assert.Contains("EnablePerChannelQuantization is true but both weight and activation modes are per-tensor", ex.Message);
        }

        [Fact]
        public void CreateDefault_ReturnsDefaultConfiguration()
        {
            // Act
            var config = QuantizationConfig.CreateDefault();

            // Assert
            Assert.Equal(QuantizationMode.PerTensorSymmetric, config.WeightQuantization);
            Assert.Equal(QuantizationMode.PerTensorSymmetric, config.ActivationQuantization);
            Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
            Assert.Equal(32, config.CalibrationBatchSize);
            Assert.Equal(QuantizationType.Int8, config.QuantizationType);
            Assert.True(config.FallbackToFP32);
            Assert.Equal(0.01f, config.AccuracyThreshold);
            Assert.False(config.EnablePerChannelQuantization);
            Assert.False(config.EnableMixedPrecision);
        }

        [Fact]
        public void CreateAggressive_ReturnsAggressiveConfiguration()
        {
            // Act
            var config = QuantizationConfig.CreateAggressive();

            // Assert
            Assert.Equal(QuantizationMode.PerChannelSymmetric, config.WeightQuantization);
            Assert.Equal(QuantizationMode.PerTensorSymmetric, config.ActivationQuantization);
            Assert.Equal(CalibrationMethod.Entropy, config.CalibrationMethod);
            Assert.Equal(64, config.CalibrationBatchSize);
            Assert.Equal(QuantizationType.Int8, config.QuantizationType);
            Assert.False(config.FallbackToFP32);
            Assert.Equal(0.05f, config.AccuracyThreshold);
            Assert.True(config.EnablePerChannelQuantization);
            Assert.False(config.EnableMixedPrecision);
        }

        [Fact]
        public void CreateConservative_ReturnsConservativeConfiguration()
        {
            // Act
            var config = QuantizationConfig.CreateConservative();

            // Assert
            Assert.Equal(QuantizationMode.PerTensorSymmetric, config.WeightQuantization);
            Assert.Equal(QuantizationMode.PerTensorSymmetric, config.ActivationQuantization);
            Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
            Assert.Equal(16, config.CalibrationBatchSize);
            Assert.Equal(QuantizationType.Int8, config.QuantizationType);
            Assert.True(config.FallbackToFP32);
            Assert.Equal(0.005f, config.AccuracyThreshold);
            Assert.False(config.EnablePerChannelQuantization);
            Assert.True(config.EnableMixedPrecision);
        }

        [Fact]
        public void ToString_ReturnsFormattedString()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorAsymmetric,
                CalibrationMethod = CalibrationMethod.MinMax,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8
            };

            // Act
            var str = config.ToString();

            // Assert
            Assert.Contains("PerTensorSymmetric", str);
            Assert.Contains("PerTensorAsymmetric", str);
            Assert.Contains("MinMax", str);
            Assert.Contains("BatchSize: 32", str);
        }

        [Theory]
        [InlineData(0.0f)]
        [InlineData(0.5f)]
        [InlineData(1.0f)]
        public void Validate_WithValidAccuracyThreshold_ReturnsTrue(float threshold)
        {
            // Arrange
            var config = new QuantizationConfig
            {
                AccuracyThreshold = threshold
            };

            // Act
            var isValid = config.Validate();

            // Assert
            Assert.True(isValid);
        }

        [Theory]
        [InlineData(CalibrationMethod.MinMax)]
        [InlineData(CalibrationMethod.Entropy)]
        [InlineData(CalibrationMethod.Percentile)]
        [InlineData(CalibrationMethod.MovingAverage)]
        public void Constructor_WithAllCalibrationMethods_SetsCorrectly(CalibrationMethod method)
        {
            // Arrange & Act
            var config = new QuantizationConfig
            {
                CalibrationMethod = method
            };

            // Assert
            Assert.Equal(method, config.CalibrationMethod);
        }

        [Theory]
        [InlineData(QuantizationMode.PerTensorSymmetric)]
        [InlineData(QuantizationMode.PerTensorAsymmetric)]
        [InlineData(QuantizationMode.PerChannelSymmetric)]
        [InlineData(QuantizationMode.PerChannelAsymmetric)]
        public void Constructor_WithAllWeightQuantizationModes_SetsCorrectly(QuantizationMode mode)
        {
            // Arrange & Act
            var config = new QuantizationConfig
            {
                WeightQuantization = mode
            };

            // Assert
            Assert.Equal(mode, config.WeightQuantization);
        }

        [Theory]
        [InlineData(QuantizationType.Int8)]
        [InlineData(QuantizationType.UInt8)]
        public void Constructor_WithAllQuantizationTypes_SetsCorrectly(QuantizationType type)
        {
            // Arrange & Act
            var config = new QuantizationConfig
            {
                QuantizationType = type
            };

            // Assert
            Assert.Equal(type, config.QuantizationType);
        }
    }
}
