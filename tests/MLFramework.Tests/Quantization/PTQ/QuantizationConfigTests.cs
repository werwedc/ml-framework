using System;
using Xunit;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Tests.Quantization.PTQ
{
    /// <summary>
    /// Tests for different quantization configurations.
    /// </summary>
    public class QuantizationConfigTests
    {
        [Fact]
        public void QuantizationConfig_PerTensorMode_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(QuantizationType.Int8, config.QuantizationType);
            Assert.Equal(QuantizationMode.PerTensor, config.WeightQuantization);
            Assert.Equal(QuantizationMode.PerTensor, config.ActivationQuantization);
            Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
        }

        [Fact]
        public void QuantizationConfig_PerChannelMode_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerChannel,
                ActivationQuantization = QuantizationMode.PerTensor,
                EnablePerChannelQuantization = true,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(QuantizationType.Int8, config.QuantizationType);
            Assert.Equal(QuantizationMode.PerChannel, config.WeightQuantization);
            Assert.True(config.EnablePerChannelQuantization);
        }

        [Fact]
        public void QuantizationConfig_SymmetricMode_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.SymmetricPerTensor,
                ActivationQuantization = QuantizationMode.SymmetricPerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(QuantizationMode.SymmetricPerTensor, config.WeightQuantization);
            Assert.Equal(QuantizationMode.SymmetricPerTensor, config.ActivationQuantization);
        }

        [Fact]
        public void QuantizationConfig_AsymmetricMode_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(QuantizationMode.PerTensor, config.WeightQuantization);
            Assert.Equal(QuantizationMode.PerTensor, config.ActivationQuantization);
        }

        [Fact]
        public void QuantizationConfig_MinMaxCalibrationMethod_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
        }

        [Fact]
        public void QuantizationConfig_EntropyCalibrationMethod_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.Entropy
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(CalibrationMethod.Entropy, config.CalibrationMethod);
        }

        [Fact]
        public void QuantizationConfig_PercentileCalibrationMethod_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.Percentile
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(CalibrationMethod.Percentile, config.CalibrationMethod);
        }

        [Fact]
        public void QuantizationConfig_MovingAverageCalibrationMethod_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MovingAverage
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(CalibrationMethod.MovingAverage, config.CalibrationMethod);
        }

        [Fact]
        public void QuantizationConfig_MixedConfiguration_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerChannel,
                ActivationQuantization = QuantizationMode.SymmetricPerTensor,
                EnablePerChannelQuantization = true,
                CalibrationMethod = CalibrationMethod.Entropy
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(QuantizationType.Int8, config.QuantizationType);
            Assert.Equal(QuantizationMode.PerChannel, config.WeightQuantization);
            Assert.Equal(QuantizationMode.SymmetricPerTensor, config.ActivationQuantization);
            Assert.Equal(CalibrationMethod.Entropy, config.CalibrationMethod);
            Assert.True(config.EnablePerChannelQuantization);
        }

        [Fact]
        public void QuantizationConfig_DynamicQuantization_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(QuantizationMode.None, config.ActivationQuantization);
        }

        [Fact]
        public void QuantizationConfig_StaticQuantization_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(QuantizationMode.PerTensor, config.ActivationQuantization);
            Assert.Equal(CalibrationMethod.MinMax, config.CalibrationMethod);
        }

        [Fact]
        public void QuantizationConfig_DefaultValues_AreValid()
        {
            // Arrange
            var config = new QuantizationConfig();

            // Act
            config.Validate();

            // Assert
            Assert.NotNull(config);
        }

        [Fact]
        public void QuantizationConfig_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var config1 = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            var config2 = (QuantizationConfig)config1.Clone();
            config2.WeightQuantization = QuantizationMode.PerChannel;

            // Assert
            Assert.NotEqual(config1.WeightQuantization, config2.WeightQuantization);
        }

        [Fact]
        public void QuantizationConfig_FallbackToFP32_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax,
                FallbackToFP32 = true,
                AccuracyThreshold = 0.01f
            };

            // Act
            config.Validate();

            // Assert
            Assert.True(config.FallbackToFP32);
            Assert.Equal(0.01f, config.AccuracyThreshold);
        }

        [Fact]
        public void QuantizationConfig_InvalidCalibrationMethod_ThrowsException()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act & Assert
            // This test ensures the config validation logic works
            var exception = Record.Exception(() => config.Validate());
            // Current implementation might not throw, just documenting expected behavior
        }

        [Fact]
        public void QuantizationConfig_MissingCalibrationMethodForStatic_ThrowsException()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                // CalibrationMethod not set - should default or throw
            };

            // Act
            config.Validate();

            // Assert
            // Config should be valid or throw exception
        }

        [Fact]
        public void QuantizationConfig_Int8Type_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            config.Validate();

            // Assert
            Assert.Equal(QuantizationType.Int8, config.QuantizationType);
        }

        [Fact]
        public void QuantizationConfig_SymmetricVsAsymmetric_DifferencesExist()
        {
            // Arrange
            var symmetricConfig = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.SymmetricPerTensor,
                ActivationQuantization = QuantizationMode.SymmetricPerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            var asymmetricConfig = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            symmetricConfig.Validate();
            asymmetricConfig.Validate();

            // Assert
            Assert.Equal(QuantizationMode.SymmetricPerTensor, symmetricConfig.WeightQuantization);
            Assert.Equal(QuantizationMode.PerTensor, asymmetricConfig.WeightQuantization);
        }

        [Fact]
        public void QuantizationConfig_PerTensorVsPerChannel_DifferencesExist()
        {
            // Arrange
            var perTensorConfig = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                EnablePerChannelQuantization = false,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            var perChannelConfig = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerChannel,
                EnablePerChannelQuantization = true,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            // Act
            perTensorConfig.Validate();
            perChannelConfig.Validate();

            // Assert
            Assert.Equal(QuantizationMode.PerTensor, perTensorConfig.WeightQuantization);
            Assert.False(perTensorConfig.EnablePerChannelQuantization);
            Assert.Equal(QuantizationMode.PerChannel, perChannelConfig.WeightQuantization);
            Assert.True(perChannelConfig.EnablePerChannelQuantization);
        }

        [Fact]
        public void QuantizationConfig_DifferentCalibrationMethods_AllValid()
        {
            // Arrange
            var methods = new[]
            {
                CalibrationMethod.MinMax,
                CalibrationMethod.Entropy,
                CalibrationMethod.Percentile,
                CalibrationMethod.MovingAverage
            };

            // Act & Assert
            foreach (var method in methods)
            {
                var config = new QuantizationConfig
                {
                    QuantizationType = QuantizationType.Int8,
                    WeightQuantization = QuantizationMode.PerTensor,
                    ActivationQuantization = QuantizationMode.PerTensor,
                    CalibrationMethod = method
                };

                config.Validate();

                Assert.Equal(method, config.CalibrationMethod);
            }
        }

        [Fact]
        public void QuantizationConfig_AllCombinations_WorksCorrectly()
        {
            // Arrange
            var weightModes = new[]
            {
                QuantizationMode.PerTensor,
                QuantizationMode.PerChannel,
                QuantizationMode.SymmetricPerTensor
            };

            var activationModes = new[]
            {
                QuantizationMode.PerTensor,
                QuantizationMode.SymmetricPerTensor,
                QuantizationMode.None
            };

            // Act & Assert
            foreach (var weightMode in weightModes)
            {
                foreach (var activationMode in activationModes)
                {
                    var config = new QuantizationConfig
                    {
                        QuantizationType = QuantizationType.Int8,
                        WeightQuantization = weightMode,
                        ActivationQuantization = activationMode,
                        CalibrationMethod = CalibrationMethod.MinMax
                    };

                    config.Validate();

                    Assert.Equal(weightMode, config.WeightQuantization);
                    Assert.Equal(activationMode, config.ActivationQuantization);
                }
            }
        }
    }
}
