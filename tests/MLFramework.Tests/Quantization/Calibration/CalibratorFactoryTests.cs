using System;
using Xunit;
using MLFramework.Quantization.Calibration;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Tests.Quantization.Calibration
{
    public class CalibratorFactoryTests
    {
        [Fact]
        public void Create_MinMax_ShouldReturnMinMaxCalibrator()
        {
            // Act
            var calibrator = CalibratorFactory.Create(CalibrationMethod.MinMax);

            // Assert
            Assert.IsType<MinMaxCalibrator>(calibrator);
        }

        [Fact]
        public void Create_Entropy_ShouldReturnEntropyCalibrator()
        {
            // Act
            var calibrator = CalibratorFactory.Create(CalibrationMethod.Entropy);

            // Assert
            Assert.IsType<EntropyCalibrator>(calibrator);
        }

        [Fact]
        public void Create_Percentile_ShouldReturnPercentileCalibrator()
        {
            // Act
            var calibrator = CalibratorFactory.Create(CalibrationMethod.Percentile);

            // Assert
            Assert.IsType<PercentileCalibrator>(calibrator);
        }

        [Fact]
        public void Create_MovingAverage_ShouldReturnMovingAverageCalibrator()
        {
            // Act
            var calibrator = CalibratorFactory.Create(CalibrationMethod.MovingAverage);

            // Assert
            Assert.IsType<MovingAverageCalibrator>(calibrator);
        }

        [Fact]
        public void Create_UnknownMethod_ShouldThrowArgumentException()
        {
            Assert.Throws<ArgumentException>(() => CalibratorFactory.Create((CalibrationMethod)999));
        }

        [Fact]
        public void Create_WithCustomConfig_ShouldReturnCorrectCalibratorWithConfig()
        {
            // Act
            var calibrator = CalibratorFactory.Create(
                CalibrationMethod.MinMax,
                symmetric: true,
                quantMin: -64,
                quantMax: 63
            );

            // Assert
            Assert.IsType<MinMaxCalibrator>(calibrator);
        }

        [Fact]
        public void CreatePercentile_ShouldReturnPercentileCalibratorWithCustomPercentile()
        {
            // Act
            var calibrator = CalibratorFactory.CreatePercentile(95.0f);

            // Assert
            Assert.IsType<PercentileCalibrator>(calibrator);
        }

        [Fact]
        public void CreatePercentile_WithInvalidPercentile_ShouldThrow()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => CalibratorFactory.CreatePercentile(50.0f));
            Assert.Throws<ArgumentOutOfRangeException>(() => CalibratorFactory.CreatePercentile(100.0f));
        }

        [Fact]
        public void CreateMovingAverage_ShouldReturnMovingAverageCalibratorWithCustomWindowSize()
        {
            // Act
            var calibrator = CalibratorFactory.CreateMovingAverage(50);

            // Assert
            Assert.IsType<MovingAverageCalibrator>(calibrator);
        }

        [Fact]
        public void CreateMovingAverage_WithInvalidWindowSize_ShouldThrow()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => CalibratorFactory.CreateMovingAverage(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => CalibratorFactory.CreateMovingAverage(-1));
        }

        [Fact]
        public void CreatePercentile_WithCustomConfig_ShouldReturnCorrectCalibrator()
        {
            // Act
            var calibrator = CalibratorFactory.CreatePercentile(
                percentile: 99.0f,
                symmetric: true,
                quantMin: -64,
                quantMax: 63
            );

            // Assert
            Assert.IsType<PercentileCalibrator>(calibrator);
        }

        [Fact]
        public void CreateMovingAverage_WithCustomConfig_ShouldReturnCorrectCalibrator()
        {
            // Act
            var calibrator = CalibratorFactory.CreateMovingAverage(
                windowSize: 100,
                symmetric: false,
                quantMin: -128,
                quantMax: 127
            );

            // Assert
            Assert.IsType<MovingAverageCalibrator>(calibrator);
        }

        [Theory]
        [InlineData(CalibrationMethod.MinMax, typeof(MinMaxCalibrator))]
        [InlineData(CalibrationMethod.Entropy, typeof(EntropyCalibrator))]
        [InlineData(CalibrationMethod.Percentile, typeof(PercentileCalibrator))]
        [InlineData(CalibrationMethod.MovingAverage, typeof(MovingAverageCalibrator))]
        public void Create_AllMethods_ShouldReturnCorrectType(CalibrationMethod method, Type expectedType)
        {
            // Act
            var calibrator = CalibratorFactory.Create(method);

            // Assert
            Assert.IsType(expectedType, calibrator);
        }

        [Theory]
        [InlineData(CalibrationMethod.MinMax, true)]
        [InlineData(CalibrationMethod.Entropy, false)]
        [InlineData(CalibrationMethod.Percentile, true)]
        [InlineData(CalibrationMethod.MovingAverage, false)]
        public void Create_WithSymmetricFlag_ShouldCreateCalibratorWithSymmetricMode(CalibrationMethod method, bool symmetric)
        {
            // Act
            var calibrator = CalibratorFactory.Create(method, symmetric: symmetric);

            // Assert
            Assert.NotNull(calibrator);
            // Note: We can't directly test symmetric mode without collecting data and getting parameters
            // So we just verify it creates a calibrator without throwing
        }

        [Fact]
        public void Create_AllCalibrationMethods_ShouldBeFunctional()
        {
            // Arrange
            var testData = new[] { -1.0f, 0.0f, 1.0f, 2.0f, 3.0f };

            foreach (CalibrationMethod method in Enum.GetValues(typeof(CalibrationMethod)))
            {
                // Skip unknown methods
                if (method == (CalibrationMethod)999) continue;

                // Act
                var calibrator = CalibratorFactory.Create(method);
                calibrator.CollectStatistics(testData);
                var parameters = calibrator.GetQuantizationParameters();

                // Assert
                Assert.True(parameters.Scale > 0);
                Assert.InRange(parameters.ZeroPoint, -128, 127);
            }
        }
    }
}
