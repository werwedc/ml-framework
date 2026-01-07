using System;
using System.Linq;
using Xunit;
using MLFramework.Quantization.Calibration;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Tests.Quantization.Calibration
{
    public class MinMaxCalibratorTests
    {
        [Fact]
        public void CollectStatistics_WithValidData_ShouldTrackMinMax()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator();
            var data = new[] { -1.0f, 0.0f, 1.0f, 2.0f, 3.0f };

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
            Assert.InRange(parameters.ZeroPoint, -128, 127);
        }

        [Fact]
        public void CollectStatistics_WithNaNValues_ShouldFilterThemOut()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator();
            var data = new[] { -1.0f, float.NaN, 1.0f, 2.0f, float.PositiveInfinity };

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void CollectStatistics_WithEmptyArray_ShouldNotThrow()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator();
            var data = Array.Empty<float>();

            // Act & Assert
            var exception = Record.Exception(() => calibrator.CollectStatistics(data));
            Assert.Null(exception);
        }

        [Fact]
        public void GetQuantizationParameters_WithoutCollectingData_ShouldThrow()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator();

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(() => calibrator.GetQuantizationParameters());
            Assert.Contains("No statistics have been collected", exception.Message);
        }

        [Fact]
        public void GetQuantizationParameters_WithSameValues_ShouldHandleEdgeCase()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator();
            var data = Enumerable.Repeat(5.0f, 100).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void CollectStatistics_WithMultipleBatches_ShouldTrackOverallMinMax()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator();
            var batch1 = new[] { -5.0f, -2.0f, 0.0f };
            var batch2 = new[] { 3.0f, 5.0f, 10.0f };

            // Act
            calibrator.CollectStatistics(batch1);
            calibrator.CollectStatistics(batch2);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void Reset_ShouldClearAllStatistics()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator();
            var data = new[] { -1.0f, 0.0f, 1.0f };

            // Act
            calibrator.CollectStatistics(data);
            calibrator.Reset();

            // Assert
            Assert.Throws<InvalidOperationException>(() => calibrator.GetQuantizationParameters());
        }

        [Fact]
        public void GetQuantizationParameters_SymmetricMode_ShouldHaveZeroPointZero()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator(symmetric: true);
            var data = new[] { -3.0f, -1.0f, 0.0f, 1.0f, 3.0f };

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.Equal(0, parameters.ZeroPoint);
        }

        [Fact]
        public void CollectStatistics_WithNullArray_ShouldNotThrow()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator();

            // Act & Assert
            var exception = Record.Exception(() => calibrator.CollectStatistics(null!));
            Assert.Null(exception);
        }

        [Fact]
        public void GetQuantizationParameters_CustomQuantRange_ShouldUseProvidedRange()
        {
            // Arrange
            var calibrator = new MinMaxCalibrator(symmetric: false, quantMin: -64, quantMax: 63);
            var data = new[] { -1.0f, 0.0f, 1.0f };

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
            Assert.InRange(parameters.ZeroPoint, -64, 63);
        }
    }
}
