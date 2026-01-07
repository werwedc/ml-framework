using System;
using System.Linq;
using Xunit;
using MLFramework.Quantization.Calibration;

namespace MLFramework.Tests.Quantization.Calibration
{
    public class PercentileCalibratorTests
    {
        [Fact]
        public void CollectStatistics_WithValidData_ShouldTrackDataPoints()
        {
            // Arrange
            var calibrator = new PercentileCalibrator();
            var data = Enumerable.Range(0, 1000).Select(i => (float)i).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void CollectStatistics_WithOutliers_ShouldExcludeThem()
        {
            // Arrange
            var calibrator = new PercentileCalibrator(percentile: 99.0f);
            var data = Enumerable.Range(0, 950)
                .Select(i => (float)i / 10.0f)
                .Concat(new[] { 1000.0f, 2000.0f, 3000.0f, 4000.0f, 5000.0f })
                .ToArray();

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            // The scale should be much smaller than if we included outliers
            // With outliers, max would be 5000, with percentile (99%), it should be ~94
            // So scale should be roughly 50x smaller
            Assert.True(parameters.Scale < 0.5f);
        }

        [Fact]
        public void Constructor_WithInvalidPercentile_ShouldThrow()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new PercentileCalibrator(percentile: 50.0f));
            Assert.Throws<ArgumentOutOfRangeException>(() => new PercentileCalibrator(percentile: 100.0f));
            Assert.Throws<ArgumentOutOfRangeException>(() => new PercentileCalibrator(percentile: 0.0f));
        }

        [Fact]
        public void GetQuantizationParameters_WithoutCollectingData_ShouldThrow()
        {
            // Arrange
            var calibrator = new PercentileCalibrator();

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(() => calibrator.GetQuantizationParameters());
            Assert.Contains("No statistics have been collected", exception.Message);
        }

        [Fact]
        public void CollectStatistics_WithNaNValues_ShouldFilterThemOut()
        {
            // Arrange
            var calibrator = new PercentileCalibrator();
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
            var calibrator = new PercentileCalibrator();
            var data = Array.Empty<float>();

            // Act & Assert
            var exception = Record.Exception(() => calibrator.CollectStatistics(data));
            Assert.Null(exception);
        }

        [Fact]
        public void CollectStatistics_WithNullArray_ShouldNotThrow()
        {
            // Arrange
            var calibrator = new PercentileCalibrator();

            // Act & Assert
            var exception = Record.Exception(() => calibrator.CollectStatistics(null!));
            Assert.Null(exception);
        }

        [Fact]
        public void GetQuantizationParameters_SymmetricMode_ShouldHaveZeroPointZero()
        {
            // Arrange
            var calibrator = new PercentileCalibrator(symmetric: true);
            var data = Enumerable.Range(-50, 101).Select(i => (float)i).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.Equal(0, parameters.ZeroPoint);
        }

        [Fact]
        public void GetQuantizationParameters_CustomPercentile_ShouldUseProvidedValue()
        {
            // Arrange
            var calibrator95 = new PercentileCalibrator(percentile: 95.0f);
            var calibrator99 = new PercentileCalibrator(percentile: 99.0f);
            var data = Enumerable.Range(0, 1000).Select(i => (float)i).ToArray();

            // Act
            calibrator95.CollectStatistics(data);
            calibrator99.CollectStatistics(data);
            var params95 = calibrator95.GetQuantizationParameters();
            var params99 = calibrator99.GetQuantizationParameters();

            // Assert
            // 99% percentile should have larger scale than 95% (includes more range)
            Assert.True(params99.Scale > params95.Scale);
        }

        [Fact]
        public void Reset_ShouldClearAllStatistics()
        {
            // Arrange
            var calibrator = new PercentileCalibrator();
            var data = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            calibrator.Reset();

            // Assert
            Assert.Throws<InvalidOperationException>(() => calibrator.GetQuantizationParameters());
        }

        [Fact]
        public void CollectStatistics_WithMultipleBatches_ShouldCombineAllData()
        {
            // Arrange
            var calibrator = new PercentileCalibrator();
            var batch1 = Enumerable.Range(0, 50).Select(i => (float)i).ToArray();
            var batch2 = Enumerable.Range(50, 50).Select(i => (float)i).ToArray();

            // Act
            calibrator.CollectStatistics(batch1);
            calibrator.CollectStatistics(batch2);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }
    }
}
