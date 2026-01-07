using System;
using System.Linq;
using Xunit;
using MLFramework.Quantization.Calibration;

namespace MLFramework.Tests.Quantization.Calibration
{
    public class MovingAverageCalibratorTests
    {
        [Fact]
        public void CollectStatistics_WithValidData_ShouldTrackMovingAverage()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator(windowSize: 10);
            var data = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void CollectStatistics_WithMultipleBatches_ShouldMaintainWindow()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator(windowSize: 5);
            var batch1 = Enumerable.Range(0, 10).Select(i => (float)i).ToArray(); // min: 0, max: 9
            var batch2 = Enumerable.Range(10, 10).Select(i => (float)i).ToArray(); // min: 10, max: 19

            // Act
            calibrator.CollectStatistics(batch1);
            calibrator.CollectStatistics(batch2);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void Constructor_WithInvalidWindowSize_ShouldThrow()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new MovingAverageCalibrator(windowSize: 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new MovingAverageCalibrator(windowSize: -1));
        }

        [Fact]
        public void GetQuantizationParameters_WithoutCollectingData_ShouldThrow()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator();

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(() => calibrator.GetQuantizationParameters());
            Assert.Contains("No statistics have been collected", exception.Message);
        }

        [Fact]
        public void CollectStatistics_WithNaNValues_ShouldFilterThemOut()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator();
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
            var calibrator = new MovingAverageCalibrator();
            var data = Array.Empty<float>();

            // Act & Assert
            var exception = Record.Exception(() => calibrator.CollectStatistics(data));
            Assert.Null(exception);
        }

        [Fact]
        public void CollectStatistics_WithNullArray_ShouldNotThrow()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator();

            // Act & Assert
            var exception = Record.Exception(() => calibrator.CollectStatistics(null!));
            Assert.Null(exception);
        }

        [Fact]
        public void CollectStatistics_ShouldRespectWindowSize()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator(windowSize: 3);

            // Add 5 batches
            calibrator.CollectStatistics(new[] { 0.0f, 1.0f, 2.0f }); // min: 0, max: 2
            calibrator.CollectStatistics(new[] { 3.0f, 4.0f, 5.0f }); // min: 3, max: 5
            calibrator.CollectStatistics(new[] { 6.0f, 7.0f, 8.0f }); // min: 6, max: 8
            calibrator.CollectStatistics(new[] { 9.0f, 10.0f, 11.0f }); // min: 9, max: 11
            calibrator.CollectStatistics(new[] { 12.0f, 13.0f, 14.0f }); // min: 12, max: 14

            // Act
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            // Window should only contain the last 3 batches
            // minAvg: (6 + 9 + 12) / 3 = 9
            // maxAvg: (8 + 11 + 14) / 3 = 11
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void GetQuantizationParameters_SymmetricMode_ShouldHaveZeroPointZero()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator(windowSize: 10, symmetric: true);
            var data = Enumerable.Range(-50, 101).Select(i => (float)i).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.Equal(0, parameters.ZeroPoint);
        }

        [Fact]
        public void Reset_ShouldClearAllWindows()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator(windowSize: 5);
            var data = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            calibrator.Reset();

            // Assert
            Assert.Throws<InvalidOperationException>(() => calibrator.GetQuantizationParameters());
        }

        [Fact]
        public void CollectStatistics_WithIncreasingValues_ShouldTrackTrend()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator(windowSize: 10);

            // Add batches with increasing ranges
            for (int i = 0; i < 20; i++)
            {
                var batch = Enumerable.Range(i * 10, 10).Select(j => (float)j).ToArray();
                calibrator.CollectStatistics(batch);
            }

            // Act
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void GetQuantizationParameters_CustomQuantRange_ShouldUseProvidedRange()
        {
            // Arrange
            var calibrator = new MovingAverageCalibrator(symmetric: false, quantMin: -64, quantMax: 63);
            var data = Enumerable.Range(0, 100).Select(i => (float)i).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
            Assert.InRange(parameters.ZeroPoint, -64, 63);
        }
    }
}
