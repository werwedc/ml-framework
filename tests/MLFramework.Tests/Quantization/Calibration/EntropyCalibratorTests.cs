using System;
using System.Linq;
using Xunit;
using MLFramework.Quantization.Calibration;

namespace MLFramework.Tests.Quantization.Calibration
{
    public class EntropyCalibratorTests
    {
        [Fact]
        public void CollectStatistics_WithValidData_ShouldTrackDataPoints()
        {
            // Arrange
            var calibrator = new EntropyCalibrator();
            var data = GenerateNormalDistribution(1000, mean: 0, stdDev: 1);

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void GetQuantizationParameters_WithoutCollectingData_ShouldThrow()
        {
            // Arrange
            var calibrator = new EntropyCalibrator();

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(() => calibrator.GetQuantizationParameters());
            Assert.Contains("No statistics have been collected", exception.Message);
        }

        [Fact]
        public void CollectStatistics_WithNaNValues_ShouldFilterThemOut()
        {
            // Arrange
            var calibrator = new EntropyCalibrator();
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
            var calibrator = new EntropyCalibrator();
            var data = Array.Empty<float>();

            // Act & Assert
            var exception = Record.Exception(() => calibrator.CollectStatistics(data));
            Assert.Null(exception);
        }

        [Fact]
        public void CollectStatistics_WithNullArray_ShouldNotThrow()
        {
            // Arrange
            var calibrator = new EntropyCalibrator();

            // Act & Assert
            var exception = Record.Exception(() => calibrator.CollectStatistics(null!));
            Assert.Null(exception);
        }

        [Fact]
        public void GetQuantizationParameters_WithSmallDataset_ShouldUseMinMax()
        {
            // Arrange
            var calibrator = new EntropyCalibrator();
            var data = new[] { -1.0f, 0.0f, 1.0f }; // Small dataset (< 100 samples)

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void GetQuantizationParameters_SymmetricMode_ShouldHaveZeroPointZero()
        {
            // Arrange
            var calibrator = new EntropyCalibrator(symmetric: true);
            var data = GenerateNormalDistribution(500, mean: 0, stdDev: 1);

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.Equal(0, parameters.ZeroPoint);
        }

        [Fact]
        public void GetQuantizationParameters_WithOutliers_ShouldHandleBetterThanMinMax()
        {
            // Arrange
            var calibrator = new EntropyCalibrator();
            var minMaxCalibrator = new MinMaxCalibrator();

            // Generate data with outliers
            var normalData = GenerateNormalDistribution(950, mean: 0, stdDev: 1);
            var outliers = new[] { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f };
            var data = normalData.Concat(outliers).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            minMaxCalibrator.CollectStatistics(data);

            var entropyParams = calibrator.GetQuantizationParameters();
            var minMaxParams = minMaxCalibrator.GetQuantizationParameters();

            // Assert
            // Entropy calibrator should have smaller scale (better handling of outliers)
            Assert.True(entropyParams.Scale < minMaxParams.Scale);
        }

        [Fact]
        public void Reset_ShouldClearAllStatistics()
        {
            // Arrange
            var calibrator = new EntropyCalibrator();
            var data = GenerateNormalDistribution(200, mean: 0, stdDev: 1);

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
            var calibrator = new EntropyCalibrator();
            var batch1 = GenerateNormalDistribution(200, mean: 0, stdDev: 1);
            var batch2 = GenerateNormalDistribution(200, mean: 0.5, stdDev: 1);

            // Act
            calibrator.CollectStatistics(batch1);
            calibrator.CollectStatistics(batch2);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void GetQuantizationParameters_WithAllSameValues_ShouldHandleEdgeCase()
        {
            // Arrange
            var calibrator = new EntropyCalibrator();
            var data = Enumerable.Repeat(5.0f, 200).ToArray();

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
        }

        [Fact]
        public void GetQuantizationParameters_CustomQuantRange_ShouldUseProvidedRange()
        {
            // Arrange
            var calibrator = new EntropyCalibrator(symmetric: false, quantMin: -64, quantMax: 63);
            var data = GenerateNormalDistribution(500, mean: 0, stdDev: 1);

            // Act
            calibrator.CollectStatistics(data);
            var parameters = calibrator.GetQuantizationParameters();

            // Assert
            Assert.True(parameters.Scale > 0);
            Assert.InRange(parameters.ZeroPoint, -64, 63);
        }

        /// <summary>
        /// Helper method to generate normally distributed data using Box-Muller transform.
        /// </summary>
        private float[] GenerateNormalDistribution(int count, float mean, float stdDev)
        {
            var random = new Random(42); // Fixed seed for reproducibility
            var result = new float[count];

            for (int i = 0; i < count; i++)
            {
                // Box-Muller transform
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();
                double z0 = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                result[i] = mean + (float)(z0 * stdDev);
            }

            return result;
        }
    }
}
