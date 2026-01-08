using MLFramework.Visualization.Histograms.Statistics;

namespace MLFramework.Visualization.Tests.Histograms;

/// <summary>
/// Unit tests for statistical calculators
/// </summary>
public class StatisticsTests
{
    #region QuantileCalculator Tests

    [Fact]
    public void QuantileCalculator_Calculate_WithValidData_ReturnsCorrectQuantile()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };

        // Act
        var median = QuantileCalculator.Calculate(data, 0.5f);

        // Assert
        Assert.Equal(5.5f, median, precision: 1);
    }

    [Fact]
    public void QuantileCalculator_Calculate_WithSingleValue_ReturnsThatValue()
    {
        // Arrange
        var data = new float[] { 42.0f };

        // Act
        var result = QuantileCalculator.Calculate(data, 0.5f);

        // Assert
        Assert.Equal(42.0f, result);
    }

    [Fact]
    public void QuantileCalculator_Calculate_WithInvalidQuantile_ThrowsException()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => QuantileCalculator.Calculate(data, 1.5f));
        Assert.Throws<ArgumentOutOfRangeException>(() => QuantileCalculator.Calculate(data, -0.5f));
    }

    [Fact]
    public void QuantileCalculator_Calculate_WithNullData_ThrowsException()
    {
        // Arrange
        float[] data = null!;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => QuantileCalculator.Calculate(data, 0.5f));
    }

    [Fact]
    public void QuantileCalculator_CalculateMultiple_WithValidData_ReturnsCorrectQuantiles()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
        var quantiles = new[] { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f };

        // Act
        var results = QuantileCalculator.CalculateMultiple(data, quantiles);

        // Assert
        Assert.Equal(5, results.Length);
        Assert.Equal(2.8f, results[0], precision: 1); // 10th percentile
        Assert.Equal(3.25f, results[1], precision: 1); // 25th percentile
        Assert.Equal(5.5f, results[2], precision: 1); // 50th percentile
        Assert.Equal(7.75f, results[3], precision: 1); // 75th percentile
        Assert.Equal(8.2f, results[4], precision: 1); // 90th percentile
    }

    #endregion

    #region MomentsCalculator Tests

    [Fact]
    public void MomentsCalculator_CalculateSkewness_WithSymmetricData_ReturnsNearZero()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var mean = 3.0f;
        var std = 1.41421354f;

        // Act
        var skewness = MomentsCalculator.CalculateSkewness(data, mean, std);

        // Assert
        Assert.Equal(0.0f, skewness, precision: 6);
    }

    [Fact]
    public void MomentsCalculator_CalculateSkewness_WithRightSkewedData_ReturnsPositive()
    {
        // Arrange
        var data = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 10.0f };

        // Act
        var skewness = MomentsCalculator.CalculateSkewness(data);

        // Assert
        Assert.True(skewness > 0, "Skewness should be positive for right-skewed data");
    }

    [Fact]
    public void MomentsCalculator_CalculateSkewness_WithAllSameValues_ReturnsZero()
    {
        // Arrange
        var data = new float[] { 5.0f, 5.0f, 5.0f, 5.0f, 5.0f };

        // Act
        var skewness = MomentsCalculator.CalculateSkewness(data);

        // Assert
        Assert.Equal(0.0f, skewness);
    }

    [Fact]
    public void MomentsCalculator_CalculateKurtosis_WithNormalDistribution_ReturnsNearZero()
    {
        // Arrange
        var random = new Random(42);
        var data = new float[1000];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)random.NextGaussian();
        }

        // Act
        var kurtosis = MomentsCalculator.CalculateKurtosis(data);

        // Assert
        // Kurtosis of normal distribution should be close to 0 (excess kurtosis)
        Assert.True(Math.Abs(kurtosis) < 1.0f, $"Kurtosis {kurtosis} should be close to 0 for normal distribution");
    }

    [Fact]
    public void MomentsCalculator_CalculateKurtosis_WithHeavyTails_ReturnsPositive()
    {
        // Arrange
        var data = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 100.0f, -100.0f };

        // Act
        var kurtosis = MomentsCalculator.CalculateKurtosis(data);

        // Assert
        Assert.True(kurtosis > 0, "Kurtosis should be positive for heavy-tailed distribution");
    }

    [Fact]
    public void MomentsCalculator_CalculateBoth_ReturnsCorrectValues()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        var (skewness, kurtosis) = MomentsCalculator.CalculateBoth(data);

        // Assert
        Assert.Equal(0.0f, skewness, precision: 6);
    }

    #endregion

    #region OutlierDetector Tests

    [Fact]
    public void OutlierDetector_DetectOutliersByStd_WithOutliers_DetectsCorrectly()
    {
        // Arrange
        var data = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 100.0f };

        // Act
        var (count, indices) = OutlierDetector.DetectOutliersByStd(data, 3.0f);

        // Assert
        Assert.Equal(1, count);
        Assert.Contains(5, indices); // Index of 100.0
    }

    [Fact]
    public void OutlierDetector_DetectOutliersByStd_WithNoOutliers_ReturnsZero()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        var (count, indices) = OutlierDetector.DetectOutliersByStd(data, 3.0f);

        // Assert
        Assert.Equal(0, count);
        Assert.Empty(indices);
    }

    [Fact]
    public void OutlierDetector_DetectOutliersByIQR_WithOutliers_DetectsCorrectly()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 100.0f };

        // Act
        var (count, indices) = OutlierDetector.DetectOutliersByIQR(data, 1.5f);

        // Assert
        Assert.Equal(1, count);
        Assert.Contains(5, indices); // Index of 100.0
    }

    [Fact]
    public void OutlierDetector_CountDeadNeurons_WithZeros_CountsCorrectly()
    {
        // Arrange
        var data = new float[] { 0.0f, 1.0f, 0.0f, 2.0f, 0.0f };

        // Act
        var count = OutlierDetector.CountDeadNeurons(data);

        // Assert
        Assert.Equal(3, count);
    }

    [Fact]
    public void OutlierDetector_CountDeadNeurons_WithThreshold_CountsNearZeros()
    {
        // Arrange
        var data = new float[] { 0.0f, 0.001f, 0.01f, 1.0f, 2.0f };

        // Act
        var count = OutlierDetector.CountDeadNeurons(data, 0.01f);

        // Assert
        Assert.Equal(3, count);
    }

    [Fact]
    public void OutlierDetector_GetOutlierBounds_ReturnsCorrectBounds()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        var (lower, upper) = OutlierDetector.GetOutlierBounds(data, 2.0f);

        // Assert
        Assert.True(lower < 1.0f);
        Assert.True(upper > 5.0f);
    }

    [Fact]
    public void OutlierDetector_GetOutlierValues_ReturnsCorrectValues()
    {
        // Arrange
        var data = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 100.0f };

        // Act
        var values = OutlierDetector.GetOutlierValues(data, 3.0f);

        // Assert
        Assert.Single(values);
        Assert.Contains(100.0f, values);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Statistics_WithLargeArray_CompletesQuickly()
    {
        // Arrange
        var random = new Random(42);
        var data = new float[1000000];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)random.NextGaussian();
        }

        // Act
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var quantiles = QuantileCalculator.CalculateMultiple(data, new[] { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f });
        var skewness = MomentsCalculator.CalculateSkewness(data);
        var kurtosis = MomentsCalculator.CalculateKurtosis(data);
        var (outlierCount, _) = OutlierDetector.DetectOutliersByStd(data);
        stopwatch.Stop();

        // Assert
        Assert.Equal(5, quantiles.Length);
        Assert.True(stopwatch.ElapsedMilliseconds < 200, $"Statistics calculation took {stopwatch.ElapsedMilliseconds}ms, expected < 200ms");
    }

    [Fact]
    public void Statistics_MultipleCalculators_ConsistentResults()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };

        // Act
        var median = QuantileCalculator.Calculate(data, 0.5f);
        var quantiles = QuantileCalculator.CalculateMultiple(data, new[] { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f });

        // Assert
        Assert.Equal(quantiles[2], median);
    }

    #endregion
}
