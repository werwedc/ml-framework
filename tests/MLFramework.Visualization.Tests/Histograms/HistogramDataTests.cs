using MLFramework.Visualization.Histograms;
using MLFramework.Visualization.Histograms.Statistics;

namespace MLFramework.Visualization.Tests.Histograms;

/// <summary>
/// Unit tests for HistogramData
/// </summary>
public class HistogramDataTests
{
    [Fact]
    public void Create_WithValidData_ReturnsValidHistogram()
    {
        // Arrange
        var name = "test_histogram";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var config = new HistogramBinConfig { BinCount = 5 };

        // Act
        var histogram = HistogramData.Create(name, values, config);

        // Assert
        Assert.NotNull(histogram);
        Assert.Equal(name, histogram.Name);
        Assert.Equal(values.Length, histogram.TotalCount);
        Assert.Equal(5, histogram.BinCounts.Length);
        Assert.Equal(6, histogram.BinEdges.Length);
        Assert.Equal(1.0f, histogram.Min);
        Assert.Equal(5.0f, histogram.Max);
    }

    [Fact]
    public void Create_WithNullValues_ThrowsArgumentException()
    {
        // Arrange
        var name = "test_histogram";
        float[] values = null!;
        var config = new HistogramBinConfig();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => HistogramData.Create(name, values, config));
    }

    [Fact]
    public void Create_WithEmptyValues_ThrowsArgumentException()
    {
        // Arrange
        var name = "test_histogram";
        var values = Array.Empty<float>();
        var config = new HistogramBinConfig();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => HistogramData.Create(name, values, config));
    }

    [Fact]
    public void Create_WithNullName_ThrowsArgumentException()
    {
        // Arrange
        string name = null!;
        var values = new float[] { 1.0f, 2.0f, 3.0f };
        var config = new HistogramBinConfig();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => HistogramData.Create(name, values, config));
    }

    [Fact]
    public void Create_WithAllSameValues_HandlesCorrectly()
    {
        // Arrange
        var name = "test_histogram";
        var values = new float[] { 5.0f, 5.0f, 5.0f, 5.0f, 5.0f };
        var config = new HistogramBinConfig { BinCount = 3 };

        // Act
        var histogram = HistogramData.Create(name, values, config);

        // Assert
        Assert.NotNull(histogram);
        Assert.Equal(5.0f, histogram.Min);
        Assert.Equal(5.0f, histogram.Max);
        Assert.Equal(5.0f, histogram.Mean);
        Assert.Equal(0.0f, histogram.Std);
    }

    [Fact]
    public void Create_CalculatesCorrectStatistics()
    {
        // Arrange
        var name = "test_histogram";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var config = new HistogramBinConfig { BinCount = 5 };

        // Act
        var histogram = HistogramData.Create(name, values, config);

        // Assert
        Assert.Equal(3.0f, histogram.Mean, precision: 6);
        Assert.True(histogram.Std > 0); // Should be non-zero
        Assert.Equal(1.0f, histogram.Min);
        Assert.Equal(5.0f, histogram.Max);
    }

    [Fact]
    public void Create_CalculatesCorrectQuantiles()
    {
        // Arrange
        var name = "test_histogram";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
        var config = new HistogramBinConfig { BinCount = 5 };

        // Act
        var histogram = HistogramData.Create(name, values, config);

        // Assert
        Assert.Equal(5, histogram.Quantiles.Length);
        Assert.Equal(2.8f, histogram.Quantiles[0], precision: 1); // 10th percentile
        Assert.Equal(3.25f, histogram.Quantiles[1], precision: 1); // 25th percentile
        Assert.Equal(5.5f, histogram.Quantiles[2], precision: 1); // 50th percentile (median)
        Assert.Equal(7.75f, histogram.Quantiles[3], precision: 1); // 75th percentile
        Assert.Equal(8.2f, histogram.Quantiles[4], precision: 1); // 90th percentile
    }

    [Fact]
    public void Create_WithLinearBins_CreatesCorrectBinEdges()
    {
        // Arrange
        var name = "test_histogram";
        var values = new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
        var config = new HistogramBinConfig { BinCount = 10, UseLogScale = false };

        // Act
        var histogram = HistogramData.Create(name, values, config);

        // Assert
        Assert.Equal(11, histogram.BinEdges.Length);
        Assert.Equal(0.0f, histogram.BinEdges[0]);
        Assert.Equal(10.0f, histogram.BinEdges[10]);
        Assert.Equal(1.0f, histogram.BinEdges[1]);
    }

    [Fact]
    public void Create_WithLogBins_CreatesCorrectBinEdges()
    {
        // Arrange
        var name = "test_histogram";
        var values = new float[] { 1.0f, 10.0f, 100.0f, 1000.0f };
        var config = new HistogramBinConfig { BinCount = 3, UseLogScale = true };

        // Act
        var histogram = HistogramData.Create(name, values, config);

        // Assert
        Assert.Equal(4, histogram.BinEdges.Length);
        Assert.Equal(1.0f, histogram.BinEdges[0], precision: 2);
        Assert.Equal(1000.0f, histogram.BinEdges[3], precision: 2);
    }

    [Fact]
    public void Create_WithLargeArray_CompletesQuickly()
    {
        // Arrange
        var name = "test_histogram";
        var values = new float[1000000];
        var random = new Random(42);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)random.NextGaussian();
        }
        var config = new HistogramBinConfig { BinCount = 50 };

        // Act
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var histogram = HistogramData.Create(name, values, config);
        stopwatch.Stop();

        // Assert
        Assert.NotNull(histogram);
        Assert.True(stopwatch.ElapsedMilliseconds < 100, $"Histogram creation took {stopwatch.ElapsedMilliseconds}ms, expected < 100ms");
    }
}
