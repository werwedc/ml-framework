using MLFramework.Visualization.Histograms;

namespace MLFramework.Visualization.Tests.Histograms;

/// <summary>
/// Unit tests for HistogramLogger
/// </summary>
public class HistogramLoggerTests
{
    [Fact]
    public void LogHistogram_WithValidData_LogsSuccessfully()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_histogram";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        logger.LogHistogram(name, values);

        // Assert
        var histogram = logger.GetHistogram(name, 0);
        Assert.NotNull(histogram);
        Assert.Equal(name, histogram.Name);
        Assert.Equal(values.Length, histogram.TotalCount);
    }

    [Fact]
    public void LogHistogram_WithCustomStep_LogsSuccessfully()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_histogram";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        long step = 42;

        // Act
        logger.LogHistogram(name, values, step);

        // Assert
        var histogram = logger.GetHistogram(name, step);
        Assert.NotNull(histogram);
        Assert.Equal(step, histogram.Step);
    }

    [Fact]
    public void LogHistogram_WithAutoIncrement_LogsCorrectly()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_histogram";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        logger.LogHistogram(name, values);
        logger.LogHistogram(name, values);
        logger.LogHistogram(name, values);

        // Assert
        var histogram0 = logger.GetHistogram(name, 0);
        var histogram1 = logger.GetHistogram(name, 1);
        var histogram2 = logger.GetHistogram(name, 2);

        Assert.NotNull(histogram0);
        Assert.NotNull(histogram1);
        Assert.NotNull(histogram2);
        Assert.Equal(0, histogram0.Step);
        Assert.Equal(1, histogram1.Step);
        Assert.Equal(2, histogram2.Step);
    }

    [Fact]
    public void LogHistogram_WithNullName_ThrowsArgumentException()
    {
        // Arrange
        var logger = new HistogramLogger();
        string name = null!;
        var values = new float[] { 1.0f, 2.0f, 3.0f };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => logger.LogHistogram(name, values));
    }

    [Fact]
    public void LogHistogram_WithNullValues_ThrowsArgumentException()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_histogram";
        float[] values = null!;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => logger.LogHistogram(name, values));
    }

    [Fact]
    public void LogDistribution_WithValidData_LogsSuccessfully()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_distribution";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        logger.LogDistribution(name, values);

        // Assert
        var distribution = logger.GetDistribution(name, 0);
        Assert.NotNull(distribution);
        Assert.Equal(name, distribution.Name);
        Assert.Equal(values.Length, distribution.TotalCount);
    }

    [Fact]
    public void LogDistribution_CalculatesExtendedStatistics()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_distribution";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        logger.LogDistribution(name, values);

        // Assert
        var distribution = logger.GetDistribution(name, 0);
        Assert.NotNull(distribution);
        Assert.True(distribution.Skewness != 0 || distribution.TotalCount < 3);
        Assert.True(distribution.Kurtosis != 0 || distribution.TotalCount < 4);
        Assert.NotNull(distribution.Histogram);
    }

    [Fact]
    public void LogDistribution_DetectsDeadNeurons()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_distribution";
        var values = new float[] { 0.0f, 1.0f, 2.0f, 0.0f, 3.0f };

        // Act
        logger.LogDistribution(name, values);

        // Assert
        var distribution = logger.GetDistribution(name, 0);
        Assert.NotNull(distribution);
        Assert.Equal(2, distribution.DeadNeuronCount);
    }

    [Fact]
    public void LogDistribution_DetectsOutliers()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_distribution";
        var values = new float[] { 1.0f, 1.0f, 1.0f, 100.0f, 1.0f };

        // Act
        logger.LogDistribution(name, values);

        // Assert
        var distribution = logger.GetDistribution(name, 0);
        Assert.NotNull(distribution);
        Assert.True(distribution.OutlierCount > 0);
    }

    [Fact]
    public void GetHistogram_WithNonExistentName_ReturnsNull()
    {
        // Arrange
        var logger = new HistogramLogger();

        // Act
        var histogram = logger.GetHistogram("non_existent", 0);

        // Assert
        Assert.Null(histogram);
    }

    [Fact]
    public void GetHistogramsOverTime_ReturnsOrderedResults()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_histogram";
        var values = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        logger.LogHistogram(name, values, 10);
        logger.LogHistogram(name, values, 5);
        logger.LogHistogram(name, values, 15);

        var histograms = logger.GetHistogramsOverTime(name).ToList();

        // Assert
        Assert.Equal(3, histograms.Count);
        Assert.Equal(5, histograms[0].Step);
        Assert.Equal(10, histograms[1].Step);
        Assert.Equal(15, histograms[2].Step);
    }

    [Fact]
    public void GetHistogramNames_ReturnsAllNames()
    {
        // Arrange
        var logger = new HistogramLogger();
        var values = new float[] { 1.0f, 2.0f, 3.0f };

        // Act
        logger.LogHistogram("hist1", values);
        logger.LogHistogram("hist2", values);
        logger.LogHistogram("hist3", values);

        var names = logger.GetHistogramNames().ToList();

        // Assert
        Assert.Equal(3, names.Count);
        Assert.Contains("hist1", names);
        Assert.Contains("hist2", names);
        Assert.Contains("hist3", names);
    }

    [Fact]
    public void Clear_RemovesAllData()
    {
        // Arrange
        var logger = new HistogramLogger();
        var values = new float[] { 1.0f, 2.0f, 3.0f };

        logger.LogHistogram("hist1", values);
        logger.LogDistribution("dist1", values);

        // Act
        logger.Clear();

        // Assert
        Assert.Empty(logger.GetHistogramNames());
        Assert.Empty(logger.GetDistributionNames());
    }

    [Fact]
    public void Clear_WithName_RemovesOnlyThatData()
    {
        // Arrange
        var logger = new HistogramLogger();
        var values = new float[] { 1.0f, 2.0f, 3.0f };

        logger.LogHistogram("hist1", values);
        logger.LogHistogram("hist2", values);

        // Act
        logger.Clear("hist1");

        // Assert
        Assert.Null(logger.GetHistogram("hist1", 0));
        Assert.NotNull(logger.GetHistogram("hist2", 0));
    }

    [Fact]
    public async Task LogHistogramAsync_LogsSuccessfully()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_histogram";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        await logger.LogHistogramAsync(name, values);

        // Assert
        var histogram = logger.GetHistogram(name, 0);
        Assert.NotNull(histogram);
        Assert.Equal(name, histogram.Name);
    }

    [Fact]
    public async Task LogDistributionAsync_LogsSuccessfully()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_distribution";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        await logger.LogDistributionAsync(name, values);

        // Assert
        var distribution = logger.GetDistribution(name, 0);
        Assert.NotNull(distribution);
        Assert.Equal(name, distribution.Name);
    }

    [Fact]
    public void DefaultBinConfig_IsUsed()
    {
        // Arrange
        var logger = new HistogramLogger();
        logger.DefaultBinConfig = new HistogramBinConfig { BinCount = 10 };
        var name = "test_histogram";
        var values = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

        // Act
        logger.LogHistogram(name, values);

        // Assert
        var histogram = logger.GetHistogram(name, 0);
        Assert.NotNull(histogram);
        Assert.Equal(10, histogram.BinCounts.Length);
    }

    [Fact]
    public void WithLargeArray_LogsQuickly()
    {
        // Arrange
        var logger = new HistogramLogger();
        var name = "test_histogram";
        var values = new float[1000000];
        var random = new Random(42);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)random.NextGaussian();
        }

        // Act
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        logger.LogHistogram(name, values);
        stopwatch.Stop();

        // Assert
        var histogram = logger.GetHistogram(name, 0);
        Assert.NotNull(histogram);
        Assert.True(stopwatch.ElapsedMilliseconds < 100, $"Logging took {stopwatch.ElapsedMilliseconds}ms, expected < 100ms");
    }
}
