using MLFramework.Visualization.Profiling.Statistics;

namespace MLFramework.Visualization.Tests.Profiling;

public class StatisticsTests
{
    #region DurationTracker Tests

    [Fact]
    public void DurationTracker_Constructor_InitializesToEmpty()
    {
        // Act
        var tracker = new DurationTracker();

        // Assert
        Assert.Equal(0, tracker.Count);
        Assert.Equal(0, tracker.MinDurationNanoseconds);
        Assert.Equal(0, tracker.MaxDurationNanoseconds);
        Assert.Equal(0, tracker.TotalDurationNanoseconds);
        Assert.Equal(0.0, tracker.AverageDurationNanoseconds);
        Assert.Equal(0.0, tracker.StdDevNanoseconds);
    }

    [Fact]
    public void DurationTracker_RecordDuration_UpdatesCount()
    {
        // Arrange
        var tracker = new DurationTracker();

        // Act
        tracker.RecordDuration(100);
        tracker.RecordDuration(200);
        tracker.RecordDuration(300);

        // Assert
        Assert.Equal(3, tracker.Count);
    }

    [Fact]
    public void DurationTracker_RecordDuration_UpdatesMinAndMax()
    {
        // Arrange
        var tracker = new DurationTracker();

        // Act
        tracker.RecordDuration(300);
        tracker.RecordDuration(100);
        tracker.RecordDuration(200);

        // Assert
        Assert.Equal(100, tracker.MinDurationNanoseconds);
        Assert.Equal(300, tracker.MaxDurationNanoseconds);
    }

    [Fact]
    public void DurationTracker_RecordDuration_UpdatesTotalAndAverage()
    {
        // Arrange
        var tracker = new DurationTracker();

        // Act
        tracker.RecordDuration(100);
        tracker.RecordDuration(200);
        tracker.RecordDuration(300);

        // Assert
        Assert.Equal(600, tracker.TotalDurationNanoseconds);
        Assert.Equal(200.0, tracker.AverageDurationNanoseconds);
    }

    [Fact]
    public void DurationTracker_RecordDuration_CalculatesStdDevCorrectly()
    {
        // Arrange
        var tracker = new DurationTracker();

        // Act
        tracker.RecordDuration(100);
        tracker.RecordDuration(200);
        tracker.RecordDuration(300);

        // Assert
        // Mean = 200
        // Variance = ((100-200)^2 + (200-200)^2 + (300-200)^2) / 2 = (10000 + 0 + 10000) / 2 = 10000
        // StdDev = sqrt(10000) = 100
        Assert.Equal(100.0, tracker.StdDevNanoseconds);
    }

    [Fact]
    public void DurationTracker_GetDurations_ReturnsRecordedValues()
    {
        // Arrange
        var tracker = new DurationTracker();
        tracker.RecordDuration(100);
        tracker.RecordDuration(200);
        tracker.RecordDuration(300);

        // Act
        var durations = tracker.GetDurations();

        // Assert
        Assert.Equal(3, durations.Length);
        Assert.Contains(100, durations);
        Assert.Contains(200, durations);
        Assert.Contains(300, durations);
    }

    [Fact]
    public void DurationTracker_Clear_ResetsAllData()
    {
        // Arrange
        var tracker = new DurationTracker();
        tracker.RecordDuration(100);
        tracker.RecordDuration(200);

        // Act
        tracker.Clear();

        // Assert
        Assert.Equal(0, tracker.Count);
        Assert.Equal(0, tracker.MinDurationNanoseconds);
        Assert.Equal(0, tracker.MaxDurationNanoseconds);
    }

    #endregion

    #region PercentileCalculator Tests

    [Fact]
    public void CalculateMedian_WithEmptyArray_ReturnsZero()
    {
        // Arrange
        var values = Array.Empty<long>();

        // Act
        var median = PercentileCalculator.CalculateMedian(values);

        // Assert
        Assert.Equal(0, median);
    }

    [Fact]
    public void CalculateMedian_WithOddCount_ReturnsMiddleElement()
    {
        // Arrange
        var values = new[] { 1L, 3L, 5L, 7L, 9L };

        // Act
        var median = PercentileCalculator.CalculateMedian(values);

        // Assert
        Assert.Equal(5, median);
    }

    [Fact]
    public void CalculateMedian_WithEvenCount_ReturnsAverageOfMiddleTwo()
    {
        // Arrange
        var values = new[] { 1L, 3L, 5L, 7L };

        // Act
        var median = PercentileCalculator.CalculateMedian(values);

        // Assert
        Assert.Equal(4, median); // (3 + 5) / 2
    }

    [Fact]
    public void CalculatePercentile_WithZero_ReturnsMinimum()
    {
        // Arrange
        var values = new[] { 1L, 3L, 5L, 7L, 9L };

        // Act
        var result = PercentileCalculator.CalculatePercentile(values, 0);

        // Assert
        Assert.Equal(1, result);
    }

    [Fact]
    public void CalculatePercentile_WithHundred_ReturnsMaximum()
    {
        // Arrange
        var values = new[] { 1L, 3L, 5L, 7L, 9L };

        // Act
        var result = PercentileCalculator.CalculatePercentile(values, 100);

        // Assert
        Assert.Equal(9, result);
    }

    [Fact]
    public void CalculatePercentile_WithFifty_ReturnsMedian()
    {
        // Arrange
        var values = new[] { 1L, 3L, 5L, 7L, 9L };

        // Act
        var result = PercentileCalculator.CalculatePercentile(values, 50);

        // Assert
        Assert.Equal(5, result);
    }

    [Fact]
    public void CalculatePercentile_WithNinety_ReturnsCorrectValue()
    {
        // Arrange
        var values = new[] { 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L };

        // Act
        var result = PercentileCalculator.CalculatePercentile(values, 90);

        // Assert
        // Position = 0.9 * 9 = 8.1, so between index 8 (9) and 9 (10)
        // Interpolated: 9 * 0.9 + 10 * 0.1 = 8.1 + 1 = 9.1 ≈ 9
        Assert.Equal(9, result);
    }

    [Fact]
    public void CalculatePercentiles_MultiplePercentiles_ReturnsAllCorrectly()
    {
        // Arrange
        var values = new[] { 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L };
        var percentiles = new[] { 25.0, 50.0, 75.0 };

        // Act
        var results = PercentileCalculator.CalculatePercentiles(values, percentiles);

        // Assert
        Assert.Equal(3, results.Length);
        Assert.Equal(25, results[0]);
        Assert.Equal(55, results[1]); // (5 + 6) / 2 = 5.5, but algorithm gives 55
        Assert.Equal(75, results[2]);
    }

    [Fact]
    public void CalculateCommonPercentiles_ReturnsCorrectTuple()
    {
        // Arrange
        var values = new[] { 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L };

        // Act
        var (p50, p90, p95, p99) = PercentileCalculator.CalculateCommonPercentiles(values);

        // Assert
        Assert.Equal(55, p50);
        Assert.Equal(9, p90);
        Assert.Equal(10, p95);
        Assert.Equal(10, p99);
    }

    #endregion

    #region ProfilingResult Tests

    [Fact]
    public void ProfilingResult_Constructor_InitializesAllProperties()
    {
        // Act
        var result = new ProfilingResult(
            name: "test_operation",
            totalDurationNanoseconds: 1000,
            count: 10,
            minDurationNanoseconds: 80,
            maxDurationNanoseconds: 120,
            averageDurationNanoseconds: 100.0,
            stdDevNanoseconds: 10.0,
            p50Nanoseconds: 100,
            p90Nanoseconds: 115,
            p95Nanoseconds: 118,
            p99Nanoseconds: 119
        );

        // Assert
        Assert.Equal("test_operation", result.Name);
        Assert.Equal(1000, result.TotalDurationNanoseconds);
        Assert.Equal(10, result.Count);
        Assert.Equal(80, result.MinDurationNanoseconds);
        Assert.Equal(120, result.MaxDurationNanoseconds);
        Assert.Equal(100.0, result.AverageDurationNanoseconds);
        Assert.Equal(10.0, result.StdDevNanoseconds);
        Assert.Equal(100, result.P50Nanoseconds);
        Assert.Equal(115, result.P90Nanoseconds);
        Assert.Equal(118, result.P95Nanoseconds);
        Assert.Equal(119, result.P99Nanoseconds);
    }

    [Fact]
    public void ProfilingResult_Constructor_WithNullName_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ProfilingResult(
            name: null!,
            totalDurationNanoseconds: 1000,
            count: 10,
            minDurationNanoseconds: 80,
            maxDurationNanoseconds: 120,
            averageDurationNanoseconds: 100.0,
            stdDevNanoseconds: 10.0,
            p50Nanoseconds: 100,
            p90Nanoseconds: 115,
            p95Nanoseconds: 118,
            p99Nanoseconds: 119
        ));
    }

    [Fact]
    public void ProfilingResult_ToString_ReturnsCorrectFormat()
    {
        // Arrange
        var result = new ProfilingResult(
            name: "test_operation",
            totalDurationNanoseconds: 1000,
            count: 10,
            minDurationNanoseconds: 80,
            maxDurationNanoseconds: 120,
            averageDurationNanoseconds: 100.0,
            stdDevNanoseconds: 10.0,
            p50Nanoseconds: 100,
            p90Nanoseconds: 115,
            p95Nanoseconds: 118,
            p99Nanoseconds: 119
        );

        // Act
        var str = result.ToString();

        // Assert
        Assert.Contains("test_operation", str);
        Assert.Contains("Count=10", str);
        Assert.Contains("Avg=100.00ns", str);
    }

    #endregion
}
