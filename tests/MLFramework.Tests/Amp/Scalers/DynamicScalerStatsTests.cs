using MLFramework.Amp;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for DynamicScalerStats class
/// </summary>
public class DynamicScalerStatsTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithAllParameters_CreatesStats()
    {
        var stats = new DynamicScalerStats(
            currentScale: 1000.0f,
            totalOverflows: 5,
            totalSuccessfulIterations: 95,
            scaleIncreaseCount: 3,
            scaleDecreaseCount: 2,
            minScaleReached: 500.0f,
            maxScaleReached: 2000.0f);

        Assert.Equal(1000.0f, stats.CurrentScale);
        Assert.Equal(5, stats.TotalOverflows);
        Assert.Equal(95, stats.TotalSuccessfulIterations);
        Assert.Equal(3, stats.ScaleIncreaseCount);
        Assert.Equal(2, stats.ScaleDecreaseCount);
        Assert.Equal(500.0f, stats.MinScaleReached);
        Assert.Equal(2000.0f, stats.MaxScaleReached);
    }

    #endregion

    #region SuccessRate Property Tests

    [Fact]
    public void SuccessRate_WithBothSuccessAndFailure_ReturnsCorrectRate()
    {
        var stats = new DynamicScalerStats(
            currentScale: 1000.0f,
            totalOverflows: 5,
            totalSuccessfulIterations: 95,
            scaleIncreaseCount: 3,
            scaleDecreaseCount: 2,
            minScaleReached: 500.0f,
            maxScaleReached: 2000.0f);

        Assert.Equal(0.95f, stats.SuccessRate, precision: 2);
    }

    [Fact]
    public void SuccessRate_WithOnlySuccess_ReturnsOne()
    {
        var stats = new DynamicScalerStats(
            currentScale: 1000.0f,
            totalOverflows: 0,
            totalSuccessfulIterations: 100,
            scaleIncreaseCount: 3,
            scaleDecreaseCount: 0,
            minScaleReached: 1000.0f,
            maxScaleReached: 1000.0f);

        Assert.Equal(1.0f, stats.SuccessRate);
    }

    [Fact]
    public void SuccessRate_WithOnlyFailure_ReturnsZero()
    {
        var stats = new DynamicScalerStats(
            currentScale: 1000.0f,
            totalOverflows: 10,
            totalSuccessfulIterations: 0,
            scaleIncreaseCount: 0,
            scaleDecreaseCount: 3,
            minScaleReached: 500.0f,
            maxScaleReached: 1000.0f);

        Assert.Equal(0.0f, stats.SuccessRate);
    }

    [Fact]
    public void SuccessRate_WithNoIterations_ReturnsOne()
    {
        var stats = new DynamicScalerStats(
            currentScale: 1000.0f,
            totalOverflows: 0,
            totalSuccessfulIterations: 0,
            scaleIncreaseCount: 0,
            scaleDecreaseCount: 0,
            minScaleReached: 1000.0f,
            maxScaleReached: 1000.0f);

        Assert.Equal(1.0f, stats.SuccessRate);
    }

    #endregion

    #region ToString Method Tests

    [Fact]
    public void ToString_ReturnsFormattedString()
    {
        var stats = new DynamicScalerStats(
            currentScale: 1000.0f,
            totalOverflows: 5,
            totalSuccessfulIterations: 95,
            scaleIncreaseCount: 3,
            scaleDecreaseCount: 2,
            minScaleReached: 500.0f,
            maxScaleReached: 2000.0f);

        var str = stats.ToString();

        Assert.Contains("DynamicScalerStats", str);
        Assert.Contains("1000", str);
        Assert.Contains("5", str);
        Assert.Contains("95", str);
    }

    #endregion
}
