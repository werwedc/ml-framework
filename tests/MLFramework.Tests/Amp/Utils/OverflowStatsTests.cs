using MLFramework.Amp;
using System;
using Xunit;

namespace MLFramework.Tests.Amp.Utils;

/// <summary>
/// Tests for OverflowStats class
/// </summary>
public class OverflowStatsTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidParameters_CreatesStats()
    {
        var stats = new OverflowStats(
            totalGradients: 10,
            overflowCount: 2,
            overflowParameters: new[] { "param1", "param2" });

        Assert.Equal(10, stats.TotalGradients);
        Assert.Equal(2, stats.OverflowCount);
        Assert.Equal(2, stats.OverflowParameters.Count);
        Assert.Equal(0.2f, stats.OverflowRate, precision: 4);
    }

    [Fact]
    public void Constructor_WithNoOverflow_CreatesValidStats()
    {
        var stats = new OverflowStats(
            totalGradients: 5,
            overflowCount: 0,
            overflowParameters: Array.Empty<string>());

        Assert.Equal(5, stats.TotalGradients);
        Assert.Equal(0, stats.OverflowCount);
        Assert.Empty(stats.OverflowParameters);
        Assert.Equal(0.0f, stats.OverflowRate);
        Assert.False(stats.HasOverflow);
    }

    [Fact]
    public void Constructor_WithAllOverflow_CreatesValidStats()
    {
        var stats = new OverflowStats(
            totalGradients: 3,
            overflowCount: 3,
            overflowParameters: new[] { "param1", "param2", "param3" });

        Assert.Equal(3, stats.TotalGradients);
        Assert.Equal(3, stats.OverflowCount);
        Assert.Equal(1.0f, stats.OverflowRate);
        Assert.True(stats.HasOverflow);
    }

    [Fact]
    public void Constructor_WithZeroTotalGradients_CreatesValidStats()
    {
        var stats = new OverflowStats(
            totalGradients: 0,
            overflowCount: 0,
            overflowParameters: Array.Empty<string>());

        Assert.Equal(0, stats.TotalGradients);
        Assert.Equal(0, stats.OverflowCount);
        Assert.Equal(0.0f, stats.OverflowRate);
        Assert.False(stats.HasOverflow);
    }

    [Fact]
    public void Constructor_WithNegativeTotalGradients_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new OverflowStats(
                totalGradients: -1,
                overflowCount: 0,
                overflowParameters: Array.Empty<string>()));
    }

    [Fact]
    public void Constructor_WithNegativeOverflowCount_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new OverflowStats(
                totalGradients: 10,
                overflowCount: -1,
                overflowParameters: Array.Empty<string>()));
    }

    [Fact]
    public void Constructor_WithOverflowCountGreaterThanTotal_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            new OverflowStats(
                totalGradients: 5,
                overflowCount: 6,
                overflowParameters: Array.Empty<string>()));
    }

    [Fact]
    public void Constructor_WithNullOverflowParameters_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new OverflowStats(
                totalGradients: 10,
                overflowCount: 2,
                overflowParameters: null));
    }

    #endregion

    #region Properties Tests

    [Fact]
    public void HasOverflow_WithOverflowCountGreaterThanZero_ReturnsTrue()
    {
        var stats = new OverflowStats(
            totalGradients: 10,
            overflowCount: 1,
            overflowParameters: new[] { "param1" });

        Assert.True(stats.HasOverflow);
    }

    [Fact]
    public void HasOverflow_WithZeroOverflowCount_ReturnsFalse()
    {
        var stats = new OverflowStats(
            totalGradients: 10,
            overflowCount: 0,
            overflowParameters: Array.Empty<string>());

        Assert.False(stats.HasOverflow);
    }

    [Fact]
    public void OverflowRate_WithNonZeroTotal_CalculatesCorrectly()
    {
        var stats = new OverflowStats(
            totalGradients: 100,
            overflowCount: 25,
            overflowParameters: new string[25]);

        Assert.Equal(0.25f, stats.OverflowRate, precision: 4);
    }

    [Fact]
    public void OverflowRate_WithZeroTotal_ReturnsZero()
    {
        var stats = new OverflowStats(
            totalGradients: 0,
            overflowCount: 0,
            overflowParameters: Array.Empty<string>());

        Assert.Equal(0.0f, stats.OverflowRate);
    }

    #endregion

    #region ToString Tests

    [Fact]
    public void ToString_WithValidStats_ReturnsFormattedString()
    {
        var stats = new OverflowStats(
            totalGradients: 10,
            overflowCount: 2,
            overflowParameters: new[] { "param1", "param2" });

        var result = stats.ToString();

        Assert.Contains("Overflow Statistics", result);
        Assert.Contains("Total Gradients: 10", result);
        Assert.Contains("Overflow Count: 2", result);
        Assert.Contains("Overflow Rate:", result);
        Assert.Contains("param1", result);
        Assert.Contains("param2", result);
    }

    [Fact]
    public void ToString_WithNoOverflow_ReturnsCorrectFormat()
    {
        var stats = new OverflowStats(
            totalGradients: 5,
            overflowCount: 0,
            overflowParameters: Array.Empty<string>());

        var result = stats.ToString();

        Assert.Contains("Total Gradients: 5", result);
        Assert.Contains("Overflow Count: 0", result);
        Assert.Contains("Overflow Rate: 0.00%", result);
        Assert.DoesNotContain("Overflow Parameters", result);
    }

    #endregion
}
