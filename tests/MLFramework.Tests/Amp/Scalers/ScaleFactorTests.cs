using MLFramework.Amp;
using MLFramework.Core;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for ScaleFactor utility class
/// </summary>
public class ScaleFactorTests
{
    #region Constant Values Tests

    [Fact]
    public void None_Equals1()
    {
        Assert.Equal(1.0f, ScaleFactor.None);
    }

    [Fact]
    public void Conservative_Equals256()
    {
        Assert.Equal(256.0f, ScaleFactor.Conservative);
    }

    [Fact]
    public void Moderate_Equals65536()
    {
        Assert.Equal(65536.0f, ScaleFactor.Moderate);
    }

    [Fact]
    public void Aggressive_Equals1048576()
    {
        Assert.Equal(1048576.0f, ScaleFactor.Aggressive);
    }

    #endregion

    #region PowerOfTwo Tests

    [Fact]
    public void PowerOfTwo_WithZero_Returns1()
    {
        var scale = ScaleFactor.PowerOfTwo(0);

        Assert.Equal(1.0f, scale);
    }

    [Fact]
    public void PowerOfTwo_With8_Returns256()
    {
        var scale = ScaleFactor.PowerOfTwo(8);

        Assert.Equal(256.0f, scale);
    }

    [Fact]
    public void PowerOfTwo_With16_Returns65536()
    {
        var scale = ScaleFactor.PowerOfTwo(16);

        Assert.Equal(65536.0f, scale);
    }

    [Fact]
    public void PowerOfTwo_With20_Returns1048576()
    {
        var scale = ScaleFactor.PowerOfTwo(20);

        Assert.Equal(1048576.0f, scale);
    }

    [Fact]
    public void PowerOfTwo_WithNegativeExponent_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ScaleFactor.PowerOfTwo(-1));
    }

    [Fact]
    public void PowerOfTwo_WithLargeExponent_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ScaleFactor.PowerOfTwo(32));
    }

    [Fact]
    public void PowerOfTwo_VariousExponents_ReturnsCorrectValues()
    {
        Assert.Equal(2.0f, ScaleFactor.PowerOfTwo(1));
        Assert.Equal(4.0f, ScaleFactor.PowerOfTwo(2));
        Assert.Equal(8.0f, ScaleFactor.PowerOfTwo(3));
        Assert.Equal(16.0f, ScaleFactor.PowerOfTwo(4));
        Assert.Equal(32.0f, ScaleFactor.PowerOfTwo(5));
    }

    #endregion

    #region GetRecommendedScale Tests

    [Fact]
    public void GetRecommendedScale_ForFloat16_ReturnsModerate()
    {
        var scale = ScaleFactor.GetRecommendedScale(DataType.Float16);

        Assert.Equal(ScaleFactor.Moderate, scale);
    }

    [Fact]
    public void GetRecommendedScale_ForBFloat16_ReturnsConservative()
    {
        var scale = ScaleFactor.GetRecommendedScale(DataType.BFloat16);

        Assert.Equal(ScaleFactor.Conservative, scale);
    }

    [Fact]
    public void GetRecommendedScale_ForFloat32_ReturnsNone()
    {
        var scale = ScaleFactor.GetRecommendedScale(DataType.Float32);

        Assert.Equal(ScaleFactor.None, scale);
    }

    [Fact]
    public void GetRecommendedScale_ForUnsupportedType_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ScaleFactor.GetRecommendedScale(DataType.Int32));
    }

    [Fact]
    public void GetRecommendedScale_ForFloat64_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            ScaleFactor.GetRecommendedScale(DataType.Float64));
    }

    #endregion

    #region IsValidScale Tests

    [Fact]
    public void IsValidScale_WithPositiveValue_ReturnsTrue()
    {
        Assert.True(ScaleFactor.IsValidScale(100.0f));
        Assert.True(ScaleFactor.IsValidScale(1.0f));
        Assert.True(ScaleFactor.IsValidScale(65536.0f));
    }

    [Fact]
    public void IsValidScale_WithZero_ReturnsFalse()
    {
        Assert.False(ScaleFactor.IsValidScale(0.0f));
    }

    [Fact]
    public void IsValidScale_WithNegativeValue_ReturnsFalse()
    {
        Assert.False(ScaleFactor.IsValidScale(-1.0f));
        Assert.False(ScaleFactor.IsValidScale(-100.0f));
    }

    [Fact]
    public void IsValidScale_WithNaN_ReturnsFalse()
    {
        Assert.False(ScaleFactor.IsValidScale(float.NaN));
    }

    [Fact]
    public void IsValidScale_WithPositiveInfinity_ReturnsFalse()
    {
        Assert.False(ScaleFactor.IsValidScale(float.PositiveInfinity));
    }

    [Fact]
    public void IsValidScale_WithNegativeInfinity_ReturnsFalse()
    {
        Assert.False(ScaleFactor.IsValidScale(float.NegativeInfinity));
    }

    [Fact]
    public void IsValidScale_WithMaxValue_ReturnsTrue()
    {
        Assert.True(ScaleFactor.IsValidScale(float.MaxValue - 1.0f));
    }

    [Fact]
    public void IsValidScale_WithVerySmallPositiveValue_ReturnsTrue()
    {
        Assert.True(ScaleFactor.IsValidScale(float.Epsilon));
    }

    #endregion

    #region ClampScale Tests

    [Fact]
    public void ClampScale_WithValidRange_ReturnsOriginal()
    {
        var clamped = ScaleFactor.ClampScale(100.0f, 1.0f, 1000.0f);

        Assert.Equal(100.0f, clamped);
    }

    [Fact]
    public void ClampScale_WithBelowMin_ReturnsMin()
    {
        var clamped = ScaleFactor.ClampScale(0.5f, 1.0f, 1000.0f);

        Assert.Equal(1.0f, clamped);
    }

    [Fact]
    public void ClampScale_WithAboveMax_ReturnsMax()
    {
        var clamped = ScaleFactor.ClampScale(2000.0f, 1.0f, 1000.0f);

        Assert.Equal(1000.0f, clamped);
    }

    [Fact]
    public void ClampScale_WithNaN_ReturnsMin()
    {
        var clamped = ScaleFactor.ClampScale(float.NaN, 1.0f, 1000.0f);

        Assert.Equal(1.0f, clamped);
    }

    [Fact]
    public void ClampScale_WithPositiveInfinity_ReturnsMax()
    {
        var clamped = ScaleFactor.ClampScale(float.PositiveInfinity, 1.0f, 1000.0f);

        Assert.Equal(1000.0f, clamped);
    }

    [Fact]
    public void ClampScale_WithNegativeInfinity_ReturnsMin()
    {
        var clamped = ScaleFactor.ClampScale(float.NegativeInfinity, 1.0f, 1000.0f);

        Assert.Equal(1.0f, clamped);
    }

    [Fact]
    public void ClampScale_WithDefaultParameters_ClampsCorrectly()
    {
        Assert.Equal(1.0f, ScaleFactor.ClampScale(0.5f));
        Assert.Equal(16777216.0f, ScaleFactor.ClampScale(20000000.0f));
        Assert.Equal(100.0f, ScaleFactor.ClampScale(100.0f));
    }

    [Fact]
    public void ClampScale_AtBoundaryValues_ReturnsBoundary()
    {
        var clampedMin = ScaleFactor.ClampScale(1.0f, 1.0f, 1000.0f);
        var clampedMax = ScaleFactor.ClampScale(1000.0f, 1.0f, 1000.0f);

        Assert.Equal(1.0f, clampedMin);
        Assert.Equal(1000.0f, clampedMax);
    }

    #endregion
}
