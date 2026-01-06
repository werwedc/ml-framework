using MLFramework.Amp;
using Xunit;

namespace MLFramework.Tests.Amp.Datatypes;

/// <summary>
/// Tests for Half (16-bit floating point) type
/// </summary>
public class HalfTests
{
    [Fact]
    public void Constructor_FromFloat_PreservesValueWithinRange()
    {
        var half = new Half(1.5f);
        var backToFloat = (float)half;

        Assert.InRange(Math.Abs(backToFloat - 1.5f), 0, 0.001f);
    }

    [Fact]
    public void Constructor_FromDouble_PreservesValueWithinRange()
    {
        var half = new Half(2.75);
        var backToFloat = (float)half;

        Assert.InRange(Math.Abs(backToFloat - 2.75), 0, 0.01);
    }

    [Fact]
    public void Constructor_FromRawBits_CreatesCorrectValue()
    {
        var half = new Half((ushort)0x3C00); // Represents 1.0
        var backToFloat = (float)half;

        Assert.InRange(Math.Abs(backToFloat - 1.0f), 0, 0.001f);
    }

    [Fact]
    public void ConversionToFloat_WorksCorrectly()
    {
        var half = new Half(3.14f);
        float result = half;

        Assert.InRange(Math.Abs(result - 3.14f), 0, 0.01);
    }

    [Fact]
    public void ConversionToDouble_WorksCorrectly()
    {
        var half = new Half(2.718f);
        double result = half;

        Assert.InRange(Math.Abs(result - 2.718), 0, 0.01);
    }

    [Fact]
    public void AdditionOperator_WorksCorrectly()
    {
        var h1 = new Half(2.5f);
        var h2 = new Half(1.5f);
        var result = h1 + h2;

        Assert.InRange(Math.Abs((float)result - 4.0f), 0, 0.01);
    }

    [Fact]
    public void SubtractionOperator_WorksCorrectly()
    {
        var h1 = new Half(5.0f);
        var h2 = new Half(2.0f);
        var result = h1 - h2;

        Assert.InRange(Math.Abs((float)result - 3.0f), 0, 0.01);
    }

    [Fact]
    public void MultiplicationOperator_WorksCorrectly()
    {
        var h1 = new Half(3.0f);
        var h2 = new Half(4.0f);
        var result = h1 * h2;

        Assert.InRange(Math.Abs((float)result - 12.0f), 0, 0.01);
    }

    [Fact]
    public void DivisionOperator_WorksCorrectly()
    {
        var h1 = new Half(10.0f);
        var h2 = new Half(2.0f);
        var result = h1 / h2;

        Assert.InRange(Math.Abs((float)result - 5.0f), 0, 0.01);
    }

    [Fact]
    public void EqualityOperator_WorksForEqualValues()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(1.5f);

        Assert.True(h1 == h2);
    }

    [Fact]
    public void EqualityOperator_WorksForDifferentValues()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(2.5f);

        Assert.False(h1 == h2);
    }

    [Fact]
    public void InequalityOperator_WorksCorrectly()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(2.5f);

        Assert.True(h1 != h2);
    }

    [Fact]
    public void LessThanOperator_WorksCorrectly()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(2.5f);

        Assert.True(h1 < h2);
        Assert.False(h2 < h1);
    }

    [Fact]
    public void GreaterThanOperator_WorksCorrectly()
    {
        var h1 = new Half(2.5f);
        var h2 = new Half(1.5f);

        Assert.True(h1 > h2);
        Assert.False(h2 > h1);
    }

    [Fact]
    public void LessThanOrEqualOperator_WorksCorrectly()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(1.5f);
        var h3 = new Half(2.5f);

        Assert.True(h1 <= h2);
        Assert.True(h1 <= h3);
    }

    [Fact]
    public void GreaterThanOrEqualOperator_WorksCorrectly()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(1.5f);
        var h3 = new Half(0.5f);

        Assert.True(h1 >= h2);
        Assert.True(h1 >= h3);
    }

    [Fact]
    public void CompareTo_WorksCorrectly()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(2.5f);
        var h3 = new Half(1.5f);

        Assert.True(h1.CompareTo(h2) < 0);
        Assert.True(h2.CompareTo(h1) > 0);
        Assert.True(h1.CompareTo(h3) == 0);
    }

    [Fact]
    public void Equals_ReturnsTrueForEqualValues()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(1.5f);

        Assert.True(h1.Equals(h2));
    }

    [Fact]
    public void Equals_ReturnsFalseForDifferentValues()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(2.5f);

        Assert.False(h1.Equals(h2));
    }

    [Fact]
    public void Equals_ObjectOverride_WorksCorrectly()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(1.5f);

        Assert.True(h1.Equals((object)h2));
    }

    [Fact]
    public void GetHashCode_ReturnsSameForEqualValues()
    {
        var h1 = new Half(1.5f);
        var h2 = new Half(1.5f);

        Assert.Equal(h1.GetHashCode(), h2.GetHashCode());
    }

    [Fact]
    public void IsNaN_ReturnsTrueForNaN()
    {
        var half = Half.NaN;

        Assert.True(half.IsNaN);
    }

    [Fact]
    public void IsNaN_ReturnsFalseForNormalValue()
    {
        var half = new Half(1.5f);

        Assert.False(half.IsNaN);
    }

    [Fact]
    public void IsInfinity_ReturnsTrueForInfinity()
    {
        var posInf = Half.PositiveInfinity;
        var negInf = Half.NegativeInfinity;

        Assert.True(posInf.IsInfinity);
        Assert.True(negInf.IsInfinity);
    }

    [Fact]
    public void IsInfinity_ReturnsFalseForNormalValue()
    {
        var half = new Half(1.5f);

        Assert.False(half.IsInfinity);
    }

    [Fact]
    public void IsPositiveInfinity_ReturnsTrueForPositiveInfinity()
    {
        var half = Half.PositiveInfinity;

        Assert.True(half.IsPositiveInfinity);
    }

    [Fact]
    public void IsPositiveInfinity_ReturnsFalseForNegativeInfinity()
    {
        var half = Half.NegativeInfinity;

        Assert.False(half.IsPositiveInfinity);
    }

    [Fact]
    public void IsNegativeInfinity_ReturnsTrueForNegativeInfinity()
    {
        var half = Half.NegativeInfinity;

        Assert.True(half.IsNegativeInfinity);
    }

    [Fact]
    public void IsNegativeInfinity_ReturnsFalseForPositiveInfinity()
    {
        var half = Half.PositiveInfinity;

        Assert.False(half.IsNegativeInfinity);
    }

    [Fact]
    public void Epsilon_IsDefined()
    {
        var epsilon = Half.Epsilon;

        Assert.True(epsilon > new Half(0));
    }

    [Fact]
    public void MaxValue_IsPositive()
    {
        var max = Half.MaxValue;
        var asFloat = (float)max;

        Assert.True(asFloat > 0);
        Assert.True(float.IsFinite(asFloat));
    }

    [Fact]
    public void MinValue_IsNegative()
    {
        var min = Half.MinValue;
        var asFloat = (float)min;

        Assert.True(asFloat < 0);
        Assert.True(float.IsFinite(asFloat));
    }

    [Fact]
    public void ToString_ReturnsStringRepresentation()
    {
        var half = new Half(1.5f);
        var str = half.ToString();

        Assert.NotNull(str);
        Assert.Contains("1.5", str);
    }

    [Fact]
    public void Conversion_PreservesZero()
    {
        var half = new Half(0.0f);
        var backToFloat = (float)half;

        Assert.Equal(0.0f, backToFloat);
    }

    [Fact]
    public void Conversion_PreservesNegativeZero()
    {
        var half = new Half(-0.0f);
        var backToFloat = (float)half;

        Assert.Equal(-0.0f, backToFloat);
    }

    [Fact]
    public void VerySmallValue_RoundsToZero()
    {
        var half = new Half(1e-10f);
        var backToFloat = (float)half;

        Assert.Equal(0.0f, backToFloat);
    }
}
