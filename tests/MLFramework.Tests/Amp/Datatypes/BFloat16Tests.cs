using MLFramework.Amp;
using Xunit;

namespace MLFramework.Tests.Amp.Datatypes;

/// <summary>
/// Tests for BFloat16 (16-bit Brain Floating Point) type
/// </summary>
public class BFloat16Tests
{
    [Fact]
    public void Constructor_FromFloat_PreservesValueWithinRange()
    {
        var bfloat = new BFloat16(1.5f);
        var backToFloat = (float)bfloat;

        Assert.InRange(Math.Abs(backToFloat - 1.5f), 0, 0.01f);
    }

    [Fact]
    public void Constructor_FromDouble_PreservesValueWithinRange()
    {
        var bfloat = new BFloat16(2.75);
        var backToFloat = (float)bfloat;

        Assert.InRange(Math.Abs(backToFloat - 2.75), 0, 0.01);
    }

    [Fact]
    public void Constructor_FromRawBits_CreatesCorrectValue()
    {
        var bfloat = new BFloat16((ushort)0x3F80); // Represents 1.0
        var backToFloat = (float)bfloat;

        Assert.Equal(1.0f, backToFloat);
    }

    [Fact]
    public void ConversionToFloat_WorksCorrectly()
    {
        var bfloat = new BFloat16(3.14f);
        float result = bfloat;

        // BFloat16 has lower precision (about 2 decimal digits)
        Assert.InRange(Math.Abs(result - 3.14f), 0, 0.1);
    }

    [Fact]
    public void ConversionToDouble_WorksCorrectly()
    {
        var bfloat = new BFloat16(2.718f);
        double result = bfloat;

        Assert.InRange(Math.Abs(result - 2.718), 0, 0.1);
    }

    [Fact]
    public void AdditionOperator_WorksCorrectly()
    {
        var bf1 = new BFloat16(2.5f);
        var bf2 = new BFloat16(1.5f);
        var result = bf1 + bf2;

        Assert.InRange(Math.Abs((float)result - 4.0f), 0, 0.1);
    }

    [Fact]
    public void SubtractionOperator_WorksCorrectly()
    {
        var bf1 = new BFloat16(5.0f);
        var bf2 = new BFloat16(2.0f);
        var result = bf1 - bf2;

        Assert.InRange(Math.Abs((float)result - 3.0f), 0, 0.1);
    }

    [Fact]
    public void MultiplicationOperator_WorksCorrectly()
    {
        var bf1 = new BFloat16(3.0f);
        var bf2 = new BFloat16(4.0f);
        var result = bf1 * bf2;

        Assert.InRange(Math.Abs((float)result - 12.0f), 0, 0.1);
    }

    [Fact]
    public void DivisionOperator_WorksCorrectly()
    {
        var bf1 = new BFloat16(10.0f);
        var bf2 = new BFloat16(2.0f);
        var result = bf1 / bf2;

        Assert.InRange(Math.Abs((float)result - 5.0f), 0, 0.1);
    }

    [Fact]
    public void EqualityOperator_WorksForEqualValues()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(1.5f);

        Assert.True(bf1 == bf2);
    }

    [Fact]
    public void EqualityOperator_WorksForDifferentValues()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(2.5f);

        Assert.False(bf1 == bf2);
    }

    [Fact]
    public void InequalityOperator_WorksCorrectly()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(2.5f);

        Assert.True(bf1 != bf2);
    }

    [Fact]
    public void LessThanOperator_WorksCorrectly()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(2.5f);

        Assert.True(bf1 < bf2);
        Assert.False(bf2 < bf1);
    }

    [Fact]
    public void GreaterThanOperator_WorksCorrectly()
    {
        var bf1 = new BFloat16(2.5f);
        var bf2 = new BFloat16(1.5f);

        Assert.True(bf1 > bf2);
        Assert.False(bf2 > bf1);
    }

    [Fact]
    public void LessThanOrEqualOperator_WorksCorrectly()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(1.5f);
        var bf3 = new BFloat16(2.5f);

        Assert.True(bf1 <= bf2);
        Assert.True(bf1 <= bf3);
    }

    [Fact]
    public void GreaterThanOrEqualOperator_WorksCorrectly()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(1.5f);
        var bf3 = new BFloat16(0.5f);

        Assert.True(bf1 >= bf2);
        Assert.True(bf1 >= bf3);
    }

    [Fact]
    public void CompareTo_WorksCorrectly()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(2.5f);
        var bf3 = new BFloat16(1.5f);

        Assert.True(bf1.CompareTo(bf2) < 0);
        Assert.True(bf2.CompareTo(bf1) > 0);
        Assert.True(bf1.CompareTo(bf3) == 0);
    }

    [Fact]
    public void Equals_ReturnsTrueForEqualValues()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(1.5f);

        Assert.True(bf1.Equals(bf2));
    }

    [Fact]
    public void Equals_ReturnsFalseForDifferentValues()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(2.5f);

        Assert.False(bf1.Equals(bf2));
    }

    [Fact]
    public void Equals_ObjectOverride_WorksCorrectly()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(1.5f);

        Assert.True(bf1.Equals((object)bf2));
    }

    [Fact]
    public void GetHashCode_ReturnsSameForEqualValues()
    {
        var bf1 = new BFloat16(1.5f);
        var bf2 = new BFloat16(1.5f);

        Assert.Equal(bf1.GetHashCode(), bf2.GetHashCode());
    }

    [Fact]
    public void IsNaN_ReturnsTrueForNaN()
    {
        var bfloat = BFloat16.NaN;

        Assert.True(bfloat.IsNaN);
    }

    [Fact]
    public void IsNaN_ReturnsFalseForNormalValue()
    {
        var bfloat = new BFloat16(1.5f);

        Assert.False(bfloat.IsNaN);
    }

    [Fact]
    public void IsInfinity_ReturnsTrueForInfinity()
    {
        var posInf = BFloat16.PositiveInfinity;
        var negInf = BFloat16.NegativeInfinity;

        Assert.True(posInf.IsInfinity);
        Assert.True(negInf.IsInfinity);
    }

    [Fact]
    public void IsInfinity_ReturnsFalseForNormalValue()
    {
        var bfloat = new BFloat16(1.5f);

        Assert.False(bfloat.IsInfinity);
    }

    [Fact]
    public void IsPositiveInfinity_ReturnsTrueForPositiveInfinity()
    {
        var bfloat = BFloat16.PositiveInfinity;

        Assert.True(bfloat.IsPositiveInfinity);
    }

    [Fact]
    public void IsPositiveInfinity_ReturnsFalseForNegativeInfinity()
    {
        var bfloat = BFloat16.NegativeInfinity;

        Assert.False(bfloat.IsPositiveInfinity);
    }

    [Fact]
    public void IsNegativeInfinity_ReturnsTrueForNegativeInfinity()
    {
        var bfloat = BFloat16.NegativeInfinity;

        Assert.True(bfloat.IsNegativeInfinity);
    }

    [Fact]
    public void IsNegativeInfinity_ReturnsFalseForPositiveInfinity()
    {
        var bfloat = BFloat16.PositiveInfinity;

        Assert.False(bfloat.IsNegativeInfinity);
    }

    [Fact]
    public void Epsilon_IsDefined()
    {
        var epsilon = BFloat16.Epsilon;

        Assert.True(epsilon > new BFloat16(0));
    }

    [Fact]
    public void MaxValue_IsPositive()
    {
        var max = BFloat16.MaxValue;
        var asFloat = (float)max;

        Assert.True(asFloat > 0);
        Assert.True(float.IsFinite(asFloat));
    }

    [Fact]
    public void MinValue_IsNegative()
    {
        var min = BFloat16.MinValue;
        var asFloat = (float)min;

        Assert.True(asFloat < 0);
        Assert.True(float.IsFinite(asFloat));
    }

    [Fact]
    public void ToString_ReturnsStringRepresentation()
    {
        var bfloat = new BFloat16(1.5f);
        var str = bfloat.ToString();

        Assert.NotNull(str);
    }

    [Fact]
    public void Conversion_PreservesZero()
    {
        var bfloat = new BFloat16(0.0f);
        var backToFloat = (float)bfloat;

        Assert.Equal(0.0f, backToFloat);
    }

    [Fact]
    public void Conversion_PreservesNegativeZero()
    {
        var bfloat = new BFloat16(-0.0f);
        var backToFloat = (float)bfloat;

        Assert.Equal(-0.0f, backToFloat);
    }

    [Fact]
    public void VerySmallValue_PreservesExponent()
    {
        // BFloat16 preserves exponent bits unlike Half
        var bfloat = new BFloat16(1e-38f);
        var backToFloat = (float)bfloat;

        // Should still be non-zero due to preserved exponent
        Assert.NotEqual(0.0f, backToFloat);
    }

    [Fact]
    public void VeryLargeValue_PreservesRange()
    {
        // BFloat16 has same exponent range as FP32
        var bfloat = new BFloat16(1e38f);
        var backToFloat = (float)bfloat;

        // Should not be infinity
        Assert.True(float.IsFinite(backToFloat));
    }

    [Fact]
    public void MantissaTruncation_CausesPrecisionLoss()
    {
        var bfloat = new BFloat16(1.2345678f);
        var backToFloat = (float)bfloat;

        // Should not be exactly the same due to mantissa truncation
        Assert.NotEqual(1.2345678f, backToFloat);
        // But should be close
        Assert.InRange(Math.Abs(backToFloat - 1.2345678f), 0, 0.01);
    }
}
