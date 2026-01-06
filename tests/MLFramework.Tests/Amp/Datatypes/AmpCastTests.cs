using MLFramework.Amp;
using Xunit;

namespace MLFramework.Tests.Amp.Datatypes;

/// <summary>
/// Tests for AmpCast high-performance casting utilities
/// </summary>
public class AmpCastTests
{
    [Fact]
    public void CastToHalf_WithValidInput_ReturnsHalfArray()
    {
        var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var result = AmpCast.CastToHalf(input);

        Assert.NotNull(result);
        Assert.Equal(input.Length, result.Length);
    }

    [Fact]
    public void CastToHalf_WithNullInput_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => AmpCast.CastToHalf(null!));
    }

    [Fact]
    public void CastToHalf_PreservesValues()
    {
        var input = new float[] { 1.5f, 2.75f, 3.14f };
        var result = AmpCast.CastToHalf(input);

        for (int i = 0; i < input.Length; i++)
        {
            var backToFloat = (float)result[i];
            Assert.InRange(Math.Abs(backToFloat - input[i]), 0, 0.01);
        }
    }

    [Fact]
    public void CastToHalf_WithEmptyArray_ReturnsEmptyArray()
    {
        var input = Array.Empty<float>();
        var result = AmpCast.CastToHalf(input);

        Assert.NotNull(result);
        Assert.Empty(result);
    }

    [Fact]
    public void CastToHalf_WithSpecialValues_HandlesCorrectly()
    {
        var input = new float[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN, 0.0f };
        var result = AmpCast.CastToHalf(input);

        Assert.True(result[0].IsPositiveInfinity);
        Assert.True(result[1].IsNegativeInfinity);
        Assert.True(result[2].IsNaN);
        Assert.Equal(0.0f, (float)result[3]);
    }

    [Fact]
    public void CastToBFloat16_WithValidInput_ReturnsBFloat16Array()
    {
        var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var result = AmpCast.CastToBFloat16(input);

        Assert.NotNull(result);
        Assert.Equal(input.Length, result.Length);
    }

    [Fact]
    public void CastToBFloat16_WithNullInput_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => AmpCast.CastToBFloat16(null!));
    }

    [Fact]
    public void CastToBFloat16_PreservesValues()
    {
        var input = new float[] { 1.5f, 2.75f, 3.14f };
        var result = AmpCast.CastToBFloat16(input);

        for (int i = 0; i < input.Length; i++)
        {
            var backToFloat = (float)result[i];
            Assert.InRange(Math.Abs(backToFloat - input[i]), 0, 0.1);
        }
    }

    [Fact]
    public void CastToBFloat16_WithEmptyArray_ReturnsEmptyArray()
    {
        var input = Array.Empty<float>();
        var result = AmpCast.CastToBFloat16(input);

        Assert.NotNull(result);
        Assert.Empty(result);
    }

    [Fact]
    public void CastToBFloat16_WithSpecialValues_HandlesCorrectly()
    {
        var input = new float[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN, 0.0f };
        var result = AmpCast.CastToBFloat16(input);

        Assert.True(result[0].IsPositiveInfinity);
        Assert.True(result[1].IsNegativeInfinity);
        Assert.True(result[2].IsNaN);
        Assert.Equal(0.0f, (float)result[3]);
    }

    [Fact]
    public void CastToFloat_FromHalf_WithValidInput_ReturnsFloatArray()
    {
        var input = new Half[] { new Half(1.0f), new Half(2.0f), new Half(3.0f) };
        var result = AmpCast.CastToFloat(input);

        Assert.NotNull(result);
        Assert.Equal(input.Length, result.Length);
    }

    [Fact]
    public void CastToFloat_FromHalf_WithNullInput_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => AmpCast.CastToFloat(null!));
    }

    [Fact]
    public void CastToFloat_FromHalf_PreservesValues()
    {
        var input = new Half[] { new Half(1.5f), new Half(2.75f), new Half(3.14f) };
        var result = AmpCast.CastToFloat(input);

        for (int i = 0; i < input.Length; i++)
        {
            var original = (float)input[i];
            Assert.InRange(Math.Abs(result[i] - original), 0, 0.001);
        }
    }

    [Fact]
    public void CastToFloat_FromBFloat16_WithValidInput_ReturnsFloatArray()
    {
        var input = new BFloat16[] { new BFloat16(1.0f), new BFloat16(2.0f), new BFloat16(3.0f) };
        var result = AmpCast.CastToFloat(input);

        Assert.NotNull(result);
        Assert.Equal(input.Length, result.Length);
    }

    [Fact]
    public void CastToFloat_FromBFloat16_WithNullInput_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => AmpCast.CastToFloat(null!));
    }

    [Fact]
    public void CastToFloat_FromBFloat16_PreservesValues()
    {
        var input = new BFloat16[] { new BFloat16(1.5f), new BFloat16(2.75f), new BFloat16(3.14f) };
        var result = AmpCast.CastToFloat(input);

        for (int i = 0; i < input.Length; i++)
        {
            var original = (float)input[i];
            Assert.InRange(Math.Abs(result[i] - original), 0, 0.001);
        }
    }

    [Fact]
    public void CastInPlace_FromHalf_WithValidInput_WritesToOutput()
    {
        var input = new Half[] { new Half(1.5f), new Half(2.75f), new Half(3.14f) };
        var output = new float[input.Length];

        AmpCast.CastInPlace(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            var expected = (float)input[i];
            Assert.InRange(Math.Abs(output[i] - expected), 0, 0.001);
        }
    }

    [Fact]
    public void CastInPlace_FromHalf_WithNullInput_ThrowsArgumentNullException()
    {
        var output = new float[3];

        Assert.Throws<ArgumentNullException>(() => AmpCast.CastInPlace(null!, output));
    }

    [Fact]
    public void CastInPlace_FromHalf_WithNullOutput_ThrowsArgumentNullException()
    {
        var input = new Half[] { new Half(1.5f), new Half(2.75f), new Half(3.14f) };

        Assert.Throws<ArgumentNullException>(() => AmpCast.CastInPlace(input, null!));
    }

    [Fact]
    public void CastInPlace_FromHalf_WithDifferentLengths_ThrowsArgumentException()
    {
        var input = new Half[] { new Half(1.5f), new Half(2.75f), new Half(3.14f) };
        var output = new float[2];

        Assert.Throws<ArgumentException>(() => AmpCast.CastInPlace(input, output));
    }

    [Fact]
    public void CastInPlace_FromBFloat16_WithValidInput_WritesToOutput()
    {
        var input = new BFloat16[] { new BFloat16(1.5f), new BFloat16(2.75f), new BFloat16(3.14f) };
        var output = new float[input.Length];

        AmpCast.CastInPlace(input, output);

        for (int i = 0; i < input.Length; i++)
        {
            var expected = (float)input[i];
            Assert.InRange(Math.Abs(output[i] - expected), 0, 0.001);
        }
    }

    [Fact]
    public void CastInPlace_FromBFloat16_WithNullInput_ThrowsArgumentNullException()
    {
        var output = new float[3];

        Assert.Throws<ArgumentNullException>(() => AmpCast.CastInPlace(null!, output));
    }

    [Fact]
    public void CastInPlace_FromBFloat16_WithNullOutput_ThrowsArgumentNullException()
    {
        var input = new BFloat16[] { new BFloat16(1.5f), new BFloat16(2.75f), new BFloat16(3.14f) };

        Assert.Throws<ArgumentNullException>(() => AmpCast.CastInPlace(input, null!));
    }

    [Fact]
    public void CastInPlace_FromBFloat16_WithDifferentLengths_ThrowsArgumentException()
    {
        var input = new BFloat16[] { new BFloat16(1.5f), new BFloat16(2.75f), new BFloat16(3.14f) };
        var output = new float[2];

        Assert.Throws<ArgumentException>(() => AmpCast.CastInPlace(input, output));
    }

    [Fact]
    public void CastToFloat_WithLargeArray_WorksCorrectly()
    {
        var input = new float[1000];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = i * 0.001f;
        }

        var halfResult = AmpCast.CastToHalf(input);
        var floatResult = AmpCast.CastToFloat(halfResult);

        Assert.Equal(input.Length, floatResult.Length);
    }

    [Fact]
    public void CastToBFloat16_WithLargeArray_WorksCorrectly()
    {
        var input = new float[1000];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = i * 0.001f;
        }

        var bfloatResult = AmpCast.CastToBFloat16(input);

        Assert.Equal(input.Length, bfloatResult.Length);
    }

    [Fact]
    public void RoundTripConversion_Half_PreservesValues()
    {
        var original = new float[] { 1.0f, 2.5f, 3.75f, 4.125f };
        var half = AmpCast.CastToHalf(original);
        var backToFloat = AmpCast.CastToFloat(half);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.InRange(Math.Abs(backToFloat[i] - original[i]), 0, 0.01);
        }
    }

    [Fact]
    public void RoundTripConversion_BFloat16_PreservesValues()
    {
        var original = new float[] { 1.0f, 2.5f, 3.75f, 4.125f };
        var bfloat = AmpCast.CastToBFloat16(original);
        var backToFloat = AmpCast.CastToFloat(bfloat);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.InRange(Math.Abs(backToFloat[i] - original[i]), 0, 0.1);
        }
    }
}
