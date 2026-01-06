using MLFramework.Core;
using Xunit;

namespace MLFramework.Tests.Amp.Core;

/// <summary>
/// Tests for DataType extension methods related to AMP
/// </summary>
public class DataTypeExtensionsTests
{
    [Fact]
    public void GetSize_Float16_ReturnsTwo()
    {
        Assert.Equal(2, DataType.Float16.GetSize());
    }

    [Fact]
    public void GetSize_BFloat16_ReturnsTwo()
    {
        Assert.Equal(2, DataType.BFloat16.GetSize());
    }

    [Fact]
    public void GetSize_Float32_ReturnsFour()
    {
        Assert.Equal(4, DataType.Float32.GetSize());
    }

    [Fact]
    public void GetSize_Float64_ReturnsEight()
    {
        Assert.Equal(8, DataType.Float64.GetSize());
    }

    [Fact]
    public void GetSize_Int32_ReturnsFour()
    {
        Assert.Equal(4, DataType.Int32.GetSize());
    }

    [Fact]
    public void IsFloatType_Float16_ReturnsTrue()
    {
        Assert.True(DataType.Float16.IsFloatType());
    }

    [Fact]
    public void IsFloatType_BFloat16_ReturnsTrue()
    {
        Assert.True(DataType.BFloat16.IsFloatType());
    }

    [Fact]
    public void IsFloatType_Float32_ReturnsTrue()
    {
        Assert.True(DataType.Float32.IsFloatType());
    }

    [Fact]
    public void IsFloatType_Int32_ReturnsFalse()
    {
        Assert.False(DataType.Int32.IsFloatType());
    }

    [Fact]
    public void IsFloatType_Bool_ReturnsFalse()
    {
        Assert.False(DataType.Bool.IsFloatType());
    }

    [Fact]
    public void IsLowPrecision_Float16_ReturnsTrue()
    {
        Assert.True(DataType.Float16.IsLowPrecision());
    }

    [Fact]
    public void IsLowPrecision_BFloat16_ReturnsTrue()
    {
        Assert.True(DataType.BFloat16.IsLowPrecision());
    }

    [Fact]
    public void IsLowPrecision_Float32_ReturnsFalse()
    {
        Assert.False(DataType.Float32.IsLowPrecision());
    }

    [Fact]
    public void IsLowPrecision_Int32_ReturnsFalse()
    {
        Assert.False(DataType.Int32.IsLowPrecision());
    }

    [Fact]
    public void GetHigherPrecision_Float16_ReturnsFloat32()
    {
        Assert.Equal(DataType.Float32, DataType.Float16.GetHigherPrecision());
    }

    [Fact]
    public void GetHigherPrecision_BFloat16_ReturnsFloat32()
    {
        Assert.Equal(DataType.Float32, DataType.BFloat16.GetHigherPrecision());
    }

    [Fact]
    public void GetHigherPrecision_Float32_ReturnsFloat64()
    {
        Assert.Equal(DataType.Float64, DataType.Float32.GetHigherPrecision());
    }

    [Fact]
    public void GetHigherPrecision_Float64_ReturnsFloat64()
    {
        Assert.Equal(DataType.Float64, DataType.Float64.GetHigherPrecision());
    }

    [Fact]
    public void GetHigherPrecision_Int32_ReturnsInt32()
    {
        Assert.Equal(DataType.Int32, DataType.Int32.GetHigherPrecision());
    }

    [Fact]
    public void GetLowerPrecision_Float32_ReturnsBFloat16()
    {
        Assert.Equal(DataType.BFloat16, DataType.Float32.GetLowerPrecision());
    }

    [Fact]
    public void GetLowerPrecision_Float64_ReturnsFloat32()
    {
        Assert.Equal(DataType.Float32, DataType.Float64.GetLowerPrecision());
    }

    [Fact]
    public void GetLowerPrecision_Int32_ReturnsInt32()
    {
        Assert.Equal(DataType.Int32, DataType.Int32.GetLowerPrecision());
    }

    [Fact]
    public void GetLowerPrecision_Float16_ReturnsFloat16()
    {
        Assert.Equal(DataType.Float16, DataType.Float16.GetLowerPrecision());
    }

    [Theory]
    [InlineData(DataType.Float16, true)]
    [InlineData(DataType.BFloat16, true)]
    [InlineData(DataType.Float32, true)]
    [InlineData(DataType.Float64, true)]
    [InlineData(DataType.Int32, false)]
    [InlineData(DataType.Int64, false)]
    [InlineData(DataType.Int16, false)]
    [InlineData(DataType.Int8, false)]
    [InlineData(DataType.UInt8, false)]
    [InlineData(DataType.Bool, false)]
    public void IsFloatType_AllTypes(DataType dtype, bool expected)
    {
        Assert.Equal(expected, dtype.IsFloatType());
    }

    [Theory]
    [InlineData(DataType.Float16, 2)]
    [InlineData(DataType.BFloat16, 2)]
    [InlineData(DataType.Float32, 4)]
    [InlineData(DataType.Float64, 8)]
    [InlineData(DataType.Int32, 4)]
    [InlineData(DataType.Int64, 8)]
    [InlineData(DataType.Int16, 2)]
    [InlineData(DataType.Int8, 1)]
    [InlineData(DataType.UInt8, 1)]
    [InlineData(DataType.Bool, 1)]
    public void GetSize_AllTypes(DataType dtype, int expectedSize)
    {
        Assert.Equal(expectedSize, dtype.GetSize());
    }
}
