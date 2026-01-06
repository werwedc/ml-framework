using MLFramework.Amp;
using MLFramework.Core;
using Xunit;

namespace MLFramework.Tests.Amp.Core;

/// <summary>
/// Tests for AmpConfig class
/// </summary>
public class AmpConfigTests
{
    [Fact]
    public void CreateDefault_UsesBFloat16()
    {
        var config = AmpConfig.CreateDefault();

        Assert.True(config.Enabled);
        Assert.Equal(DataType.BFloat16, config.TargetPrecision);
        Assert.Equal(DataType.Float32, config.HigherPrecision);
        Assert.True(config.UseViewCasting);
        Assert.True(config.EnableKernelFusion);
    }

    [Fact]
    public void CreateFp16_UsesFloat16()
    {
        var config = AmpConfig.CreateFp16();

        Assert.True(config.Enabled);
        Assert.Equal(DataType.Float16, config.TargetPrecision);
        Assert.Equal(DataType.Float32, config.HigherPrecision);
        Assert.True(config.UseViewCasting);
        Assert.True(config.EnableKernelFusion);
    }

    [Fact]
    public void CreateBf16_UsesBFloat16()
    {
        var config = AmpConfig.CreateBf16();

        Assert.True(config.Enabled);
        Assert.Equal(DataType.BFloat16, config.TargetPrecision);
        Assert.Equal(DataType.Float32, config.HigherPrecision);
        Assert.True(config.UseViewCasting);
        Assert.True(config.EnableKernelFusion);
    }

    [Fact]
    public void Constructor_AllowsCustomConfiguration()
    {
        var config = new AmpConfig
        {
            TargetPrecision = DataType.Float16,
            Enabled = false,
            HigherPrecision = DataType.Float64,
            UseViewCasting = false,
            EnableKernelFusion = false
        };

        Assert.False(config.Enabled);
        Assert.Equal(DataType.Float16, config.TargetPrecision);
        Assert.Equal(DataType.Float64, config.HigherPrecision);
        Assert.False(config.UseViewCasting);
        Assert.False(config.EnableKernelFusion);
    }

    [Fact]
    public void Enabled_DefaultCanBeChanged()
    {
        var config = AmpConfig.CreateBf16();
        Assert.True(config.Enabled);

        config.Enabled = false;
        Assert.False(config.Enabled);
    }

    [Fact]
    public void TargetPrecision_DefaultCanBeChanged()
    {
        var config = AmpConfig.CreateBf16();
        Assert.Equal(DataType.BFloat16, config.TargetPrecision);

        config.TargetPrecision = DataType.Float16;
        Assert.Equal(DataType.Float16, config.TargetPrecision);
    }

    [Fact]
    public void HigherPrecision_DefaultCanBeChanged()
    {
        var config = AmpConfig.CreateBf16();
        Assert.Equal(DataType.Float32, config.HigherPrecision);

        config.HigherPrecision = DataType.Float64;
        Assert.Equal(DataType.Float64, config.HigherPrecision);
    }

    [Fact]
    public void UseViewCasting_DefaultCanBeChanged()
    {
        var config = AmpConfig.CreateBf16();
        Assert.True(config.UseViewCasting);

        config.UseViewCasting = false;
        Assert.False(config.UseViewCasting);
    }

    [Fact]
    public void EnableKernelFusion_DefaultCanBeChanged()
    {
        var config = AmpConfig.CreateBf16();
        Assert.True(config.EnableKernelFusion);

        config.EnableKernelFusion = false;
        Assert.False(config.EnableKernelFusion);
    }

    [Fact]
    public void MultipleConfigs_DontInterfere()
    {
        var config1 = AmpConfig.CreateBf16();
        var config2 = AmpConfig.CreateFp16();

        Assert.Equal(DataType.BFloat16, config1.TargetPrecision);
        Assert.Equal(DataType.Float16, config2.TargetPrecision);

        config1.TargetPrecision = DataType.Float32;

        Assert.Equal(DataType.Float32, config1.TargetPrecision);
        Assert.Equal(DataType.Float16, config2.TargetPrecision);
    }

    [Theory]
    [InlineData(DataType.Float16)]
    [InlineData(DataType.BFloat16)]
    [InlineData(DataType.Float32)]
    public void TargetPrecision_AcceptsLowPrecisionAndFloat32(DataType dtype)
    {
        var config = new AmpConfig
        {
            TargetPrecision = dtype
        };

        Assert.Equal(dtype, config.TargetPrecision);
    }
}
