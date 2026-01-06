using MLFramework.Amp;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for DynamicScalerConfig class
/// </summary>
public class DynamicScalerConfigTests
{
    #region Property Tests

    [Fact]
    public void Properties_AllSettableAndGettable()
    {
        var config = new DynamicScalerConfig
        {
            InitialScale = 1000.0f,
            GrowthFactor = 1.5f,
            BackoffFactor = 0.75f,
            GrowthInterval = 5000,
            MinScale = 0.5f,
            MaxScale = 1000000.0f
        };

        Assert.Equal(1000.0f, config.InitialScale);
        Assert.Equal(1.5f, config.GrowthFactor);
        Assert.Equal(0.75f, config.BackoffFactor);
        Assert.Equal(5000, config.GrowthInterval);
        Assert.Equal(0.5f, config.MinScale);
        Assert.Equal(1000000.0f, config.MaxScale);
    }

    #endregion

    #region CreateDefault Tests

    [Fact]
    public void CreateDefault_ReturnsConfigWithDefaultValues()
    {
        var config = DynamicScalerConfig.CreateDefault();

        Assert.NotNull(config);
        Assert.Equal(65536.0f, config.InitialScale);
        Assert.Equal(2.0f, config.GrowthFactor);
        Assert.Equal(0.5f, config.BackoffFactor);
        Assert.Equal(2000, config.GrowthInterval);
        Assert.Equal(1.0f, config.MinScale);
        Assert.Equal(16777216.0f, config.MaxScale);
    }

    #endregion

    #region CreateConservative Tests

    [Fact]
    public void CreateConservative_ReturnsConfigWithConservativeValues()
    {
        var config = DynamicScalerConfig.CreateConservative();

        Assert.NotNull(config);
        Assert.Equal(32768.0f, config.InitialScale);
        Assert.Equal(2.0f, config.GrowthFactor);
        Assert.Equal(0.5f, config.BackoffFactor);
        Assert.Equal(5000, config.GrowthInterval);
        Assert.Equal(1.0f, config.MinScale);
        Assert.Equal(16777216.0f, config.MaxScale);
    }

    #endregion

    #region CreateAggressive Tests

    [Fact]
    public void CreateAggressive_ReturnsConfigWithAggressiveValues()
    {
        var config = DynamicScalerConfig.CreateAggressive();

        Assert.NotNull(config);
        Assert.Equal(65536.0f, config.InitialScale);
        Assert.Equal(2.0f, config.GrowthFactor);
        Assert.Equal(0.5f, config.BackoffFactor);
        Assert.Equal(1000, config.GrowthInterval);
        Assert.Equal(1.0f, config.MinScale);
        Assert.Equal(16777216.0f, config.MaxScale);
    }

    #endregion
}
