using MLFramework.Amp;
using Xunit;

namespace MLFramework.Tests.Amp.Scalers;

/// <summary>
/// Tests for GradScalerFactory class
/// </summary>
public class GradScalerFactoryTests
{
    #region CreateDefault Tests

    [Fact]
    public void CreateDefault_ReturnsGradScalerWithDefaultSettings()
    {
        var scaler = GradScalerFactory.CreateDefault();

        Assert.NotNull(scaler);
        Assert.Equal(65536.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    #endregion

    #region CreateStatic Tests

    [Fact]
    public void CreateStatic_ReturnsGradScalerWithDefaultScale()
    {
        var scaler = GradScalerFactory.CreateStatic();

        Assert.NotNull(scaler);
        Assert.Equal(65536.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    [Fact]
    public void CreateStatic_WithCustomScale_ReturnsGradScalerWithCustomScale()
    {
        var scaler = GradScalerFactory.CreateStatic(256.0f);

        Assert.NotNull(scaler);
        Assert.Equal(256.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    #endregion

    #region CreateConservative Tests

    [Fact]
    public void CreateConservative_ReturnsGradScalerWithConservativeSettings()
    {
        var scaler = GradScalerFactory.CreateConservative();

        Assert.NotNull(scaler);
        Assert.Equal(32768.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    #endregion

    #region CreateAggressive Tests

    [Fact]
    public void CreateAggressive_ReturnsGradScalerWithAggressiveSettings()
    {
        var scaler = GradScalerFactory.CreateAggressive();

        Assert.NotNull(scaler);
        Assert.Equal(65536.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    #endregion

    #region CreateForFP16 Tests

    [Fact]
    public void CreateForFP16_ReturnsGradScalerForFP16()
    {
        var scaler = GradScalerFactory.CreateForFP16();

        Assert.NotNull(scaler);
        Assert.Equal(65536.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    #endregion

    #region CreateForBF16 Tests

    [Fact]
    public void CreateForBF16_ReturnsGradScalerForBF16()
    {
        var scaler = GradScalerFactory.CreateForBF16();

        Assert.NotNull(scaler);
        Assert.Equal(8.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    #endregion

    #region CreateFromConfig Tests

    [Fact]
    public void CreateFromConfig_WithValidConfig_ReturnsGradScaler()
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

        var scaler = GradScalerFactory.CreateFromConfig(config);

        Assert.NotNull(scaler);
        Assert.Equal(1000.0f, scaler.Scale);
        Assert.True(scaler.Enabled);
    }

    [Fact]
    public void CreateFromConfig_WithNullConfig_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            GradScalerFactory.CreateFromConfig(null!));
    }

    #endregion
}
