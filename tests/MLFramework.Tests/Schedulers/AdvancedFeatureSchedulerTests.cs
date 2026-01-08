using Xunit;
using MLFramework.Schedulers;

namespace MLFramework.Tests.Schedulers;

public class AdvancedFeatureSchedulerTests
{
    #region PolynomialDecayScheduler Tests

    [Fact]
    public void PolynomialDecayScheduler_AtZero_ReturnsInitialLR()
    {
        var scheduler = new PolynomialDecayScheduler(
            initialLearningRate: 0.01f,
            finalLearningRate: 0.0001f,
            totalSteps: 1000f
        );

        float lr = scheduler.GetLearningRate(0, 0.1f);

        Assert.Equal(0.01f, lr);
    }

    [Fact]
    public void PolynomialDecayScheduler_AtTotalSteps_ReturnsFinalLR()
    {
        var scheduler = new PolynomialDecayScheduler(
            initialLearningRate: 0.01f,
            finalLearningRate: 0.0001f,
            totalSteps: 1000f
        );

        float lr = scheduler.GetLearningRate(1000, 0.1f);

        Assert.Equal(0.0001f, lr);
    }

    [Fact]
    public void PolynomialDecayScheduler_DifferentPower_WorksCorrectly()
    {
        var linear = new PolynomialDecayScheduler(0.1f, 0f, 100f, 1.0f);
        var quadratic = new PolynomialDecayScheduler(0.1f, 0f, 100f, 2.0f);

        float lrLinear = linear.GetLearningRate(50, 0.1f);
        float lrQuadratic = quadratic.GetLearningRate(50, 0.1f);

        // Quadratic decays faster
        Assert.True(lrQuadratic < lrLinear);
    }

    #endregion

    #region LayerWiseLRDecayScheduler Tests

    [Fact]
    public void LayerWiseLRDecayScheduler_LastLayer_HasMultiplierOne()
    {
        var scheduler = new LayerWiseLRDecayScheduler(decayFactor: 0.8f);

        float multiplier = scheduler.GetLayerMultiplier(
            layerIndex: 3,
            totalLayers: 4
        );

        Assert.Equal(1.0f, multiplier);
    }

    [Fact]
    public void LayerWiseLRDecayScheduler_EarlierLayers_HaveLowerMultipliers()
    {
        var scheduler = new LayerWiseLRDecayScheduler(decayFactor: 0.8f);

        float m1 = scheduler.GetLayerMultiplier(0, 4);
        float m2 = scheduler.GetLayerMultiplier(1, 4);
        float m3 = scheduler.GetLayerMultiplier(2, 4);
        float m4 = scheduler.GetLayerMultiplier(3, 4);

        Assert.Equal(0.512f, m1);  // 0.8^3
        Assert.Equal(0.64f, m2);   // 0.8^2
        Assert.Equal(0.8f, m3);    // 0.8^1
        Assert.Equal(1.0f, m4);    // 0.8^0
    }

    [Fact]
    public void LayerWiseLRDecayScheduler_ExcludedLayer_HasMultiplierOne()
    {
        var scheduler = new LayerWiseLRDecayScheduler(
            decayFactor: 0.8f,
            excludedLayers: new[] { "embedding" }
        );

        float multiplier = scheduler.GetLayerMultiplier(
            layerIndex: 0,
            totalLayers: 4,
            layerName: "embedding"
        );

        Assert.Equal(1.0f, multiplier);
    }

    #endregion

    #region DiscriminativeLRScheduler Tests

    [Fact]
    public void DiscriminativeLRScheduler_ByIndex_ReturnsCorrectLR()
    {
        var scheduler = new DiscriminativeLRScheduler(
            baseLearningRate: 0.01f,
            layerMultipliers: new[] { 0.1f, 0.2f, 0.5f, 1.0f }
        );

        Assert.Equal(0.001f, scheduler.GetGroupLearningRate(0));   // 0.01 * 0.1
        Assert.Equal(0.002f, scheduler.GetGroupLearningRate(1));   // 0.01 * 0.2
        Assert.Equal(0.005f, scheduler.GetGroupLearningRate(2));   // 0.01 * 0.5
        Assert.Equal(0.01f, scheduler.GetGroupLearningRate(3));    // 0.01 * 1.0
    }

    [Fact]
    public void DiscriminativeLRScheduler_ByName_ReturnsCorrectLR()
    {
        var scheduler = new DiscriminativeLRScheduler(
            baseLearningRate: 0.01f,
            layerMultipliers: new[] { 0.1f, 1.0f },
            layerNames: new[] { "encoder", "decoder" }
        );

        Assert.Equal(0.001f, scheduler.GetGroupLearningRate(0, "encoder"));
        Assert.Equal(0.01f, scheduler.GetGroupLearningRate(1, "decoder"));
    }

    #endregion
}
