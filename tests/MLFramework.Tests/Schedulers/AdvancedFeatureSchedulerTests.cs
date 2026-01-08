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

    [Fact]
    public void PolynomialDecayScheduler_AtHalfSteps_ReturnsIntermediateValue()
    {
        var scheduler = new PolynomialDecayScheduler(
            initialLearningRate: 0.01f,
            finalLearningRate: 0.001f,
            totalSteps: 1000f,
            power: 1.0f
        );

        float lr = scheduler.GetLearningRate(500, 0.1f);

        // Linear decay: should be halfway between initial and final
        Assert.Equal(0.0055f, lr);
    }

    [Fact]
    public void PolynomialDecayScheduler_BeyondTotalSteps_ReturnsFinalLR()
    {
        var scheduler = new PolynomialDecayScheduler(
            initialLearningRate: 0.01f,
            finalLearningRate: 0.0001f,
            totalSteps: 1000f
        );

        float lr1 = scheduler.GetLearningRate(1000, 0.1f);
        float lr2 = scheduler.GetLearningRate(2000, 0.1f);

        // Both should return final LR
        Assert.Equal(0.0001f, lr1);
        Assert.Equal(0.0001f, lr2);
    }

    [Fact]
    public void PolynomialDecayScheduler_StateSerialization_Works()
    {
        var scheduler = new PolynomialDecayScheduler(
            initialLearningRate: 0.01f,
            finalLearningRate: 0.001f,
            totalSteps: 1000f,
            power: 2.0f
        );

        scheduler.Step();
        scheduler.Step();
        scheduler.StepEpoch();

        var state = scheduler.GetState();

        Assert.Equal(0.01f, state.Get<float>("initial_lr"));
        Assert.Equal(0.001f, state.Get<float>("final_lr"));
        Assert.Equal(1000f, state.Get<float>("total_steps"));
        Assert.Equal(2.0f, state.Get<float>("power"));
        Assert.Equal(3, state.Get<int>("step_count"));
        Assert.Equal(1, state.Get<int>("epoch_count"));
    }

    [Fact]
    public void PolynomialDecayScheduler_StateDeserialization_Works()
    {
        var scheduler1 = new PolynomialDecayScheduler(
            initialLearningRate: 0.01f,
            finalLearningRate: 0.001f,
            totalSteps: 1000f,
            power: 2.0f
        );

        scheduler1.Step();
        scheduler1.Step();
        scheduler1.StepEpoch();

        var state = scheduler1.GetState();

        var scheduler2 = new PolynomialDecayScheduler(
            initialLearningRate: 0.02f, // Different initial
            finalLearningRate: 0.002f, // Different final
            totalSteps: 2000f,
            power: 1.0f
        );

        scheduler2.LoadState(state);

        Assert.Equal(3, scheduler2.StepCount);
        Assert.Equal(1, scheduler2.EpochCount);
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

    [Fact]
    public void LayerWiseLRDecayScheduler_SingleLayer_HasMultiplierOne()
    {
        var scheduler = new LayerWiseLRDecayScheduler(decayFactor: 0.8f);

        float multiplier = scheduler.GetLayerMultiplier(
            layerIndex: 0,
            totalLayers: 1
        );

        Assert.Equal(1.0f, multiplier);
    }

    [Fact]
    public void LayerWiseLRDecayScheduler_DifferentDecayFactors_WorkCorrectly()
    {
        var scheduler1 = new LayerWiseLRDecayScheduler(decayFactor: 0.9f);
        var scheduler2 = new LayerWiseLRDecayScheduler(decayFactor: 0.5f);

        float m1 = scheduler1.GetLayerMultiplier(0, 3);
        float m2 = scheduler2.GetLayerMultiplier(0, 3);

        // Lower decay factor means faster decay
        Assert.True(m2 < m1);
    }

    [Fact]
    public void LayerWiseLRDecayScheduler_StateSerialization_Works()
    {
        var scheduler = new LayerWiseLRDecayScheduler(
            decayFactor: 0.8f,
            excludedLayers: new[] { "embedding", "layer_norm" }
        );

        scheduler.Step();
        scheduler.StepEpoch();

        var state = scheduler.GetState();

        Assert.Equal(0.8f, state.Get<float>("decay_factor"));
        Assert.Equal(new[] { "embedding", "layer_norm" }, state.Get<string[]>("excluded_layers"));
        Assert.Equal(1, state.Get<int>("step_count"));
        Assert.Equal(1, state.Get<int>("epoch_count"));
    }

    [Fact]
    public void LayerWiseLRDecayScheduler_StateDeserialization_Works()
    {
        var scheduler1 = new LayerWiseLRDecayScheduler(
            decayFactor: 0.8f,
            excludedLayers: new[] { "embedding" }
        );

        scheduler1.Step();
        scheduler1.StepEpoch();

        var state = scheduler1.GetState();

        var scheduler2 = new LayerWiseLRDecayScheduler(
            decayFactor: 0.9f,
            excludedLayers: new[] { "layer_norm" }
        );

        scheduler2.LoadState(state);

        Assert.Equal(1, scheduler2.StepCount);
        Assert.Equal(1, scheduler2.EpochCount);
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

    [Fact]
    public void DiscriminativeLRScheduler_OutOfRangeIndex_ReturnsLastMultiplier()
    {
        var scheduler = new DiscriminativeLRScheduler(
            baseLearningRate: 0.01f,
            layerMultipliers: new[] { 0.1f, 0.2f, 0.5f, 1.0f }
        );

        float lr = scheduler.GetGroupLearningRate(10);

        // Should return last multiplier (1.0 * baseLR)
        Assert.Equal(0.01f, lr);
    }

    [Fact]
    public void DiscriminativeLRScheduler_StateSerialization_Works()
    {
        var scheduler = new DiscriminativeLRScheduler(
            baseLearningRate: 0.01f,
            layerMultipliers: new[] { 0.1f, 0.2f, 0.5f, 1.0f },
            layerNames: new[] { "encoder.1", "encoder.2", "decoder.1", "decoder.2" }
        );

        scheduler.Step();
        scheduler.StepEpoch();

        var state = scheduler.GetState();

        Assert.Equal(0.01f, state.Get<float>("base_lr"));
        Assert.Equal(new[] { 0.1f, 0.2f, 0.5f, 1.0f }, state.Get<float[]>("layer_multipliers"));
        Assert.Equal(new[] { "encoder.1", "encoder.2", "decoder.1", "decoder.2" }, state.Get<string[]>("layer_names"));
        Assert.Equal(1, state.Get<int>("step_count"));
        Assert.Equal(1, state.Get<int>("epoch_count"));
    }

    [Fact]
    public void DiscriminativeLRScheduler_StateDeserialization_Works()
    {
        var scheduler1 = new DiscriminativeLRScheduler(
            baseLearningRate: 0.01f,
            layerMultipliers: new[] { 0.1f, 1.0f },
            layerNames: new[] { "encoder", "decoder" }
        );

        scheduler1.Step();
        scheduler1.StepEpoch();

        var state = scheduler1.GetState();

        var scheduler2 = new DiscriminativeLRScheduler(
            baseLearningRate: 0.001f,
            layerMultipliers: new[] { 0.2f, 0.5f },
            layerNames: new[] { "layer1", "layer2" }
        );

        scheduler2.LoadState(state);

        Assert.Equal(1, scheduler2.StepCount);
        Assert.Equal(1, scheduler2.EpochCount);
    }

    #endregion
}
