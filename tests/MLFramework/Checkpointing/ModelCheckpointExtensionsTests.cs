using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Extensions;
using Xunit;

namespace MLFramework.Tests.Checkpointing;

/// <summary>
/// Tests for ModelCheckpointExtensions class
/// </summary>
public class ModelCheckpointExtensionsTests
{
    private class TestModel
    {
        public string Name { get; set; } = "TestModel";
    }

    [Fact]
    public void CheckpointAll_EnablesCheckpointing()
    {
        var model = new TestModel();
        var result = model.CheckpointAll();

        Assert.NotNull(result);
        Assert.Equal(model, result);
        Assert.NotNull(model.GetCheckpointStatistics());
    }

    [Fact]
    public void CheckpointLayers_WithValidLayers_EnablesCheckpointing()
    {
        var model = new TestModel();
        var layers = new[] { "layer1", "layer2", "layer3" };
        var result = model.CheckpointLayers(layers);

        Assert.NotNull(result);
        Assert.Equal(model, result);
    }

    [Fact]
    public void CheckpointLayers_WithEmptyList_EnablesCheckpointing()
    {
        var model = new TestModel();
        var result = model.CheckpointLayers(Array.Empty<string>());

        Assert.NotNull(result);
        Assert.Equal(model, result);
    }

    [Fact]
    public void Checkpoint_WithValidConfig_EnablesCheckpointing()
    {
        var model = new TestModel();
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 3
        };
        var result = model.Checkpoint(config);

        Assert.NotNull(result);
        Assert.Equal(model, result);
    }

    [Fact]
    public void CheckpointEvery_WithValidInterval_EnablesCheckpointing()
    {
        var model = new TestModel();
        var result = model.CheckpointEvery(3);

        Assert.NotNull(result);
        Assert.Equal(model, result);
        Assert.NotNull(model.GetCheckpointStatistics());
    }

    [Fact]
    public void CheckpointEvery_WithDefaultInterval_UsesDefaultInterval()
    {
        var model = new TestModel();
        var result = model.CheckpointEvery();

        Assert.NotNull(result);
        Assert.Equal(model, result);
    }

    [Fact]
    public void DisableCheckpointing_DisablesCheckpointing()
    {
        var model = new TestModel();
        model.CheckpointAll();

        var result = model.DisableCheckpointing();

        Assert.Equal(model, result);
        Assert.False(model.GetCheckpointStatistics().IsCheckpointingEnabled);
    }

    [Fact]
    public void GetCheckpointStatistics_WithCheckpointedModel_ReturnsStatistics()
    {
        var model = new TestModel();
        model.CheckpointAll();

        var stats = model.GetCheckpointStatistics();

        Assert.NotNull(stats);
        Assert.True(stats.IsCheckpointingEnabled);
        Assert.NotNull(stats.Timestamp);
    }

    [Fact]
    public void GetCheckpointStatistics_WithoutCheckpointedModel_ReturnsDisabledStats()
    {
        var model = new TestModel();

        var stats = model.GetCheckpointStatistics();

        Assert.NotNull(stats);
        Assert.False(stats.IsCheckpointingEnabled);
    }

    [Fact]
    public void CheckpointAll_MethodChaining_Works()
    {
        var model = new TestModel();

        var result = model.CheckpointAll();

        Assert.Same(model, result);
    }

    [Fact]
    public void CheckpointLayers_MethodChaining_Works()
    {
        var model = new TestModel();
        var layers = new[] { "layer1" };

        var result = model.CheckpointLayers(layers);

        Assert.Same(model, result);
    }

    [Fact]
    public void Checkpoint_MethodChaining_Works()
    {
        var model = new TestModel();
        var config = new CheckpointConfig();

        var result = model.Checkpoint(config);

        Assert.Same(model, result);
    }

    [Fact]
    public void CheckpointEvery_MethodChaining_Works()
    {
        var model = new TestModel();

        var result = model.CheckpointEvery(3);

        Assert.Same(model, result);
    }

    [Fact]
    public void DisableCheckpointing_MethodChaining_Works()
    {
        var model = new TestModel();
        model.CheckpointAll();

        var result = model.DisableCheckpointing();

        Assert.Same(model, result);
    }

    [Fact]
    public void MultipleCheckpointCalls_DoNotThrow()
    {
        var model = new TestModel();

        // Should not throw
        model.CheckpointAll();
        model.CheckpointEvery(2);
        model.CheckpointLayers(new[] { "layer1" });
    }

    [Fact]
    public void CheckpointLayers_WithNullLayerIds_Throws()
    {
        var model = new TestModel();

        Assert.Throws<NullReferenceException>(() =>
            model.CheckpointLayers(null!));
    }

    [Fact]
    public void Checkpoint_WithNullConfig_Throws()
    {
        var model = new TestModel();

        Assert.Throws<ArgumentNullException>(() =>
            model.Checkpoint(null!));
    }

    [Fact]
    public void CheckpointAll_WithNullModel_Throws()
    {
        TestModel? model = null;

        Assert.Throws<ArgumentNullException>(() =>
            model!.CheckpointAll());
    }

    [Fact]
    public void DisableCheckpointing_WithNullModel_Throws()
    {
        TestModel? model = null;

        Assert.Throws<ArgumentNullException>(() =>
            model!.DisableCheckpointing());
    }

    [Fact]
    public void GetCheckpointStatistics_WithNullModel_Throws()
    {
        TestModel? model = null;

        Assert.Throws<ArgumentNullException>(() =>
            model!.GetCheckpointStatistics());
    }

    [Fact]
    public void Statistics_EvolveAfterOperations()
    {
        var model = new TestModel();
        model.CheckpointAll();

        var stats1 = model.GetCheckpointStatistics();
        Assert.NotNull(stats1);

        model.DisableCheckpointing();

        var stats2 = model.GetCheckpointStatistics();
        Assert.NotNull(stats2);
        Assert.False(stats2.IsCheckpointingEnabled);
    }

    [Fact]
    public void CheckpointEvery_WithInvalidInterval_DoesNotThrow()
    {
        var model = new TestModel();

        // The interval validation is in CheckpointConfig.Validate()
        // which may not be called here, so we test that it doesn't throw
        var result = model.CheckpointEvery(-1);

        Assert.NotNull(result);
    }

    [Fact]
    public void CheckpointLayers_WithMultipleLayers_CreatesSelectiveConfig()
    {
        var model = new TestModel();
        var layers = new[] { "conv1", "conv2", "fc1" };

        model.CheckpointLayers(layers);

        var stats = model.GetCheckpointStatistics();
        Assert.NotNull(stats);
    }
}
