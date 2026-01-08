using System;
using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Extensions;
using Xunit;

namespace MLFramework.Tests.Checkpointing;

/// <summary>
/// Tests for TrainingLoopExtensions class
/// </summary>
public class TrainingLoopExtensionsTests
{
    private class TestModel
    {
        public string Name { get; set; } = "TestModel";
    }

    [Fact]
    public void WithCheckpointing_WithConfig_CreatesContext()
    {
        var model = new TestModel();
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 3
        };

        var context = model.WithCheckpointing(config);

        Assert.NotNull(context);
        Assert.Equal(model, context.Model);
        Assert.Equal(config, context.Config);
    }

    [Fact]
    public void WithCheckpointing_WithoutConfig_CreatesContextWithDefault()
    {
        var model = new TestModel();

        var context = model.WithCheckpointing();

        Assert.NotNull(context);
        Assert.Equal(model, context.Model);
        Assert.Equal(CheckpointConfig.Default.Strategy, context.Config.Strategy);
        Assert.Equal(CheckpointConfig.Default.Interval, context.Config.Interval);
    }

    [Fact]
    public void WithCheckpointing_WithValidModel_CreatesContext()
    {
        var model = new TestModel();

        var context = model.WithCheckpointing();

        Assert.NotNull(context);
        Assert.NotNull(context.Model);
        Assert.NotNull(context.Config);
    }

    [Fact]
    public void WithCheckpointing_WithNullModel_Throws()
    {
        TestModel? model = null;

        Assert.Throws<ArgumentNullException>(() =>
            model!.WithCheckpointing());
    }

    [Fact]
    public void WithCheckpointing_CanBeUsedInUsingStatement()
    {
        var model = new TestModel();

        using (var context = model.WithCheckpointing())
        {
            Assert.NotNull(context);
            Assert.Equal(model, context.Model);
        }

        // Should not throw after disposal
    }

    [Fact]
    public void WithCheckpointing_ReturnsCheckpointedTrainingContext()
    {
        var model = new TestModel();

        var context = model.WithCheckpointing();

        Assert.IsAssignableFrom<CheckpointedTrainingContext<TestModel>>(context);
    }

    [Fact]
    public void WithCheckpointing_WithCustomConfig_UsesCustomConfig()
    {
        var model = new TestModel();
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Selective,
            Interval = 5,
            MaxMemoryPercentage = 0.9f
        };

        var context = model.WithCheckpointing(config);

        Assert.Equal(config.Strategy, context.Config.Strategy);
        Assert.Equal(config.Interval, context.Config.Interval);
        Assert.Equal(config.MaxMemoryPercentage, context.Config.MaxMemoryPercentage);
    }

    [Fact]
    public void MultipleContexts_CanBeCreated()
    {
        var model = new TestModel();

        var context1 = model.WithCheckpointing();
        var context2 = model.WithCheckpointing();

        Assert.NotSame(context1, context2);
        Assert.Equal(model, context1.Model);
        Assert.Equal(model, context2.Model);
    }

    [Fact]
    public void WithCheckpointing_MethodChaining_Works()
    {
        var model = new TestModel();

        var context = model.WithCheckpointing();

        Assert.NotNull(context);
    }
}
