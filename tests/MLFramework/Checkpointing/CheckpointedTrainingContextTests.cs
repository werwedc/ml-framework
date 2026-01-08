using System;
using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Extensions;
using Xunit;

namespace MLFramework.Tests.Checkpointing;

/// <summary>
/// Tests for CheckpointedTrainingContext class
/// </summary>
public class CheckpointedTrainingContextTests
{
    private class TestModel
    {
        public string Name { get; set; } = "TestModel";
    }

    [Fact]
    public void Constructor_WithValidParameters_CreatesContext()
    {
        var model = new TestModel();
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 3
        };

        var context = new CheckpointedTrainingContext<TestModel>(model, config);

        Assert.NotNull(context);
        Assert.Equal(model, context.Model);
        Assert.Equal(config, context.Config);
    }

    [Fact]
    public void Constructor_WithNullModel_Throws()
    {
        var config = new CheckpointConfig();

        Assert.Throws<ArgumentNullException>(() =>
            new CheckpointedTrainingContext<TestModel>(null!, config));
    }

    [Fact]
    public void Constructor_WithNullConfig_Throws()
    {
        var model = new TestModel();

        Assert.Throws<ArgumentNullException>(() =>
            new CheckpointedTrainingContext<TestModel>(model, null!));
    }

    [Fact]
    public void Model_ReturnsCorrectModel()
    {
        var model = new TestModel { Name = "CustomModel" };
        var config = new CheckpointConfig();

        var context = new CheckpointedTrainingContext<TestModel>(model, config);

        Assert.Equal(model, context.Model);
        Assert.Equal("CustomModel", context.Model.Name);
    }

    [Fact]
    public void Config_ReturnsCorrectConfig()
    {
        var model = new TestModel();
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Selective,
            Interval = 4,
            MaxMemoryPercentage = 0.85f
        };

        var context = new CheckpointedTrainingContext<TestModel>(model, config);

        Assert.Equal(config, context.Config);
        Assert.Equal(CheckpointStrategy.Selective, context.Config.Strategy);
        Assert.Equal(4, context.Config.Interval);
        Assert.Equal(0.85f, context.Config.MaxMemoryPercentage);
    }

    [Fact]
    public void GetStatistics_ReturnsCorrectStatistics()
    {
        var model = new TestModel();
        var config = new CheckpointConfig();

        var context = new CheckpointedTrainingContext<TestModel>(model, config);

        var stats = context.GetStatistics();

        Assert.NotNull(stats);
        Assert.NotNull(stats.Timestamp);
    }

    [Fact]
    public void Dispose_CleansUpResources()
    {
        var model = new TestModel();
        var config = new CheckpointConfig();

        var context = new CheckpointedTrainingContext<TestModel>(model, config);

        context.Dispose();

        // Should be able to call multiple times without error
        context.Dispose();
    }

    [Fact]
    public void Dispose_WhenCalled_ExitsContext()
    {
        var model = new TestModel();
        var config = new CheckpointConfig();

        var context = new CheckpointedTrainingContext<TestModel>(model, config);

        // Should not throw
        context.Dispose();
    }

    [Fact]
    public void GetStatistics_AfterDispose_Throws()
    {
        var model = new TestModel();
        var config = new CheckpointConfig();

        var context = new CheckpointedTrainingContext<TestModel>(model, config);
        context.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
            context.GetStatistics());
    }

    [Fact]
    public void CanBeUsedInUsingStatement()
    {
        var model = new TestModel();
        var config = new CheckpointConfig();

        using (var context = new CheckpointedTrainingContext<TestModel>(model, config))
        {
            Assert.NotNull(context);
            var stats = context.GetStatistics();
            Assert.NotNull(stats);
        }

        // After disposal, should throw
        Assert.Throws<ObjectDisposedException>(() =>
            context.GetStatistics());
    }

    [Fact]
    public void MultipleContexts_CanBeCreated()
    {
        var model = new TestModel();
        var config = new CheckpointConfig();

        var context1 = new CheckpointedTrainingContext<TestModel>(model, config);
        var context2 = new CheckpointedTrainingContext<TestModel>(model, config);

        Assert.NotSame(context1, context2);
        Assert.Equal(model, context1.Model);
        Assert.Equal(model, context2.Model);
    }

    [Fact]
    public void Context_PreservesModelReference()
    {
        var model = new TestModel { Name = "OriginalModel" };
        var config = new CheckpointConfig();

        var context = new CheckpointedTrainingContext<TestModel>(model, config);

        // Modify model and verify context still has reference
        model.Name = "ModifiedModel";
        Assert.Equal("ModifiedModel", context.Model.Name);
    }

    [Fact]
    public void Context_PreservesConfig()
    {
        var model = new TestModel();
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 3
        };

        var context = new CheckpointedTrainingContext<TestModel>(model, config);

        Assert.Equal(CheckpointStrategy.Interval, context.Config.Strategy);
        Assert.Equal(3, context.Config.Interval);
    }
}
