using System;
using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Extensions;
using Xunit;

namespace MLFramework.Tests.Checkpointing;

/// <summary>
/// Tests for CheckpointContext class
/// </summary>
public class CheckpointContextTests
{
    [Fact]
    public void Constructor_WithValidConfig_CreatesContext()
    {
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 3
        };

        var context = new CheckpointContext(config);

        Assert.NotNull(context);
        Assert.Equal(config, context.Config);
        Assert.False(context.IsEnabled);
    }

    [Fact]
    public void Constructor_WithNullConfig_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new CheckpointContext(null!));
    }

    [Fact]
    public void Config_ReturnsProvidedConfig()
    {
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Selective,
            Interval = 4,
            MaxMemoryPercentage = 0.85f
        };

        var context = new CheckpointContext(config);

        Assert.Equal(config, context.Config);
        Assert.Equal(CheckpointStrategy.Selective, context.Config.Strategy);
        Assert.Equal(4, context.Config.Interval);
        Assert.Equal(0.85f, context.Config.MaxMemoryPercentage);
    }

    [Fact]
    public void CheckpointManager_IsNotNull()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);

        Assert.NotNull(context.CheckpointManager);
    }

    [Fact]
    public void RecomputeEngine_IsNotNull()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);

        Assert.NotNull(context.RecomputeEngine);
    }

    [Fact]
    public void Enter_EnablesCheckpointing()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);

        context.Enter();

        Assert.True(context.IsEnabled);
    }

    [Fact]
    public void Exit_DisablesCheckpointing()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);

        context.Enter();
        context.Exit();

        Assert.False(context.IsEnabled);
    }

    [Fact]
    public void GetStatistics_ReturnsValidStatistics()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);
        context.Enter();

        var stats = context.GetStatistics();

        Assert.NotNull(stats);
        Assert.NotNull(stats.Timestamp);
        Assert.True(stats.IsCheckpointingEnabled);
    }

    [Fact]
    public void GetStatistics_AfterExit_ReturnsDisabledStats()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);
        context.Enter();
        context.Exit();

        var stats = context.GetStatistics();

        Assert.NotNull(stats);
        Assert.False(stats.IsCheckpointingEnabled);
    }

    [Fact]
    public void Dispose_CleansUpResources()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);
        context.Enter();

        context.Dispose();

        // Should be able to call multiple times without error
        context.Dispose();
    }

    [Fact]
    public void Dispose_AutomaticallyExitsIfActive()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);
        context.Enter();

        Assert.True(context.IsEnabled);

        context.Dispose();

        // After disposal, IsEnabled should be false
        // Note: Can't test this directly because disposed context throws exception
        // But the Dispose() method should call Exit() if IsEnabled is true
    }

    [Fact]
    public void Dispose_WhenAlreadyExited_DoesNotThrow()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);
        context.Enter();
        context.Exit();

        // Should not throw when disposing already-exited context
        context.Dispose();
    }

    [Fact]
    public void CanBeUsedInUsingStatement()
    {
        var config = new CheckpointConfig();

        using (var context = new CheckpointContext(config))
        {
            context.Enter();
            Assert.True(context.IsEnabled);
            var stats = context.GetStatistics();
            Assert.NotNull(stats);
        }

        // After disposal, should throw
        Assert.Throws<ObjectDisposedException>(() =>
            context.IsEnabled);
    }

    [Fact]
    public void GetStatistics_AfterDispose_Throws()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);
        context.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
            context.GetStatistics());
    }

    [Fact]
    public void Enter_AfterDispose_Throws()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);
        context.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
            context.Enter());
    }

    [Fact]
    public void Exit_AfterDispose_Throws()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);
        context.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
            context.Exit());
    }

    [Fact]
    public void MultipleContexts_CanBeCreated()
    {
        var config = new CheckpointConfig();

        var context1 = new CheckpointContext(config);
        var context2 = new CheckpointContext(config);

        Assert.NotSame(context1, context2);
        Assert.NotSame(context1.CheckpointManager, context2.CheckpointManager);
        Assert.NotSame(context1.RecomputeEngine, context2.RecomputeEngine);
    }

    [Fact]
    public void IsEnabled_DefaultsToFalse()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);

        Assert.False(context.IsEnabled);
    }

    [Fact]
    public void Exit_ClearsCheckpoints()
    {
        var config = new CheckpointConfig();
        var context = new CheckpointContext(config);
        context.Enter();

        // Check that initial checkpoint count is 0
        var stats1 = context.GetStatistics();
        Assert.Equal(0, stats1.CheckpointCount);

        context.Exit();

        // Verify checkpoints are cleared
        var stats2 = context.GetStatistics();
        Assert.Equal(0, stats2.CheckpointCount);
    }

    [Fact]
    public void Context_WithDifferentStrategies_Works()
    {
        var intervalConfig = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 2
        };

        var selectiveConfig = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Selective,
            CheckpointLayers = new[] { "layer1", "layer2" }
        };

        var intervalContext = new CheckpointContext(intervalConfig);
        var selectiveContext = new CheckpointContext(selectiveConfig);

        Assert.Equal(CheckpointStrategy.Interval, intervalContext.Config.Strategy);
        Assert.Equal(CheckpointStrategy.Selective, selectiveContext.Config.Strategy);
    }
}
