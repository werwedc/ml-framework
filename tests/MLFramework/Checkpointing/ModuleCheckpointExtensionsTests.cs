using System;
using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Extensions;
using Xunit;

namespace MLFramework.Tests.Checkpointing;

/// <summary>
/// Tests for ModuleCheckpointExtensions class
/// </summary>
public class ModuleCheckpointExtensionsTests
{
    private class TestModule
    {
        public string Name { get; set; } = "TestModule";
    }

    [Fact]
    public void AsCheckpointed_WithLayerId_CreatesWrapper()
    {
        var module = new TestModule();
        var checkpointed = module.AsCheckpointed("layer1");

        Assert.NotNull(checkpointed);
        Assert.IsAssignableFrom<ICheckpointedModule<TestModule>>(checkpointed);
        Assert.Equal("layer1", checkpointed.LayerId);
        Assert.Equal(module, checkpointed.Module);
    }

    [Fact]
    public void AsCheckpointed_WithConfig_CreatesWrapperWithConfig()
    {
        var module = new TestModule();
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 3
        };

        var checkpointed = module.AsCheckpointed("layer1", config);

        Assert.NotNull(checkpointed);
        Assert.Equal(config.Strategy, checkpointed.Config.Strategy);
        Assert.Equal(config.Interval, checkpointed.Config.Interval);
    }

    [Fact]
    public void AsCheckpointed_WithDefaultConfig_UsesDefaultConfig()
    {
        var module = new TestModule();

        var checkpointed = module.AsCheckpointed("layer1");

        Assert.NotNull(checkpointed);
        Assert.Equal(CheckpointStrategy.Interval, checkpointed.Config.Strategy);
        Assert.Equal(2, checkpointed.Config.Interval);
    }

    [Fact]
    public void AsCheckpointed_WrapsModuleCorrectly()
    {
        var module = new TestModule { Name = "OriginalName" };

        var checkpointed = module.AsCheckpointed("test_layer");

        Assert.Equal(module, checkpointed.Module);
        Assert.Equal("OriginalName", checkpointed.Module.Name);
    }

    [Fact]
    public void AsCheckpointed_CreatesNewWrapperEachTime()
    {
        var module = new TestModule();

        var wrapper1 = module.AsCheckpointed("layer1");
        var wrapper2 = module.AsCheckpointed("layer2");

        Assert.NotSame(wrapper1, wrapper2);
        Assert.Equal("layer1", wrapper1.LayerId);
        Assert.Equal("layer2", wrapper2.LayerId);
    }

    [Fact]
    public void AsCheckpointed_ReturnsICheckpointedModule()
    {
        var module = new TestModule();

        var checkpointed = module.AsCheckpointed("layer1");

        Assert.IsAssignableFrom<ICheckpointedModule<TestModule>>(checkpointed);
    }

    [Fact]
    public void AsCheckpointed_CanUseDefaultConfig()
    {
        var module = new TestModule();

        var checkpointed = module.AsCheckpointed("test_layer");

        Assert.NotNull(checkpointed.Config);
        Assert.Equal(CheckpointConfig.Default.Strategy, checkpointed.Config.Strategy);
        Assert.Equal(CheckpointConfig.Default.Interval, checkpointed.Config.Interval);
    }

    [Fact]
    public void AsCheckpointed_CanEnableDisableCheckpointing()
    {
        var module = new TestModule();

        var checkpointed = module.AsCheckpointed("test_layer");

        Assert.True(checkpointed.GetStatistics().IsCheckpointingEnabled);

        checkpointed.DisableCheckpointing();
        Assert.False(checkpointed.GetStatistics().IsCheckpointingEnabled);

        checkpointed.EnableCheckpointing();
        Assert.True(checkpointed.GetStatistics().IsCheckpointingEnabled);
    }

    [Fact]
    public void AsCheckpointed_CanGetStatistics()
    {
        var module = new TestModule();

        var checkpointed = module.AsCheckpointed("test_layer");

        var stats = checkpointed.GetStatistics();

        Assert.NotNull(stats);
        Assert.Equal("test_layer", stats.LayerId);
        Assert.NotNull(stats.Timestamp);
    }

    [Fact]
    public void AsCheckpointed_CanDispose()
    {
        var module = new TestModule();

        var checkpointed = module.AsCheckpointed("test_layer");

        // Should not throw
        checkpointed.Dispose();
    }
}
