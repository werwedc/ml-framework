using System;
using MLFramework.Checkpointing;
using MLFramework.Checkpointing.Extensions;
using Xunit;

namespace MLFramework.Tests.Checkpointing;

/// <summary>
/// Tests for CheckpointedModuleWrapper class
/// </summary>
public class CheckpointedModuleWrapperTests
{
    private class TestModule
    {
        public string Name { get; set; } = "TestModule";
    }

    [Fact]
    public void Constructor_WithValidParameters_CreatesWrapper()
    {
        var module = new TestModule();
        var wrapper = new CheckpointedModuleWrapper<TestModule>(module, "test_layer");

        Assert.NotNull(wrapper);
        Assert.Equal(module, wrapper.Module);
        Assert.Equal("test_layer", wrapper.LayerId);
        Assert.NotNull(wrapper.Config);
        Assert.NotNull(wrapper.GetStatistics());
    }

    [Fact]
    public void Constructor_WithNullModule_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new CheckpointedModuleWrapper<TestModule>(null!, "test_layer"));
    }

    [Fact]
    public void Constructor_WithNullLayerId_ThrowsArgumentNullException()
    {
        var module = new TestModule();

        Assert.Throws<ArgumentNullException>(() =>
            new CheckpointedModuleWrapper<TestModule>(module, null!));
    }

    [Fact]
    public void Constructor_WithCustomConfig_UsesCustomConfig()
    {
        var module = new TestModule();
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Selective,
            Interval = 3
        };

        var wrapper = new CheckpointedModuleWrapper<TestModule>(module, "test_layer", config);

        Assert.Equal(CheckpointStrategy.Selective, wrapper.Config.Strategy);
        Assert.Equal(3, wrapper.Config.Interval);
    }

    [Fact]
    public void EnableCheckpointing_EnablesCheckpointing()
    {
        var module = new TestModule();
        var wrapper = new CheckpointedModuleWrapper<TestModule>(module, "test_layer");

        wrapper.DisableCheckpointing();
        Assert.False(wrapper.GetStatistics().IsCheckpointingEnabled);

        wrapper.EnableCheckpointing();
        Assert.True(wrapper.GetStatistics().IsCheckpointingEnabled);
    }

    [Fact]
    public void DisableCheckpointing_DisablesCheckpointing()
    {
        var module = new TestModule();
        var wrapper = new CheckpointedModuleWrapper<TestModule>(module, "test_layer");

        Assert.True(wrapper.GetStatistics().IsCheckpointingEnabled);

        wrapper.DisableCheckpointing();
        Assert.False(wrapper.GetStatistics().IsCheckpointingEnabled);
    }

    [Fact]
    public void GetStatistics_ReturnsCorrectStatistics()
    {
        var module = new TestModule();
        var wrapper = new CheckpointedModuleWrapper<TestModule>(module, "test_layer");

        var stats = wrapper.GetStatistics();

        Assert.NotNull(stats);
        Assert.Equal("test_layer", stats.LayerId);
        Assert.True(stats.IsCheckpointingEnabled);
        Assert.Equal(0, stats.CheckpointCount);
        Assert.Equal(0, stats.RecomputationCount);
    }

    [Fact]
    public void Dispose_CleansUpResources()
    {
        var module = new TestModule();
        var wrapper = new CheckpointedModuleWrapper<TestModule>(module, "test_layer");

        wrapper.Dispose();

        // Should be able to call multiple times without error
        wrapper.Dispose();
    }

    [Fact]
    public void Module_ReturnsCorrectModule()
    {
        var module = new TestModule { Name = "CustomModule" };
        var wrapper = new CheckpointedModuleWrapper<TestModule>(module, "test_layer");

        Assert.Equal(module, wrapper.Module);
        Assert.Equal("CustomModule", wrapper.Module.Name);
    }

    [Fact]
    public void LayerId_ReturnsCorrectLayerId()
    {
        var module = new TestModule();
        var wrapper = new CheckpointedModuleWrapper<TestModule>(module, "custom_layer_id");

        Assert.Equal("custom_layer_id", wrapper.LayerId);
    }

    [Fact]
    public void Config_ReturnsCorrectConfig()
    {
        var module = new TestModule();
        var config = new CheckpointConfig
        {
            Strategy = CheckpointStrategy.Interval,
            Interval = 5,
            MaxMemoryPercentage = 0.9f
        };

        var wrapper = new CheckpointedModuleWrapper<TestModule>(module, "test_layer", config);

        Assert.Equal(config, wrapper.Config);
        Assert.Equal(CheckpointStrategy.Interval, wrapper.Config.Strategy);
        Assert.Equal(5, wrapper.Config.Interval);
        Assert.Equal(0.9f, wrapper.Config.MaxMemoryPercentage);
    }
}
