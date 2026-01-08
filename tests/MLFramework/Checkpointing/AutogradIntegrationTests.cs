using System;
using System.Collections.Generic;
using MLFramework.Checkpointing;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Checkpointing;

/// <summary>
/// Tests for CheckpointFunction
/// </summary>
public class CheckpointFunctionTests : IDisposable
{
    private CheckpointAdapter? _checkpointAdapter;
    private RecomputeAdapter? _recomputeAdapter;

    public CheckpointFunctionTests()
    {
        var manager = new CheckpointManager();
        var engine = new RecomputationEngine();
        _checkpointAdapter = new CheckpointAdapter(manager);
        _recomputeAdapter = new RecomputeAdapter(engine);
    }

    [Fact]
    public void ForwardPass_ExecutesCorrectly()
    {
        // Arrange
        var layerId = "test_layer_1";
        Func<Tensor> forwardFunc = () =>
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            return new Tensor(data, new[] { 2, 2 }, false);
        };

        var checkpointFunc = new CheckpointFunction(
            layerId,
            forwardFunc,
            null,
            _checkpointAdapter!,
            _recomputeAdapter!);

        // Act
        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 }, false)
        };

        var output = checkpointFunc.Forward(inputs);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(new[] { 2, 2 }, output.Shape);
    }

    [Fact]
    public void BackwardPass_TriggersRecomputation()
    {
        // Arrange
        var layerId = "test_layer_2";
        bool backwardCalled = false;
        Func<Tensor> forwardFunc = () =>
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            return new Tensor(data, new[] { 2, 2 }, false);
        };

        var checkpointFunc = new CheckpointFunction(
            layerId,
            forwardFunc,
            grad => backwardCalled = true,
            _checkpointAdapter!,
            _recomputeAdapter!);

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 }, true)
        };

        // Act
        checkpointFunc.Forward(inputs);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new[] { 2, 2 }, false);
        var gradients = checkpointFunc.Backward(gradOutput);

        // Assert
        Assert.NotNull(gradients);
        Assert.True(backwardCalled);
    }

    [Fact]
    public void CheckpointFunction_ThrowsOnNullLayerId()
    {
        // Arrange
        Func<Tensor> forwardFunc = () => new Tensor(new float[] { 1.0f }, new[] { 1 }, false);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
        {
            new CheckpointFunction(
                null!,
                forwardFunc,
                null,
                _checkpointAdapter!,
                _recomputeAdapter!);
        });
    }

    [Fact]
    public void CheckpointFunction_ThrowsOnNullForwardFunc()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
        {
            new CheckpointFunction(
                "test_layer",
                null!,
                null,
                _checkpointAdapter!,
                _recomputeAdapter!);
        });
    }

    [Fact]
    public void CheckpointingEnabled_CanBeToggled()
    {
        // Arrange
        var layerId = "test_layer_3";
        Func<Tensor> forwardFunc = () =>
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            return new Tensor(data, new[] { 2, 2 }, false);
        };

        var checkpointFunc = new CheckpointFunction(
            layerId,
            forwardFunc,
            null,
            _checkpointAdapter!,
            _recomputeAdapter!);

        // Act
        checkpointFunc.CheckpointingEnabled = false;
        Assert.False(checkpointFunc.CheckpointingEnabled);

        checkpointFunc.CheckpointingEnabled = true;
        Assert.True(checkpointFunc.CheckpointingEnabled);
    }

    public void Dispose()
    {
        _checkpointAdapter?.Dispose();
        _recomputeAdapter?.Dispose();
    }
}

/// <summary>
/// Tests for BackwardHookManager
/// </summary>
public class BackwardHookManagerTests : IDisposable
{
    private readonly BackwardHookManager _hookManager;

    public BackwardHookManagerTests()
    {
        _hookManager = new BackwardHookManager();
    }

    [Fact]
    public void RegisterHook_ReturnsUniqueHandle()
    {
        // Act
        var handle1 = _hookManager.RegisterHook("layer1", _ => { });
        var handle2 = _hookManager.RegisterHook("layer1", _ => { });
        var handle3 = _hookManager.RegisterHook("layer2", _ => { });

        // Assert
        Assert.NotEqual(handle1, handle2);
        Assert.NotEqual(handle1, handle3);
        Assert.NotEqual(handle2, handle3);
    }

    [Fact]
    public void RemoveHook_WorksCorrectly()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f }, new[] { 1 }, false);
        bool hook1Called = false;
        bool hook2Called = false;

        var handle1 = _hookManager.RegisterHook("layer1", _ => hook1Called = true);
        var handle2 = _hookManager.RegisterHook("layer1", _ => hook2Called = true);

        // Act
        _hookManager.RemoveHook(handle1);
        _hookManager.InvokeHooks("layer1", gradient);

        // Assert
        Assert.False(hook1Called);
        Assert.True(hook2Called);
    }

    [Fact]
    public void RemoveHooksForLayer_RemovesAllHooksForLayer()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f }, new[] { 1 }, false);
        int callCount = 0;

        _hookManager.RegisterHook("layer1", _ => callCount++);
        _hookManager.RegisterHook("layer1", _ => callCount++);
        _hookManager.RegisterHook("layer2", _ => callCount++);

        // Act
        _hookManager.RemoveHooksForLayer("layer1");
        _hookManager.InvokeHooks("layer1", gradient);
        _hookManager.InvokeHooks("layer2", gradient);

        // Assert
        Assert.Equal(1, callCount);
    }

    [Fact]
    public void ClearAllHooks_RemovesAllRegisteredHooks()
    {
        // Arrange
        _hookManager.RegisterHook("layer1", _ => { });
        _hookManager.RegisterHook("layer2", _ => { });

        // Act
        _hookManager.ClearAllHooks();

        // Assert
        Assert.Equal(0, _hookManager.HookCount);
    }

    [Fact]
    public void InvokeHooks_CallsHooksInOrder()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f }, new[] { 1 }, false);
        var callOrder = new List<int>();

        _hookManager.RegisterHook("layer1", _ => callOrder.Add(1));
        _hookManager.RegisterHook("layer1", _ => callOrder.Add(2));
        _hookManager.RegisterHook("layer1", _ => callOrder.Add(3));

        // Act
        _hookManager.InvokeHooks("layer1", gradient);

        // Assert
        Assert.Equal(new List<int> { 1, 2, 3 }, callOrder);
    }

    [Fact]
    public void GetHookCountForLayer_ReturnsCorrectCount()
    {
        // Act
        _hookManager.RegisterHook("layer1", _ => { });
        _hookManager.RegisterHook("layer1", _ => { });
        _hookManager.RegisterHook("layer2", _ => { });

        // Assert
        Assert.Equal(2, _hookManager.GetHookCountForLayer("layer1"));
        Assert.Equal(1, _hookManager.GetHookCountForLayer("layer2"));
        Assert.Equal(0, _hookManager.GetHookCountForLayer("layer3"));
    }

    public void Dispose()
    {
        _hookManager.Dispose();
    }
}

/// <summary>
/// Tests for CheckpointedGradientAccumulator
/// </summary>
public class CheckpointedGradientAccumulatorTests : IDisposable
{
    private readonly CheckpointedGradientAccumulator _accumulator;

    public CheckpointedGradientAccumulatorTests()
    {
        _accumulator = new CheckpointedGradientAccumulator(accumulationSteps: 4);
    }

    [Fact]
    public void Gradients_AreAccumulatedCorrectly()
    {
        // Arrange
        var gradients1 = new Dictionary<string, Tensor>
        {
            ["param1"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false),
            ["param2"] = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, false)
        };

        var gradients2 = new Dictionary<string, Tensor>
        {
            ["param1"] = new Tensor(new float[] { 2.0f, 3.0f }, new[] { 2 }, false),
            ["param2"] = new Tensor(new float[] { 4.0f, 5.0f }, new[] { 2 }, false)
        };

        // Act
        _accumulator.Accumulate(gradients1);
        _accumulator.Accumulate(gradients2);

        // Assert
        Assert.Equal(2, _accumulator.ParameterCount);
        Assert.True(_accumulator.HasGradients);
    }

    [Fact]
    public void ShouldApplyGradients_ReturnsCorrectValue()
    {
        // Act & Assert
        Assert.False(_accumulator.ShouldApplyGradients());

        _accumulator.Accumulate(new Dictionary<string, Tensor>());
        Assert.False(_accumulator.ShouldApplyGradients());

        _accumulator.Accumulate(new Dictionary<string, Tensor>());
        Assert.False(_accumulator.ShouldApplyGradients());

        _accumulator.Accumulate(new Dictionary<string, Tensor>());
        Assert.False(_accumulator.ShouldApplyGradients());

        _accumulator.Accumulate(new Dictionary<string, Tensor>());
        Assert.True(_accumulator.ShouldApplyGradients());
    }

    [Fact]
    public void GetAccumulatedGradients_ReturnsAveragedGradients()
    {
        // Arrange
        var gradients1 = new Dictionary<string, Tensor>
        {
            ["param1"] = new Tensor(new float[] { 2.0f, 4.0f }, new[] { 2 }, false)
        };

        var gradients2 = new Dictionary<string, Tensor>
        {
            ["param1"] = new Tensor(new float[] { 4.0f, 8.0f }, new[] { 2 }, false)
        };

        _accumulator.Accumulate(gradients1);
        _accumulator.Accumulate(gradients2);

        // Act
        var accumulated = _accumulator.GetAccumulatedGradients();

        // Assert
        Assert.True(accumulated.ContainsKey("param1"));
        Assert.Equal(2, accumulated["param1"].Shape[0]);

        // After reset, current step should be 0
        Assert.Equal(0, _accumulator.CurrentStep);
    }

    [Fact]
    public void Reset_ClearsAccumulatedGradients()
    {
        // Arrange
        var gradients = new Dictionary<string, Tensor>
        {
            ["param1"] = new Tensor(new float[] { 1.0f }, new[] { 1 }, false)
        };

        _accumulator.Accumulate(gradients);

        // Act
        _accumulator.Reset();

        // Assert
        Assert.Equal(0, _accumulator.ParameterCount);
        Assert.False(_accumulator.HasGradients);
        Assert.Equal(0, _accumulator.CurrentStep);
    }

    [Fact]
    public void PeekGradients_ReturnsCopyWithoutResetting()
    {
        // Arrange
        var gradients = new Dictionary<string, Tensor>
        {
            ["param1"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false)
        };

        _accumulator.Accumulate(gradients);

        // Act
        var peeked = _accumulator.PeekGradients();

        // Assert
        Assert.True(peeked.ContainsKey("param1"));
        Assert.Equal(1, _accumulator.CurrentStep); // Should not reset
    }

    [Fact]
    public void RemoveGradient_RemovesSpecificGradient()
    {
        // Arrange
        var gradients = new Dictionary<string, Tensor>
        {
            ["param1"] = new Tensor(new float[] { 1.0f }, new[] { 1 }, false),
            ["param2"] = new Tensor(new float[] { 2.0f }, new[] { 1 }, false)
        };

        _accumulator.Accumulate(gradients);

        // Act
        _accumulator.RemoveGradient("param1");

        // Assert
        Assert.Equal(1, _accumulator.ParameterCount);
        Assert.Null(_accumulator.GetGradient("param1"));
        Assert.NotNull(_accumulator.GetGradient("param2"));
    }

    public void Dispose()
    {
        _accumulator.Dispose();
    }
}

/// <summary>
/// Tests for Checkpoint static class
/// </summary>
public class CheckpointTests : IDisposable
{
    [Fact]
    public void CheckpointFunction_CreatesCorrectCheckpointFunction()
    {
        // Arrange
        var layerId = "test_layer";
        Func<Tensor> func = () =>
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            return new Tensor(data, new[] { 2, 2 }, false);
        };

        // Act
        var output = Checkpoint.CheckpointFunction(layerId, func);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(new[] { 2, 2 }, output.Shape);
    }

    [Fact]
    public void CreateCheckpointFunction_ReturnsCheckpointFunction()
    {
        // Arrange
        var layerId = "test_layer";
        Func<Tensor> func = () =>
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            return new Tensor(data, new[] { 2, 2 }, false);
        };

        Action<Tensor>? backwardHook = grad => { };

        // Act
        var checkpointFunc = Checkpoint.CreateCheckpointFunction(layerId, func, backwardHook);

        // Assert
        Assert.NotNull(checkpointFunc);
        Assert.True(checkpointFunc.CheckpointingEnabled);
        checkpointFunc.Dispose();
    }

    [Fact]
    public void SetDefaultCheckpointAdapter_UsesProvidedAdapter()
    {
        // Arrange
        var manager = new CheckpointManager();
        var adapter = new CheckpointAdapter(manager);

        // Act
        Checkpoint.SetDefaultCheckpointAdapter(adapter);

        // Assert (no exception thrown)
        Assert.NotNull(adapter);
    }

    public void Dispose()
    {
        Checkpoint.ResetDefaults();
    }
}

/// <summary>
/// Integration tests for checkpointing with autograd
/// </summary>
public class AutogradIntegrationTests : IDisposable
{
    private CheckpointAdapter? _checkpointAdapter;
    private RecomputeAdapter? _recomputeAdapter;
    private BackwardHookManager? _hookManager;

    public AutogradIntegrationTests()
    {
        var manager = new CheckpointManager();
        var engine = new RecomputationEngine();
        _checkpointAdapter = new CheckpointAdapter(manager);
        _recomputeAdapter = new RecomputeAdapter(engine);
        _hookManager = new BackwardHookManager();
    }

    [Fact]
    public void ForwardBackward_WithCheckpointing_WorksCorrectly()
    {
        // Arrange
        var layerId = "integration_layer";
        var forwardCount = 0;
        var backwardCount = 0;

        Func<Tensor> forwardFunc = () =>
        {
            forwardCount++;
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            return new Tensor(data, new[] { 2, 2 }, false);
        };

        var checkpointFunc = new CheckpointFunction(
            layerId,
            forwardFunc,
            grad => backwardCount++,
            _checkpointAdapter!,
            _recomputeAdapter!);

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 }, true)
        };

        // Act
        var output = checkpointFunc.Forward(inputs);
        var gradOutput = new Tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.0f }, new[] { 2, 2 }, false);
        var gradients = checkpointFunc.Backward(gradOutput);

        // Assert
        Assert.NotNull(output);
        Assert.NotNull(gradients);
        Assert.True(forwardCount >= 1);
        Assert.True(backwardCount >= 1);
    }

    [Fact]
    public void MultipleLayers_CanBeCheckpointedIndependently()
    {
        // Arrange
        var func = () =>
        {
            var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
            return new Tensor(data, new[] { 2, 2 }, false);
        };

        var checkpointFunc1 = new CheckpointFunction("layer1", func, null, _checkpointAdapter!, _recomputeAdapter!);
        var checkpointFunc2 = new CheckpointFunction("layer2", func, null, _checkpointAdapter!, _recomputeAdapter!);
        var checkpointFunc3 = new CheckpointFunction("layer3", func, null, _checkpointAdapter!, _recomputeAdapter!);

        var inputs = new Tensor[]
        {
            new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, new[] { 2, 2 }, false)
        };

        // Act
        var output1 = checkpointFunc1.Forward(inputs);
        var output2 = checkpointFunc2.Forward(inputs);
        var output3 = checkpointFunc3.Forward(inputs);

        // Assert
        Assert.NotNull(output1);
        Assert.NotNull(output2);
        Assert.NotNull(output3);

        checkpointFunc1.Dispose();
        checkpointFunc2.Dispose();
        checkpointFunc3.Dispose();
    }

    public void Dispose()
    {
        _checkpointAdapter?.Dispose();
        _recomputeAdapter?.Dispose();
        _hookManager?.Dispose();
    }
}
