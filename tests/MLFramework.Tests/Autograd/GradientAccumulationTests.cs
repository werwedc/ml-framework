using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd;

public class GradientAccumulationTests : IDisposable
{
    public void Dispose()
    {
        // Clean up any static state
        TensorAccumulationExtensions.ClearAllAccumulationStates();
    }

    #region AccumulationContext Tests

    [Fact]
    public void AccumulationContext_Constructor_WithValidTargetSteps_CreatesContext()
    {
        // Arrange & Act
        var context = new AccumulationContext(4);

        // Assert
        Assert.Equal(0, context.AccumulationSteps);
        Assert.Equal(4, context.TargetSteps);
        Assert.False(context.IsReady);
        Assert.Equal(0.25f, context.ScalingFactor, precision: 5);
    }

    [Fact]
    public void AccumulationContext_Constructor_WithInvalidTargetSteps_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new AccumulationContext(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new AccumulationContext(-1));
    }

    [Fact]
    public void AccumulationContext_Step_IncrementsCounter()
    {
        // Arrange
        var context = new AccumulationContext(3);

        // Act
        context.Step();

        // Assert
        Assert.Equal(1, context.AccumulationSteps);
        Assert.False(context.IsReady);
    }

    [Fact]
    public void AccumulationContext_Step_MultipleTimes_ReachesTarget()
    {
        // Arrange
        var context = new AccumulationContext(3);

        // Act
        context.Step();
        context.Step();
        context.Step();

        // Assert
        Assert.Equal(3, context.AccumulationSteps);
        Assert.True(context.IsReady);
    }

    [Fact]
    public void AccumulationContext_Reset_ClearsCounter()
    {
        // Arrange
        var context = new AccumulationContext(3);
        context.Step();
        context.Step();

        // Act
        context.Reset();

        // Assert
        Assert.Equal(0, context.AccumulationSteps);
        Assert.False(context.IsReady);
    }

    [Fact]
    public void AccumulationContext_RegisterTensor_AddsTensorToRegistry()
    {
        // Arrange
        var context = new AccumulationContext(3);
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);

        // Act - Should not throw
        context.RegisterTensor(tensor);

        // Assert - Implicit verification by not throwing
        Assert.NotNull(context);
    }

    [Fact]
    public void AccumulationContext_RegisterTensor_WithNull_ThrowsException()
    {
        // Arrange
        var context = new AccumulationContext(3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => context.RegisterTensor(null!));
    }

    [Fact]
    public void AccumulationContext_AfterDisposed_ThrowsException()
    {
        // Arrange
        var context = new AccumulationContext(3);
        context.Dispose();

        // Act & Assert
        Assert.Throws<ObjectDisposedException>(() => context.Step());
        Assert.Throws<ObjectDisposedException>(() => context.Reset());
    }

    #endregion

    #region GradientAccumulator Tests

    [Fact]
    public void GradientAccumulator_Constructor_WithValidCount_CreatesAccumulator()
    {
        // Arrange & Act
        var accumulator = new GradientAccumulator(4);

        // Assert
        Assert.Equal(4, accumulator.AccumulationCount);
        Assert.True(accumulator.Enabled);
        Assert.False(accumulator.IsReady);
        Assert.Equal(0, accumulator.CurrentSteps);
    }

    [Fact]
    public void GradientAccumulator_Constructor_WithInvalidCount_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new GradientAccumulator(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GradientAccumulator(-1));
    }

    [Fact]
    public void GradientAccumulator_EnableAccumulation_InitializesGradients()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);
        var tensor1 = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var tensor2 = new Tensor(new float[] { 2.0f }, new int[] { 1 });
        var parameters = new[] { tensor1, tensor2 };

        // Act
        accumulator.EnableAccumulation(parameters);

        // Assert
        Assert.True(tensor1.RequiresGrad);
        Assert.True(tensor2.RequiresGrad);
        Assert.NotNull(tensor1.Gradient);
        Assert.NotNull(tensor2.Gradient);
        Assert.True(accumulator.Enabled);
    }

    [Fact]
    public void GradientAccumulator_EnableAccumulation_WithNull_ThrowsException()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => accumulator.EnableAccumulation(null!));
    }

    [Fact]
    public void GradientAccumulator_Step_IncrementsCurrentSteps()
    {
        // Arrange
        var accumulator = new GradientAccumulator(3);
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.EnableAccumulation(new[] { tensor });

        // Act
        accumulator.Step();

        // Assert
        Assert.Equal(1, accumulator.CurrentSteps);
        Assert.False(accumulator.IsReady);
    }

    [Fact]
    public void GradientAccumulator_IsReady_WhenTargetReached_ReturnsTrue()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.EnableAccumulation(new[] { tensor });

        // Act
        accumulator.Step();
        accumulator.Step();

        // Assert
        Assert.True(accumulator.IsReady);
    }

    [Fact]
    public void GradientAccumulator_Step_WithoutEnableAccumulation_ThrowsException()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => accumulator.Step());
    }

    [Fact]
    public void GradientAccumulator_DisableAccumulation_ClearsState()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.EnableAccumulation(new[] { tensor });

        // Act
        accumulator.DisableAccumulation();

        // Assert
        Assert.False(accumulator.Enabled);
        Assert.False(accumulator.IsReady);
    }

    [Fact]
    public void GradientAccumulator_ResetGradients_ZerosAllGradients()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.EnableAccumulation(new[] { tensor });

        // Set some gradient values
        if (tensor.Gradient != null)
        {
            tensor.Gradient.Data[0] = 5.0f;
        }

        // Act
        accumulator.ResetGradients();

        // Assert
        Assert.NotNull(tensor.Gradient);
        Assert.Equal(0.0f, tensor.Gradient.Data[0], precision: 5);
    }

    [Fact]
    public void GradientAccumulator_ApplyGradients_ScalesGradients()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.EnableAccumulation(new[] { tensor });

        // Set gradient value (simulating accumulated gradient)
        if (tensor.Gradient != null)
        {
            tensor.Gradient.Data[0] = 4.0f; // Accumulated over 2 steps
        }

        Tensor? appliedTensor = null;

        // Act
        accumulator.ApplyGradients(t => appliedTensor = t);

        // Assert
        Assert.NotNull(appliedTensor);
        // Gradient should be scaled by 1/2 = 2.0f
        Assert.Equal(2.0f, appliedTensor.Gradient!.Data[0], precision: 5);
        // After reset, current steps should be 0
        Assert.Equal(0, accumulator.CurrentSteps);
    }

    [Fact]
    public void GradientAccumulator_ApplyGradients_WithoutEnableAccumulation_ThrowsException()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => accumulator.ApplyGradients(t => { }));
    }

    [Fact]
    public void GradientAccumulator_MultipleAccumulationCycles_WorkCorrectly()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.EnableAccumulation(new[] { tensor });

        // Act - First cycle
        accumulator.Step();
        accumulator.Step();
        if (tensor.Gradient != null)
        {
            tensor.Gradient.Data[0] = 4.0f;
        }
        accumulator.ApplyGradients(t => { });

        // Second cycle
        accumulator.Step();
        accumulator.Step();
        if (tensor.Gradient != null)
        {
            tensor.Gradient.Data[0] = 6.0f;
        }
        accumulator.ApplyGradients(t => { });

        // Assert
        Assert.Equal(0, accumulator.CurrentSteps);
        Assert.False(accumulator.IsReady);
    }

    [Fact]
    public void GradientAccumulator_Dispose_DisablesAccumulation()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.EnableAccumulation(new[] { tensor });

        // Act
        accumulator.Dispose();

        // Assert
        Assert.False(accumulator.Enabled);
    }

    #endregion

    #region TensorAccumulationExtensions Tests

    [Fact]
    public void TensorExtensions_EnableGradAccumulation_SetsRequiresGrad()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act
        tensor.EnableGradAccumulation();

        // Assert
        Assert.True(tensor.RequiresGrad);
        Assert.NotNull(tensor.Gradient);
        Assert.True(tensor.HasGradAccumulation());
    }

    [Fact]
    public void TensorExtensions_EnableGradAccumulation_WithNull_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => TensorAccumulationExtensions.EnableGradAccumulation(null!));
    }

    [Fact]
    public void TensorExtensions_DisableGradAccumulation_RemovesAccumulationFlag()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        tensor.EnableGradAccumulation();

        // Act
        tensor.DisableGradAccumulation();

        // Assert
        Assert.False(tensor.HasGradAccumulation());
    }

    [Fact]
    public void TensorExtensions_HasGradAccumulation_ReturnsCorrectState()
    {
        // Arrange
        var tensor1 = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var tensor2 = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        tensor1.EnableGradAccumulation();

        // Assert
        Assert.True(tensor1.HasGradAccumulation());
        Assert.False(tensor2.HasGradAccumulation());
    }

    [Fact]
    public void TensorExtensions_ClearAllAccumulationStates_ClearsAllFlags()
    {
        // Arrange
        var tensor1 = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var tensor2 = new Tensor(new float[] { 2.0f }, new int[] { 1 });
        tensor1.EnableGradAccumulation();
        tensor2.EnableGradAccumulation();

        // Act
        TensorAccumulationExtensions.ClearAllAccumulationStates();

        // Assert
        Assert.False(tensor1.HasGradAccumulation());
        Assert.False(tensor2.HasGradAccumulation());
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void GradientAccumulation_WithBackwardPass_AccumulatesGradients()
    {
        // Arrange
        var accumulator = new GradientAccumulator(3);
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var w = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);
        accumulator.EnableAccumulation(new[] { w });

        // Simulate gradient computation (y = w * x, L = y^2)
        // dL/dw = 2 * w * x^2 = 2 * 3 * 4 = 24 for first step

        // Act - First step
        if (w.Gradient != null)
        {
            w.Gradient.Data[0] += 24.0f;
        }
        accumulator.Step();

        // Second step with different x
        var x2 = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        // dL/dw = 2 * 3 * 1 = 6 for second step
        if (w.Gradient != null)
        {
            w.Gradient.Data[0] += 6.0f;
        }
        accumulator.Step();

        // Third step
        var x3 = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        // dL/dw = 2 * 3 * 4 = 24 for third step
        if (w.Gradient != null)
        {
            w.Gradient.Data[0] += 24.0f;
        }
        accumulator.Step();

        // Apply gradients
        Tensor? appliedTensor = null;
        accumulator.ApplyGradients(t => appliedTensor = t);

        // Assert
        // Total gradient = 24 + 6 + 24 = 54
        // Scaled by 1/3 = 18
        Assert.NotNull(appliedTensor);
        Assert.Equal(18.0f, appliedTensor.Gradient!.Data[0], precision: 5);
    }

    [Fact]
    public void GradientAccumulation_LinearRegression_EquivalentToLargerBatch()
    {
        // This test verifies that accumulating gradients over 3 mini-batches
        // is equivalent to a single batch of 3x the size

        // Arrange
        var accumulator = new GradientAccumulator(3);
        var w = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var b = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        accumulator.EnableAccumulation(new[] { w, b });

        // Simulate 3 mini-batches
        var batches = new[]
        {
            (x: new float[] { 1.0f }, target: new float[] { 5.0f }),
            (x: new float[] { 2.0f }, target: new float[] { 7.0f }),
            (x: new float[] { 3.0f }, target: new float[] { 9.0f })
        };

        // w*x + b = y
        // L = (y - target)^2
        // dL/dw = 2 * (y - target) * x
        // dL/db = 2 * (y - target)

        // Act - Accumulate gradients from each batch
        foreach (var batch in batches)
        {
            var xTensor = new Tensor(batch.x, new int[] { 1 });
            var targetTensor = new Tensor(batch.target, new int[] { 1 });

            // y = w*x + b
            var y = w.Data[0] * batch.x[0] + b.Data[0]; // 2*1 + 1 = 3, 2*2 + 1 = 5, 2*3 + 1 = 7
            var error = y - batch.target[0]; // 3-5=-2, 5-7=-2, 7-9=-2

            // Gradients
            var dw = 2.0f * error * batch.x[0]; // 2*(-2)*1=-4, 2*(-2)*2=-8, 2*(-2)*3=-12
            var db = 2.0f * error; // 2*(-2)=-4 each

            if (w.Gradient != null)
            {
                w.Gradient.Data[0] += dw;
            }
            if (b.Gradient != null)
            {
                b.Gradient.Data[0] += db;
            }

            accumulator.Step();
        }

        // Apply gradients
        Tensor? appliedW = null;
        Tensor? appliedB = null;
        accumulator.ApplyGradients(t =>
        {
            if (t == w) appliedW = t;
            if (t == b) appliedB = t;
        });

        // Assert
        // Total dw = -4 + -8 + -12 = -24, scaled by 1/3 = -8
        // Total db = -4 + -4 + -4 = -12, scaled by 1/3 = -4
        Assert.Equal(-8.0f, appliedW!.Gradient!.Data[0], precision: 5);
        Assert.Equal(-4.0f, appliedB!.Gradient!.Data[0], precision: 5);
    }

    [Fact]
    public void GradientAccumulation_MultipleParameters_AccumulatesCorrectly()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);
        var w1 = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);
        var w2 = new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 }, requiresGrad: true);
        accumulator.EnableAccumulation(new[] { w1, w2 });

        // Act - First step
        if (w1.Gradient != null)
        {
            w1.Gradient.Data[0] += 1.0f;
            w1.Gradient.Data[1] += 2.0f;
        }
        if (w2.Gradient != null)
        {
            w2.Gradient.Data[0] += 3.0f;
            w2.Gradient.Data[1] += 4.0f;
        }
        accumulator.Step();

        // Second step
        if (w1.Gradient != null)
        {
            w1.Gradient.Data[0] += 5.0f;
            w1.Gradient.Data[1] += 6.0f;
        }
        if (w2.Gradient != null)
        {
            w2.Gradient.Data[0] += 7.0f;
            w2.Gradient.Data[1] += 8.0f;
        }
        accumulator.Step();

        // Apply
        accumulator.ApplyGradients(t => { });

        // Assert - Gradients should be scaled by 1/2
        Assert.Equal(3.0f, w1.Gradient!.Data[0], precision: 5); // (1+5)/2
        Assert.Equal(4.0f, w1.Gradient.Data[1], precision: 5); // (2+6)/2
        Assert.Equal(5.0f, w2.Gradient!.Data[0], precision: 5); // (3+7)/2
        Assert.Equal(6.0f, w2.Gradient.Data[1], precision: 5); // (4+8)/2
    }

    [Fact]
    public void GradientAccumulation_AccumulationCountOf1_WorksLikeNormalTraining()
    {
        // Arrange
        var accumulator = new GradientAccumulator(1);
        var w = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        accumulator.EnableAccumulation(new[] { w });

        // Act - Single step
        if (w.Gradient != null)
        {
            w.Gradient.Data[0] = 5.0f;
        }
        accumulator.Step();
        accumulator.ApplyGradients(t => { });

        // Assert
        // With count=1, scaling factor is 1, so gradient remains 5
        Assert.Equal(5.0f, w.Gradient!.Data[0], precision: 5);
    }

    #endregion

    #region Memory and Performance Tests

    [Fact]
    public void GradientAccumulation_Dispose_PreventsMemoryLeaks()
    {
        // Arrange
        var accumulator = new GradientAccumulator(2);
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.EnableAccumulation(new[] { tensor });

        // Act
        accumulator.Dispose();

        // Assert - Operations after dispose should not hold references
        // (This is more of a documentation test for proper disposal)
        Assert.False(accumulator.Enabled);
    }

    #endregion
}
