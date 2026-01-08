using System;
using System.Linq;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Unit tests for GradientTape higher-order derivative support.
/// Tests cover nested differentiation, gradient retention, and memory management.
/// </summary>
public class GradientTapeTests
{
    #region Basic Functionality Tests

    [Fact]
    public void EnableHigherOrderTracking_CreatesTapeWithHigherOrderSupport()
    {
        // Arrange & Act
        using var tape = GradientTape.EnableHigherOrderTracking();

        // Assert
        Assert.True(tape.HigherOrderEnabled);
        Assert.Equal(GradientRetentionPolicy.Keep, tape.RetentionPolicy);
        Assert.Equal(0, tape.Depth);
    }

    [Fact]
    public void Create_WithoutHigherOrder_CreatesStandardTape()
    {
        // Arrange & Act
        using var tape = GradientTape.Create(higherOrderEnabled: false);

        // Assert
        Assert.False(tape.HigherOrderEnabled);
        Assert.Equal(GradientRetentionPolicy.Discard, tape.RetentionPolicy);
    }

    [Fact]
    public void Create_WithCustomRetentionPolicy_SetsPolicyCorrectly()
    {
        // Arrange & Act
        using var tape = GradientTape.Create(
            higherOrderEnabled: true,
            retentionPolicy: GradientRetentionPolicy.Selective
        );

        // Assert
        Assert.True(tape.HigherOrderEnabled);
        Assert.Equal(GradientRetentionPolicy.Selective, tape.RetentionPolicy);
    }

    [Fact]
    public void Record_WithoutHigherOrderEnabled_ThrowsException()
    {
        // Arrange
        using var tape = GradientTape.Create(higherOrderEnabled: false);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => tape.Record());
    }

    [Fact]
    public void Record_WithHigherOrderEnabled_CreatesNestedTape()
    {
        // Arrange
        using var parentTape = GradientTape.EnableHigherOrderTracking();

        // Act
        using var childTape = parentTape.Record();

        // Assert
        Assert.NotNull(childTape);
        Assert.True(childTape.HigherOrderEnabled);
        Assert.Equal(1, childTape.Depth);
        Assert.Equal(0, parentTape.Depth);
    }

    [Fact]
    public void Watch_AddsTensorToWatchedList()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var tensor = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);

        // Act
        tape.Watch(tensor);

        // Assert
        Assert.Equal(1, tape.WatchedTensorCount);
    }

    [Fact]
    public void Watch_NullTensor_ThrowsException()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => tape.Watch(null!));
    }

    [Fact]
    public void Watch_TensorWithoutGrad_ThrowsException()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var tensor = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: false);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => tape.Watch(tensor));
    }

    [Fact]
    public void Watch_SameTensorMultipleTimes_OnlyWatchesOnce()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var tensor = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);

        // Act
        tape.Watch(tensor);
        tape.Watch(tensor);
        tape.Watch(tensor);

        // Assert
        Assert.Equal(1, tape.WatchedTensorCount);
    }

    #endregion

    #region Nested Differentiation Tests

    [Fact]
    public void NestedTapes_ComputeThirdOrderDerivativeCorrectly()
    {
        // Arrange
        // f(x) = x^4, f'(x) = 4x^3, f''(x) = 12x^2, f'''(x) = 24x
        // At x = 2: f'''(2) = 48

        using var outerTape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        outerTape.Watch(x);

        // First backward pass (first derivative)
        var y = x.Pow(4);
        var grad1 = outerTape.Gradient(y)[x];
        Assert.NotNull(grad1);

        // Second backward pass (second derivative) using nested tape
        using var innerTape1 = outerTape.Record();
        innerTape1.Watch(x);
        var grad2 = innerTape1.Gradient(grad1)[x];
        Assert.NotNull(grad2);

        // Third backward pass (third derivative) using another nested tape
        using var innerTape2 = innerTape1.Record();
        innerTape2.Watch(x);
        var grad3 = innerTape2.Gradient(grad2)[x];
        Assert.NotNull(grad3);

        // Assert: f'''(2) = 48
        var expectedValue = 24.0f * 2.0f; // 24x at x=2
        Assert.InRange(
            Math.Abs(grad3.Data[0] - expectedValue),
            0.0,
            0.1f // Allow for floating point errors in simple approximation
        );
    }

    [Fact]
    public void NestedTapes_DepthIsCorrectlyTracked()
    {
        // Arrange
        using var tape0 = GradientTape.EnableHigherOrderTracking();
        using var tape1 = tape0.Record();
        using var tape2 = tape1.Record();
        using var tape3 = tape2.Record();

        // Assert
        Assert.Equal(0, tape0.Depth);
        Assert.Equal(1, tape1.Depth);
        Assert.Equal(2, tape2.Depth);
        Assert.Equal(3, tape3.Depth);
    }

    [Fact]
    public void GradientRetention_KeepsGradientsWhenRequired()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        tape.Watch(x);

        // Act
        var y = x.Pow(2);
        var gradients = tape.Gradient(y);
        var gradientHistory = tape.GetGradientHistory(x);

        // Assert
        Assert.NotNull(gradients);
        Assert.True(gradients.ContainsKey(x));
        Assert.NotNull(gradientHistory);
        Assert.NotNull(gradientHistory.Gradient);
    }

    [Fact]
    public void GradientRetention_DiscardsGradientsWhenNotRequired()
    {
        // Arrange
        using var tape = GradientTape.Create(higherOrderEnabled: false, retentionPolicy: GradientRetentionPolicy.Discard);
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        tape.Watch(x);

        // Act
        var y = x.Pow(2);
        var gradients = tape.Gradient(y);
        var gradientHistory = tape.GetGradientHistory(x);

        // Assert
        Assert.NotNull(gradients);
        Assert.True(gradients.ContainsKey(x));
        Assert.Null(gradientHistory);
    }

    #endregion

    #region Memory Management Tests

    [Fact]
    public void MemoryUsage_NoLeaksWithNestedTapes()
    {
        // Arrange
        var initialDepth = GradientTape.GetGlobalDepth();

        // Act
        using (var tape0 = GradientTape.EnableHigherOrderTracking())
        {
            using (var tape1 = tape0.Record())
            {
                using (var tape2 = tape1.Record())
                {
                    Assert.Equal(3, GradientTape.GetGlobalDepth());
                }
                Assert.Equal(2, GradientTape.GetGlobalDepth());
            }
            Assert.Equal(1, GradientTape.GetGlobalDepth());
        }

        // Assert
        Assert.Equal(initialDepth, GradientTape.GetGlobalDepth());
    }

    [Fact]
    public void Dispose_ClearsWatchedTensorsAndGradients()
    {
        // Arrange
        var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);
        tape.Watch(x);

        // Act
        tape.Dispose();

        // Assert - accessing disposed tape should behave gracefully
        Assert.Equal(0, tape.WatchedTensorCount);
    }

    [Fact]
    public void Reset_ClearsWatchedTensorsAndGradientHistory()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        tape.Watch(x);
        var y = x.Pow(2);
        tape.Gradient(y);

        Assert.Equal(1, tape.WatchedTensorCount);
        Assert.NotNull(tape.GetGradientHistory(x));

        // Act
        tape.Reset();

        // Assert
        Assert.Equal(0, tape.WatchedTensorCount);
        Assert.Null(tape.GetGradientHistory(x));
    }

    #endregion

    #region Graph Merging Tests

    [Fact]
    public void GraphMerging_CombinesNestedTapesCorrectly()
    {
        // Arrange
        using var parentTape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        parentTape.Watch(x);

        // Act - Create nested tape and compute gradient
        using (var childTape = parentTape.Record())
        {
            childTape.Watch(x);
            var y = x.Pow(2);
            childTape.Gradient(y);

            // Child tape should have gradient history
            Assert.NotNull(childTape.GetGradientHistory(x));
        }

        // Assert - After child disposal, parent should have merged gradient history
        var parentGradientHistory = parentTape.GetGradientHistory(x);
        Assert.NotNull(parentGradientHistory);
    }

    [Fact]
    public void MergeToParent_WithoutParent_ThrowsException()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        tape.Watch(x);
        var y = x.Pow(2);
        tape.Gradient(y);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => tape.MergeToParent());
    }

    #endregion

    #region Gradient Computation Tests

    [Fact]
    public void Gradient_ComputesCorrectGradientsForQuadratic()
    {
        // Arrange: f(x) = x^2, f'(x) = 2x
        using var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);
        tape.Watch(x);

        // Act
        var y = x.Pow(2);
        var gradients = tape.Gradient(y);
        var grad = gradients[x];

        // Assert
        Assert.NotNull(grad);
        Assert.Equal(1, grad.Size);
        Assert.Equal(6.0f, grad.Data[0], precision: 4); // 2 * 3 = 6
    }

    [Fact]
    public void Gradient_MultipleWatchedTensors_ComputesAllGradients()
    {
        // Arrange: f(x, y) = x^2 + y^2
        using var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        var y = new Tensor(new float[] { 3.0f }, new int[] { 1 }, requiresGrad: true);
        tape.Watch(x);
        tape.Watch(y);

        // Act
        var z = x.Pow(2) + y.Pow(2);
        var gradients = tape.Gradient(z);

        // Assert
        Assert.Equal(2, gradients.Count);
        Assert.Equal(4.0f, gradients[x].Data[0], precision: 4); // df/dx = 2x = 4
        Assert.Equal(6.0f, gradients[y].Data[0], precision: 4); // df/dy = 2y = 6
    }

    [Fact]
    public void Gradient_NonScalarLossWithGradientOutput_ComputesCorrectly()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 2.0f }, new int[] { 1 }, requiresGrad: true);
        tape.Watch(x);

        // Act - Non-scalar loss with custom gradient output
        var y = x.Pow(2);
        var gradientOutput = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var gradients = tape.Gradient(y, gradientOutput);
        var grad = gradients[x];

        // Assert
        Assert.NotNull(grad);
        Assert.Equal(4.0f, grad.Data[0], precision: 4); // d(x^2)/dx = 2x = 4, scaled by 1.0
    }

    [Fact]
    public void Gradient_NonScalarLossWithoutGradientOutput_ThrowsException()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);
        tape.Watch(x);

        // Act
        var y = x + 1.0f;

        // Assert
        Assert.Throws<ArgumentException>(() => tape.Gradient(y));
    }

    [Fact]
    public void Gradient_NullLoss_ThrowsException()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => tape.Gradient(null!));
    }

    #endregion

    #region Tensor Extension Tests

    [Fact]
    public void IsDifferentiable_WithGrad_ReturnsTrue()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);

        // Act
        var isDifferentiable = tensor.IsDifferentiable();

        // Assert
        Assert.True(isDifferentiable);
    }

    [Fact]
    public void IsDifferentiable_WithoutGrad_ReturnsFalse()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: false);

        // Act
        var isDifferentiable = tensor.IsDifferentiable();

        // Assert
        Assert.False(isDifferentiable);
    }

    [Fact]
    public void IsDifferentiable_NullTensor_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => MLFramework.Autograd.TensorExtensions.IsDifferentiable(null!));
    }

    [Fact]
    public void Detach_CreatesCopyWithoutGrad()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);

        // Act
        var detached = tensor.Detach();

        // Assert
        Assert.NotNull(detached);
        Assert.False(detached.RequiresGrad);
        Assert.Null(detached.Gradient);
        Assert.Equal(tensor.Size, detached.Size);
        Assert.Equal(tensor.Shape, detached.Shape);
    }

    [Fact]
    public void RequiresGrad_Extension_CreatesCopyWithGrad()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: false);

        // Act
        var withGrad = MLFramework.Autograd.TensorExtensions.RequiresGrad(tensor);

        // Assert
        Assert.NotNull(withGrad);
        Assert.True(withGrad.RequiresGrad);
        Assert.NotNull(withGrad.Gradient);
    }

    [Fact]
    public void Clone_WithGradOption_CreatesCorrectCopy()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: false);

        // Act
        var copyWithGrad = tensor.Clone(requiresGrad: true);
        var copyWithoutGrad = tensor.Clone(requiresGrad: false);

        // Assert
        Assert.True(copyWithGrad.RequiresGrad);
        Assert.False(copyWithoutGrad.RequiresGrad);
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void Gradient_EmptyWatchedList_ReturnsEmptyDictionary()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        // Don't watch x

        // Act
        var y = x.Pow(2);
        var gradients = tape.Gradient(y);

        // Assert
        Assert.Empty(gradients);
    }

    [Fact]
    public void GetGradientHistory_NonWatchedTensor_ReturnsNull()
    {
        // Arrange
        using var tape = GradientTape.EnableHigherOrderTracking();
        var x = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: true);
        // Don't watch x

        // Act
        var history = tape.GetGradientHistory(x);

        // Assert
        Assert.Null(history);
    }

    [Fact]
    public void SetRetainGradient_WithNullTensor_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            MLFramework.Autograd.TensorExtensions.SetRetainGradient(null!, true));
    }

    #endregion
}
