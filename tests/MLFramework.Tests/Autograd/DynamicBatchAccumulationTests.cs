using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;
using System.Linq;

namespace MLFramework.Tests.Autograd;

/// <summary>
/// Unit tests for dynamic batch gradient accumulation features.
/// Tests variable batch sizes, scaling, scheduling, and validation.
/// </summary>
public class DynamicBatchAccumulationTests
{
    #region DynamicBatchAccumulator Tests

    [Fact]
    public void DynamicBatchAccumulator_Constructor_WithValidTarget_CreatesAccumulator()
    {
        // Arrange & Act
        var accumulator = new DynamicBatchAccumulator(32);

        // Assert
        Assert.Equal(32, accumulator.TargetBatchSize);
        Assert.Equal(0, accumulator.AccumulatedBatchSize);
        Assert.Equal(0, accumulator.CurrentStep);
        Assert.Equal(0, accumulator.AccumulationCount);
        Assert.Null(accumulator.AccumulatedGradient);
    }

    [Fact]
    public void DynamicBatchAccumulator_Constructor_WithInvalidTarget_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new DynamicBatchAccumulator(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DynamicBatchAccumulator(-1));
    }

    [Fact]
    public void DynamicBatchAccumulator_Constructor_Default_InitializesFixedStepMode()
    {
        // Arrange & Act
        var accumulator = new DynamicBatchAccumulator();

        // Assert
        Assert.Equal(0, accumulator.TargetBatchSize);
    }

    [Fact]
    public void DynamicBatchAccumulator_Accumulate_WithValidInputs_AccumulatesGradient()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var gradient = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });

        // Act
        accumulator.Accumulate(gradient, 16);

        // Assert
        Assert.Equal(16, accumulator.AccumulatedBatchSize);
        Assert.Equal(1, accumulator.CurrentStep);
        Assert.Equal(1, accumulator.AccumulationCount);
        Assert.NotNull(accumulator.AccumulatedGradient);
    }

    [Fact]
    public void DynamicBatchAccumulator_Accumulate_WithNullGradient_ThrowsException()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => accumulator.Accumulate(null!, 16));
    }

    [Fact]
    public void DynamicBatchAccumulator_Accumulate_WithInvalidBatchSize_ThrowsException()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var gradient = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => accumulator.Accumulate(gradient, 0));
        Assert.Throws<ArgumentException>(() => accumulator.Accumulate(gradient, -1));
    }

    [Fact]
    public void DynamicBatchAccumulator_Accumulate_MultipleSteps_AccumulatesCorrectly()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var grad1 = new Tensor(new float[] { 2.0f }, new int[] { 1 });
        var grad2 = new Tensor(new float[] { 4.0f }, new int[] { 1 });

        // Act
        accumulator.Accumulate(grad1, 16);
        accumulator.Accumulate(grad2, 16);

        // Assert
        Assert.Equal(32, accumulator.AccumulatedBatchSize);
        Assert.Equal(2, accumulator.CurrentStep);
        Assert.True(accumulator.IsComplete());
    }

    [Fact]
    public void DynamicBatchAccumulator_Accumulate_ShapeMismatch_ThrowsException()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var grad1 = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var grad2 = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });

        // Act & Assert
        accumulator.Accumulate(grad1, 16);
        Assert.Throws<InvalidOperationException>(() => accumulator.Accumulate(grad2, 16));
    }

    [Fact]
    public void DynamicBatchAccumulator_IsComplete_WhenTargetReached_ReturnsTrue()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var gradient = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act
        accumulator.Accumulate(gradient, 16);
        accumulator.Accumulate(gradient, 16);

        // Assert
        Assert.True(accumulator.IsComplete());
    }

    [Fact]
    public void DynamicBatchAccumulator_IsComplete_WhenTargetNotReached_ReturnsFalse()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var gradient = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act
        accumulator.Accumulate(gradient, 16);

        // Assert
        Assert.False(accumulator.IsComplete());
    }

    [Fact]
    public void DynamicBatchAccumulator_GetAccumulatedGradient_ReturnsNormalizedGradient()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var grad1 = new Tensor(new float[] { 32.0f }, new int[] { 1 });
        var grad2 = new Tensor(new float[] { 32.0f }, new int[] { 1 });

        accumulator.Accumulate(grad1, 16);
        accumulator.Accumulate(grad2, 16);

        // Act
        var result = accumulator.GetAccumulatedGradient();

        // Assert
        // Gradient 1: 32/16 = 2.0f, scaled by 16 = 32.0f
        // Gradient 2: 32/16 = 2.0f, scaled by 16 = 32.0f
        // Total: 64.0f, normalized by 32 = 2.0f
        Assert.NotNull(result);
        Assert.Equal(2.0f, result.Data[0], precision: 5);
    }

    [Fact]
    public void DynamicBatchAccumulator_GetAccumulatedGradient_WithoutAccumulation_ThrowsException()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => accumulator.GetAccumulatedGradient());
    }

    [Fact]
    public void DynamicBatchAccumulator_Reset_ClearsAccumulatorState()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var gradient = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.Accumulate(gradient, 16);

        // Act
        accumulator.Reset();

        // Assert
        Assert.Equal(0, accumulator.AccumulatedBatchSize);
        Assert.Equal(0, accumulator.CurrentStep);
        Assert.Equal(0, accumulator.AccumulationCount);
        // Gradient tensor should still exist but be zeroed
        Assert.NotNull(accumulator.AccumulatedGradient);
        Assert.Equal(0.0f, accumulator.AccumulatedGradient.Data[0], precision: 5);
    }

    [Fact]
    public void DynamicBatchAccumulator_GetProgress_ReturnsCorrectValue()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var gradient = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.Accumulate(gradient, 16);

        // Act
        var progress = accumulator.GetProgress();

        // Assert
        Assert.Equal(0.5, progress, precision: 5);
    }

    [Fact]
    public void DynamicBatchAccumulator_GetProgress_WhenComplete_ReturnsOne()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(32);
        var gradient = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        accumulator.Accumulate(gradient, 16);
        accumulator.Accumulate(gradient, 16);

        // Act
        var progress = accumulator.GetProgress();

        // Assert
        Assert.Equal(1.0, progress, precision: 5);
    }

    [Fact]
    public void DynamicBatchAccumulator_FixedStepMode_AlwaysReturnsFalseForIsComplete()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator();
        var gradient = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act
        accumulator.Accumulate(gradient, 16);

        // Assert
        Assert.False(accumulator.IsComplete());
    }

    #endregion

    #region GradientScaling Tests

    [Fact]
    public void GradientScaling_ScaleByBatchSize_WithValidInputs_ScalesGradient()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 2.0f, 4.0f }, new int[] { 2 });

        // Act
        var result = GradientScaling.ScaleByBatchSize(gradient, 16, 32);

        // Assert
        Assert.Equal(1.0f, result.Data[0], precision: 5); // 2.0 * (16/32)
        Assert.Equal(2.0f, result.Data[1], precision: 5); // 4.0 * (16/32)
    }

    [Fact]
    public void GradientScaling_ScaleByBatchSize_WithZeroBatchSize_ReturnsZeros()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        var result = GradientScaling.ScaleByBatchSize(gradient, 0, 32);

        // Assert
        Assert.Equal(0.0f, result.Data[0], precision: 5);
    }

    [Fact]
    public void GradientScaling_AverageAccumulated_WithValidInputs_NormalizesGradient()
    {
        // Arrange
        var accumulated = new Tensor(new float[] { 64.0f, 96.0f }, new int[] { 2 });

        // Act
        var result = GradientScaling.AverageAccumulated(accumulated, 32, 1);

        // Assert
        Assert.Equal(2.0f, result.Data[0], precision: 5); // 64 / 32
        Assert.Equal(3.0f, result.Data[1], precision: 5); // 96 / 32
    }

    [Fact]
    public void GradientScaling_NormalizeBatchGradient_WithValidInputs_NormalizesGradient()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 32.0f, 48.0f }, new int[] { 2 });

        // Act
        var result = GradientScaling.NormalizeBatchGradient(gradient, 16);

        // Assert
        Assert.Equal(2.0f, result.Data[0], precision: 5); // 32 / 16
        Assert.Equal(3.0f, result.Data[1], precision: 5); // 48 / 16
    }

    [Fact]
    public void GradientScaling_WeightedAverage_WithValidInputs_CombinesGradients()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 32.0f }, new int[] { 1 });
        var grad2 = new Tensor(new float[] { 32.0f }, new int[] { 1 });
        var batches = new[] { 16, 16 };

        // Act
        var result = GradientScaling.WeightedAverage(new[] { grad1, grad2 }, batches);

        // Assert
        // Both normalized to 2.0f, weighted by batch size = 32 + 32 = 64, normalized by 32 = 2.0f
        Assert.Equal(2.0f, result.Data[0], precision: 5);
    }

    [Fact]
    public void GradientScaling_WeightedAverage_WithDifferentBatchSizes_HandlesCorrectly()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 32.0f }, new int[] { 1 });
        var grad2 = new Tensor(new float[] { 48.0f }, new int[] { 1 });
        var batches = new[] { 16, 24 };

        // Act
        var result = GradientScaling.WeightedAverage(new[] { grad1, grad2 }, batches);

        // Assert
        // Grad1 normalized: 32/16 = 2.0f, weighted: 2.0f * 16 = 32.0f
        // Grad2 normalized: 48/24 = 2.0f, weighted: 2.0f * 24 = 48.0f
        // Total: 80.0f, normalized by 40 = 2.0f
        Assert.Equal(2.0f, result.Data[0], precision: 5);
    }

    #endregion

    #region VariableBatchScheduler Tests

    [Fact]
    public void VariableBatchScheduler_Constructor_WithValidSchedule_CreatesScheduler()
    {
        // Arrange & Act
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 16, 16 });

        // Assert
        Assert.Equal(3, scheduler.TotalSteps);
        Assert.Equal(48, scheduler.EffectiveBatchSize);
        Assert.Equal(0, scheduler.CurrentStep);
    }

    [Fact]
    public void VariableBatchScheduler_Constructor_WithInvalidSchedule_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new VariableBatchScheduler(null!));
        Assert.Throws<ArgumentException>(() => new VariableBatchScheduler(new List<int>()));
        Assert.Throws<ArgumentException>(() => new VariableBatchScheduler(new List<int> { 0 }));
        Assert.Throws<ArgumentException>(() => new VariableBatchScheduler(new List<int> { -1 }));
    }

    [Fact]
    public void VariableBatchScheduler_Constructor_WithArray_CreatesScheduler()
    {
        // Arrange & Act
        var scheduler = new VariableBatchScheduler(8, 16, 24);

        // Assert
        Assert.Equal(3, scheduler.TotalSteps);
        Assert.Equal(48, scheduler.EffectiveBatchSize);
    }

    [Fact]
    public void VariableBatchScheduler_GetCurrentBatchSize_ReturnsCorrectSize()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 24, 32 });

        // Act
        var size = scheduler.GetCurrentBatchSize();

        // Assert
        Assert.Equal(16, size);
    }

    [Fact]
    public void VariableBatchScheduler_GetCurrentBatchSize_AfterAdvance_ReturnsNextSize()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 24, 32 });
        scheduler.Advance();

        // Act
        var size = scheduler.GetCurrentBatchSize();

        // Assert
        Assert.Equal(24, size);
    }

    [Fact]
    public void VariableBatchScheduler_GetCurrentBatchSize_WhenComplete_ThrowsException()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16 });
        scheduler.Advance();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => scheduler.GetCurrentBatchSize());
    }

    [Fact]
    public void VariableBatchScheduler_GetRemainingSteps_ReturnsCorrectCount()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 16, 16 });

        // Act
        var remaining = scheduler.GetRemainingSteps();

        // Assert
        Assert.Equal(3, remaining);
    }

    [Fact]
    public void VariableBatchScheduler_GetRemainingSteps_AfterAdvance_Decreases()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 16, 16 });
        scheduler.Advance();

        // Act
        var remaining = scheduler.GetRemainingSteps();

        // Assert
        Assert.Equal(2, remaining);
    }

    [Fact]
    public void VariableBatchScheduler_GetEffectiveBatchSize_ReturnsCorrectAverage()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 24, 32 });

        // Act
        var effective = scheduler.GetEffectiveBatchSize(2);

        // Assert
        Assert.Equal(40, effective); // 16 + 24
    }

    [Fact]
    public void VariableBatchScheduler_Advance_IncrementsStep()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 16 });

        // Act
        var advanced = scheduler.Advance();

        // Assert
        Assert.True(advanced);
        Assert.Equal(1, scheduler.CurrentStep);
    }

    [Fact]
    public void VariableBatchScheduler_Advance_WhenComplete_ReturnsFalse()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16 });
        scheduler.Advance();

        // Act
        var advanced = scheduler.Advance();

        // Assert
        Assert.False(advanced);
    }

    [Fact]
    public void VariableBatchScheduler_IsComplete_WhenNotComplete_ReturnsFalse()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 16 });

        // Assert
        Assert.False(scheduler.IsComplete);
    }

    [Fact]
    public void VariableBatchScheduler_IsComplete_WhenComplete_ReturnsTrue()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16 });
        scheduler.Advance();

        // Assert
        Assert.True(scheduler.IsComplete);
    }

    [Fact]
    public void VariableBatchScheduler_Reset_ClearsProgress()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 16 });
        scheduler.Advance();
        scheduler.Advance();

        // Act
        scheduler.Reset();

        // Assert
        Assert.Equal(0, scheduler.CurrentStep);
        Assert.False(scheduler.IsComplete);
    }

    [Fact]
    public void VariableBatchScheduler_GetProgress_ReturnsCorrectValue()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 16, 16 });
        scheduler.Advance();

        // Act
        var progress = scheduler.GetProgress();

        // Assert
        Assert.Equal(0.333, progress, precision: 3);
    }

    [Fact]
    public void VariableBatchScheduler_GetAccumulatedBatchSize_ReturnsCorrectSum()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 24, 32 });
        scheduler.Advance();
        scheduler.Advance();

        // Act
        var accumulated = scheduler.GetAccumulatedBatchSize();

        // Assert
        Assert.Equal(40, accumulated); // 16 + 24
    }

    #endregion

    #region AccumulationBufferDynamic Tests

    [Fact]
    public void AccumulationBufferDynamic_Constructor_WithValidInputs_CreatesBuffer()
    {
        // Arrange & Act
        var shape = new SymbolicShape(3);
        var buffer = new AccumulationBufferDynamic(shape, 100);

        // Assert
        Assert.Equal(100, buffer.MaxSize);
        Assert.Equal(0, buffer.CurrentSize);
        Assert.Null(buffer.Buffer);
    }

    [Fact]
    public void AccumulationBufferDynamic_Constructor_WithConcreteShape_CreatesInitializedBuffer()
    {
        // Arrange & Act
        var buffer = new AccumulationBufferDynamic(new[] { 10 }, 100);

        // Assert
        Assert.Equal(100, buffer.MaxSize);
        Assert.Equal(0, buffer.CurrentSize);
        Assert.NotNull(buffer.Buffer);
        Assert.Equal(10, buffer.Buffer.Data.Length);
    }

    [Fact]
    public void AccumulationBufferDynamic_Accumulate_WithValidInputs_AccumulatesGradient()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 10 }, 100);
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });

        // Act
        buffer.Accumulate(gradient, 0, 3);

        // Assert
        Assert.Equal(3, buffer.CurrentSize);
        Assert.Equal(1.0f, buffer.Buffer.Data[0], precision: 5);
        Assert.Equal(2.0f, buffer.Buffer.Data[1], precision: 5);
        Assert.Equal(3.0f, buffer.Buffer.Data[2], precision: 5);
    }

    [Fact]
    public void AccumulationBufferDynamic_Accumulate_MultipleAccumulations_AddsCorrectly()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 10 }, 100);
        var grad1 = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var grad2 = new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 });

        // Act
        buffer.Accumulate(grad1, 0, 2);
        buffer.Accumulate(grad2, 0, 2);

        // Assert
        Assert.Equal(4.0f, buffer.Buffer.Data[0], precision: 5); // 1 + 3
        Assert.Equal(6.0f, buffer.Buffer.Data[1], precision: 5); // 2 + 4
    }

    [Fact]
    public void AccumulationBufferDynamic_Accumulate_WithInvalidIndices_ThrowsException()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 10 }, 100);
        var gradient = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => buffer.Accumulate(gradient, -1, 1));
        Assert.Throws<InvalidOperationException>(() => buffer.Accumulate(gradient, 10, 1));
        Assert.Throws<InvalidOperationException>(() => buffer.Accumulate(gradient, 0, 11));
    }

    [Fact]
    public void AccumulationBufferDynamic_Resize_WithValidSize_ResizesBuffer()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 10 }, 100);
        buffer.Buffer.Data[0] = 5.0f;

        // Act
        buffer.Resize(5);

        // Assert
        Assert.Equal(5, buffer.Buffer.Data.Length);
        Assert.Equal(5.0f, buffer.Buffer.Data[0], precision: 5);
    }

    [Fact]
    public void AccumulationBufferDynamic_Resize_WithInvalidSize_ThrowsException()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 10 }, 100);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => buffer.Resize(0));
        Assert.Throws<ArgumentException>(() => buffer.Resize(-1));
        Assert.Throws<ArgumentException>(() => buffer.Resize(150));
    }

    [Fact]
    public void AccumulationBufferDynamic_GetSlice_ReturnsCorrectSlice()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 10 }, 100);
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }, new int[] { 5 });
        buffer.Accumulate(gradient, 0, 5);

        // Act
        var slice = buffer.GetSlice(1, 3);

        // Assert
        Assert.Equal(3, slice.Data.Length);
        Assert.Equal(2.0f, slice.Data[0], precision: 5);
        Assert.Equal(3.0f, slice.Data[1], precision: 5);
        Assert.Equal(4.0f, slice.Data[2], precision: 5);
    }

    [Fact]
    public void AccumulationBufferDynamic_GetFull_ReturnsCopyOfBuffer()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 5 }, 100);
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }, new int[] { 5 });
        buffer.Accumulate(gradient, 0, 5);

        // Act
        var full = buffer.GetFull();

        // Assert
        Assert.Equal(buffer.Buffer.Data.Length, full.Data.Length);
        for (int i = 0; i < buffer.Buffer.Data.Length; i++)
        {
            Assert.Equal(buffer.Buffer.Data[i], full.Data[i], precision: 5);
        }
    }

    [Fact]
    public void AccumulationBufferDynamic_Clear_ClearsBuffer()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 5 }, 100);
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }, new int[] { 5 });
        buffer.Accumulate(gradient, 0, 5);

        // Act
        buffer.Clear();

        // Assert
        Assert.Equal(0, buffer.CurrentSize);
        Assert.All(buffer.Buffer.Data, val => Assert.Equal(0.0f, val, precision: 5));
    }

    [Fact]
    public void AccumulationBufferDynamic_IsFull_ReturnsCorrectValue()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 10 }, 10);

        // Assert
        Assert.False(buffer.IsFull);

        // Act
        var gradient = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        buffer.Accumulate(gradient, 8, 2);

        // Assert
        Assert.True(buffer.IsFull);
    }

    [Fact]
    public void AccumulationBufferDynamic_AvailableCapacity_ReturnsCorrectValue()
    {
        // Arrange
        var buffer = new AccumulationBufferDynamic(new[] { 10 }, 100);
        var gradient = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        buffer.Accumulate(gradient, 0, 2);

        // Act
        var capacity = buffer.AvailableCapacity;

        // Assert
        Assert.Equal(98, capacity);
    }

    #endregion

    #region GradientAccumulationValidator Tests

    [Fact]
    public void GradientAccumulationValidator_ValidateAccumulation_WithValidInputs_ReturnsTrue()
    {
        // Arrange
        var gradients = new List<Tensor>
        {
            new Tensor(new float[] { 1.0f }, new int[] { 1 }),
            new Tensor(new float[] { 2.0f }, new int[] { 1 })
        };
        var batchSizes = new List<int> { 16, 16 };

        // Act
        var result = GradientAccumulationValidator.ValidateAccumulation(gradients, batchSizes);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void GradientAccumulationValidator_ValidateAccumulation_WithNullGradients_ThrowsException()
    {
        // Arrange
        var batchSizes = new List<int> { 16, 16 };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => GradientAccumulationValidator.ValidateAccumulation(null!, batchSizes));
    }

    [Fact]
    public void GradientAccumulationValidator_ValidateAccumulation_WithMismatchedLengths_ReturnsFalse()
    {
        // Arrange
        var gradients = new List<Tensor>
        {
            new Tensor(new float[] { 1.0f }, new int[] { 1 })
        };
        var batchSizes = new List<int> { 16, 16 };

        // Act
        var result = GradientAccumulationValidator.ValidateAccumulation(gradients, batchSizes);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GradientAccumulationValidator_ValidateAccumulation_WithInvalidBatchSize_ReturnsFalse()
    {
        // Arrange
        var gradients = new List<Tensor>
        {
            new Tensor(new float[] { 1.0f }, new int[] { 1 })
        };
        var batchSizes = new List<int> { 0 };

        // Act
        var result = GradientAccumulationValidator.ValidateAccumulation(gradients, batchSizes);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GradientAccumulationValidator_CheckShapeCompatibility_WithSameShape_ReturnsTrue()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var grad2 = new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 });

        // Act
        var result = GradientAccumulationValidator.CheckShapeCompatibility(grad1, grad2);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void GradientAccumulationValidator_CheckShapeCompatibility_WithDifferentShape_ReturnsFalse()
    {
        // Arrange
        var grad1 = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var grad2 = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });

        // Act
        var result = GradientAccumulationValidator.CheckShapeCompatibility(grad1, grad2);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GradientAccumulationValidator_ValidateAccumulatedShape_WithValidInput_ReturnsTrue()
    {
        // Arrange
        var accumulated = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 });
        var components = new List<Tensor>
        {
            new Tensor(new float[] { 1.0f }, new int[] { 1 }),
            new Tensor(new float[] { 2.0f }, new int[] { 1 })
        };

        // Act
        var result = GradientAccumulationValidator.ValidateAccumulatedShape(accumulated, components);

        // Assert - Different shapes, so false
        Assert.False(result);
    }

    [Fact]
    public void GradientAccumulationValidator_IsValidBatchSize_WithValidSize_ReturnsTrue()
    {
        // Act
        var result = GradientAccumulationValidator.IsValidBatchSize(32);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void GradientAccumulationValidator_IsValidBatchSize_WithInvalidSize_ReturnsFalse()
    {
        // Act
        var result1 = GradientAccumulationValidator.IsValidBatchSize(0);
        var result2 = GradientAccumulationValidator.IsValidBatchSize(-1);

        // Assert
        Assert.False(result1);
        Assert.False(result2);
    }

    [Fact]
    public void GradientAccumulationValidator_CheckOverflow_WithOverflow_ReturnsTrue()
    {
        // Act
        var result = GradientAccumulationValidator.CheckOverflow(16, 20, 32);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void GradientAccumulationValidator_CheckOverflow_WithoutOverflow_ReturnsFalse()
    {
        // Act
        var result = GradientAccumulationValidator.CheckOverflow(16, 8, 32);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GradientAccumulationValidator_ValidateGradientValues_WithValidValues_ReturnsTrue()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });

        // Act
        var result = GradientAccumulationValidator.ValidateGradientValues(gradient);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void GradientAccumulationValidator_ValidateGradientValues_WithNaN_ReturnsFalse()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, float.NaN, 3.0f }, new int[] { 3 });

        // Act
        var result = GradientAccumulationValidator.ValidateGradientValues(gradient);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GradientAccumulationValidator_ValidateGradientValues_WithInfinity_ReturnsFalse()
    {
        // Arrange
        var gradient = new Tensor(new float[] { 1.0f, float.PositiveInfinity, 3.0f }, new int[] { 3 });

        // Act
        var result = GradientAccumulationValidator.ValidateGradientValues(gradient);

        // Assert
        Assert.False(result);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void Integration_VariableBatchAccumulation_WithScheduler_WorksCorrectly()
    {
        // Arrange
        var scheduler = new VariableBatchScheduler(new List<int> { 16, 16 });
        var accumulator = new DynamicBatchAccumulator(32);
        var grad1 = new Tensor(new float[] { 32.0f }, new int[] { 1 });
        var grad2 = new Tensor(new float[] { 32.0f }, new int[] { 1 });

        // Act - Use scheduler to guide accumulation
        var batchSize1 = scheduler.GetCurrentBatchSize();
        accumulator.Accumulate(grad1, batchSize1);
        scheduler.Advance();

        var batchSize2 = scheduler.GetCurrentBatchSize();
        accumulator.Accumulate(grad2, batchSize2);
        scheduler.Advance();

        var accumulated = accumulator.GetAccumulatedGradient();

        // Assert
        Assert.True(accumulator.IsComplete());
        Assert.True(scheduler.IsComplete);
        Assert.Equal(2.0f, accumulated.Data[0], precision: 5);
    }

    [Fact]
    public void Integration_AdaptiveAccumulation_WithVaryingBatchSizes_NormalizesCorrectly()
    {
        // Arrange
        var accumulator = new DynamicBatchAccumulator(40);
        var gradients = new[]
        {
            new Tensor(new float[] { 16.0f }, new int[] { 1 }),
            new Tensor(new float[] { 32.0f }, new int[] { 1 }),
            new Tensor(new float[] { 48.0f }, new int[] { 1 })
        };
        var batchSizes = new[] { 8, 16, 16 };

        // Act - Accumulate with varying batch sizes
        accumulator.Accumulate(gradients[0], batchSizes[0]);
        accumulator.Accumulate(gradients[1], batchSizes[1]);
        accumulator.Accumulate(gradients[2], batchSizes[2]);

        var accumulated = accumulator.GetAccumulatedGradient();

        // Assert
        // Grad1: 16/8 = 2.0f, scaled by 8 = 16.0f
        // Grad2: 32/16 = 2.0f, scaled by 16 = 32.0f
        // Grad3: 48/16 = 3.0f, scaled by 16 = 48.0f
        // Total: 96.0f, normalized by 40 = 2.4f
        Assert.Equal(2.4f, accumulated.Data[0], precision: 5);
    }

    #endregion
}
