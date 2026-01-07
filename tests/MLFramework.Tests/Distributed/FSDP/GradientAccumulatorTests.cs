using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed.FSDP;

namespace MLFramework.Tests.Distributed.FSDP
{
    /// <summary>
    /// Unit tests for GradientAccumulator.
    /// </summary>
    public class GradientAccumulatorTests
    {
        [Fact]
        public void Constructor_WithPositiveAccumulationSteps_CreatesAccumulator()
        {
            // Arrange & Act
            var accumulator = new GradientAccumulator(5);

            // Assert
            Assert.NotNull(accumulator);
            Assert.Equal(5, accumulator.AccumulationSteps);
        }

        [Fact]
        public void Constructor_WithAccumulationStepsOfOne_CreatesAccumulator()
        {
            // Arrange & Act
            var accumulator = new GradientAccumulator(1);

            // Assert
            Assert.NotNull(accumulator);
            Assert.Equal(1, accumulator.AccumulationSteps);
        }

        [Fact]
        public void Constructor_WithZeroAccumulationSteps_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new GradientAccumulator(0));
        }

        [Fact]
        public void Constructor_WithNegativeAccumulationSteps_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new GradientAccumulator(-1));
        }

        [Fact]
        public void AddGradients_WithNullDictionary_ThrowsArgumentNullException()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => accumulator.AddGradients(null!));
        }

        [Fact]
        public void AddGradients_WithEmptyDictionary_DoesNotThrow()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var emptyGradients = new Dictionary<string, Tensor>();

            // Act & Assert - Should not throw
            accumulator.AddGradients(emptyGradients);
        }

        [Fact]
        public void AddGradients_WithSingleGradient_AddsToAccumulator()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients = new Dictionary<string, Tensor>();
            var gradData = new float[] { 1.0f, 2.0f, 3.0f };
            gradients["param0"] = new Tensor(gradData, new[] { 3 }, false, DataType.Float32);

            // Act
            accumulator.AddGradients(gradients);

            // Assert
            Assert.Equal(1, accumulator.GetAccumulatedCount("param0"));
        }

        [Fact]
        public void AddGradients_WithMultipleGradients_AddsAllToAccumulator()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients = new Dictionary<string, Tensor>();
            gradients["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
            gradients["param1"] = new Tensor(new float[] { 3.0f, 4.0f, 5.0f }, new[] { 3 }, false, DataType.Float32);

            // Act
            accumulator.AddGradients(gradients);

            // Assert
            Assert.Equal(1, accumulator.GetAccumulatedCount("param0"));
            Assert.Equal(1, accumulator.GetAccumulatedCount("param1"));
        }

        [Fact]
        public void AddGradients_MultipleTimes_AccumulatesCorrectly()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            // First batch
            var gradients1 = new Dictionary<string, Tensor>();
            gradients1["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients1);

            // Second batch
            var gradients2 = new Dictionary<string, Tensor>();
            gradients2["param0"] = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients2);

            // Third batch
            var gradients3 = new Dictionary<string, Tensor>();
            gradients3["param0"] = new Tensor(new float[] { 5.0f, 6.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients3);

            // Assert
            Assert.Equal(3, accumulator.GetAccumulatedCount("param0"));
        }

        [Fact]
        public void IsComplete_WithNoGradients_ReturnsFalse()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            // Act
            var isComplete = accumulator.IsComplete;

            // Assert
            Assert.False(isComplete);
        }

        [Fact]
        public void IsComplete_WithInsufficientGradients_ReturnsFalse()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients = new Dictionary<string, Tensor>();
            gradients["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients);

            // Act
            var isComplete = accumulator.IsComplete;

            // Assert
            Assert.False(isComplete);
        }

        [Fact]
        public void IsComplete_WithExactAccumulationSteps_ReturnsTrue()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            for (int i = 0; i < 3; i++)
            {
                var gradients = new Dictionary<string, Tensor>();
                gradients["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
                accumulator.AddGradients(gradients);
            }

            // Act
            var isComplete = accumulator.IsComplete;

            // Assert
            Assert.True(isComplete);
        }

        [Fact]
        public void IsComplete_WithMoreThanAccumulationSteps_ReturnsTrue()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            for (int i = 0; i < 5; i++)
            {
                var gradients = new Dictionary<string, Tensor>();
                gradients["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
                accumulator.AddGradients(gradients);
            }

            // Act
            var isComplete = accumulator.IsComplete;

            // Assert
            Assert.True(isComplete);
        }

        [Fact]
        public void GetAndClearGradients_WithAccumulatedGradients_ReturnsSummedGradients()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            // Add first gradient
            var gradients1 = new Dictionary<string, Tensor>();
            gradients1["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients1);

            // Add second gradient
            var gradients2 = new Dictionary<string, Tensor>();
            gradients2["param0"] = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients2);

            // Act
            var summed = accumulator.GetAndClearGradients();

            // Assert
            Assert.Single(summed);
            Assert.True(summed.ContainsKey("param0"));
            Assert.Equal(4.0f, summed["param0"].Data[0]); // 1.0 + 3.0
            Assert.Equal(6.0f, summed["param0"].Data[1]); // 2.0 + 4.0
        }

        [Fact]
        public void GetAndClearGradients_AfterCall_ClearsAccumulator()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients = new Dictionary<string, Tensor>();
            gradients["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients);

            // Act
            accumulator.GetAndClearGradients();

            // Assert
            Assert.False(accumulator.IsComplete);
            Assert.Equal(0, accumulator.GetAccumulatedCount("param0"));
        }

        [Fact]
        public void GetGradients_WithAccumulatedGradients_ReturnsSummedGradients()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            // Add first gradient
            var gradients1 = new Dictionary<string, Tensor>();
            gradients1["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients1);

            // Add second gradient
            var gradients2 = new Dictionary<string, Tensor>();
            gradients2["param0"] = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients2);

            // Act
            var summed = accumulator.GetGradients();

            // Assert
            Assert.Single(summed);
            Assert.True(summed.ContainsKey("param0"));
            Assert.Equal(4.0f, summed["param0"].Data[0]); // 1.0 + 3.0
            Assert.Equal(6.0f, summed["param0"].Data[1]); // 2.0 + 4.0
        }

        [Fact]
        public void GetGradients_AfterCall_DoesNotClearAccumulator()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients = new Dictionary<string, Tensor>();
            gradients["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients);

            // Act
            accumulator.GetGradients();

            // Assert
            Assert.Equal(1, accumulator.GetAccumulatedCount("param0"));
        }

        [Fact]
        public void GetAccumulatedCount_WithNonexistentParameter_ReturnsZero()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            // Act
            var count = accumulator.GetAccumulatedCount("nonexistent");

            // Assert
            Assert.Equal(0, count);
        }

        [Fact]
        public void GetTotalAccumulatedCount_WithMultipleParameters_ReturnsTotal()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients1 = new Dictionary<string, Tensor>();
            gradients1["param0"] = new Tensor(new float[] { 1.0f }, new[] { 1 }, false, DataType.Float32);
            gradients1["param1"] = new Tensor(new float[] { 2.0f }, new[] { 1 }, false, DataType.Float32);
            accumulator.AddGradients(gradients1);

            var gradients2 = new Dictionary<string, Tensor>();
            gradients2["param0"] = new Tensor(new float[] { 3.0f }, new[] { 1 }, false, DataType.Float32);
            accumulator.AddGradients(gradients2);

            // Act
            var total = accumulator.GetTotalAccumulatedCount();

            // Assert
            Assert.Equal(3, total); // 2 for param0 + 1 for param1
        }

        [Fact]
        public void GetTrackedParameterCount_WithNoParameters_ReturnsZero()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            // Act
            var count = accumulator.GetTrackedParameterCount();

            // Assert
            Assert.Equal(0, count);
        }

        [Fact]
        public void GetTrackedParameterCount_WithMultipleParameters_ReturnsCount()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients = new Dictionary<string, Tensor>();
            gradients["param0"] = new Tensor(new float[] { 1.0f }, new[] { 1 }, false, DataType.Float32);
            gradients["param1"] = new Tensor(new float[] { 2.0f }, new[] { 1 }, false, DataType.Float32);
            gradients["param2"] = new Tensor(new float[] { 3.0f }, new[] { 1 }, false, DataType.Float32);
            accumulator.AddGradients(gradients);

            // Act
            var count = accumulator.GetTrackedParameterCount();

            // Assert
            Assert.Equal(3, count);
        }

        [Fact]
        public void GetTrackedParameterNames_WithMultipleParameters_ReturnsAllNames()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients = new Dictionary<string, Tensor>();
            gradients["param0"] = new Tensor(new float[] { 1.0f }, new[] { 1 }, false, DataType.Float32);
            gradients["param1"] = new Tensor(new float[] { 2.0f }, new[] { 1 }, false, DataType.Float32);
            gradients["param2"] = new Tensor(new float[] { 3.0f }, new[] { 1 }, false, DataType.Float32);
            accumulator.AddGradients(gradients);

            // Act
            var names = accumulator.GetTrackedParameterNames().ToList();

            // Assert
            Assert.Equal(3, names.Count);
            Assert.Contains("param0", names);
            Assert.Contains("param1", names);
            Assert.Contains("param2", names);
        }

        [Fact]
        public void Reset_WithAccumulatedGradients_ClearsAllGradients()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients = new Dictionary<string, Tensor>();
            gradients["param0"] = new Tensor(new float[] { 1.0f }, new[] { 1 }, false, DataType.Float32);
            gradients["param1"] = new Tensor(new float[] { 2.0f }, new[] { 1 }, false, DataType.Float32);
            accumulator.AddGradients(gradients);

            // Act
            accumulator.Reset();

            // Assert
            Assert.Equal(0, accumulator.GetTrackedParameterCount());
            Assert.Equal(0, accumulator.GetTotalAccumulatedCount());
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);
            var gradients = new Dictionary<string, Tensor>();
            gradients["param0"] = new Tensor(new float[] { 1.0f }, new[] { 1 }, false, DataType.Float32);
            accumulator.AddGradients(gradients);

            // Act & Assert - Should not throw
            accumulator.Dispose();
            accumulator.Dispose();
        }

        [Fact]
        public void GetGradients_WithNoGradients_ReturnsEmptyDictionary()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            // Act
            var gradients = accumulator.GetGradients();

            // Assert
            Assert.NotNull(gradients);
            Assert.Empty(gradients);
        }

        [Fact]
        public void GetAndClearGradients_WithNoGradients_ReturnsEmptyDictionary()
        {
            // Arrange
            var accumulator = new GradientAccumulator(3);

            // Act
            var gradients = accumulator.GetAndClearGradients();

            // Assert
            Assert.NotNull(gradients);
            Assert.Empty(gradients);
        }

        [Fact]
        public void AccumulateGradients_WithVaryingSizes_AccumulatesCorrectly()
        {
            // Arrange
            var accumulator = new GradientAccumulator(2);

            // First batch with smaller gradient
            var gradients1 = new Dictionary<string, Tensor>();
            gradients1["param0"] = new Tensor(new float[] { 1.0f, 2.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients1);

            // Second batch with same gradient
            var gradients2 = new Dictionary<string, Tensor>();
            gradients2["param0"] = new Tensor(new float[] { 3.0f, 4.0f }, new[] { 2 }, false, DataType.Float32);
            accumulator.AddGradients(gradients2);

            // Act
            var summed = accumulator.GetGradients();

            // Assert
            Assert.Equal(2, summed["param0"].Size);
            Assert.Equal(4.0f, summed["param0"].Data[0]); // 1.0 + 3.0
            Assert.Equal(6.0f, summed["param0"].Data[1]); // 2.0 + 4.0
        }
    }
}
