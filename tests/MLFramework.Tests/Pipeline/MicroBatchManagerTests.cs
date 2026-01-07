using System;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    public class MicroBatchManagerTests : IDisposable
    {
        private readonly IDevice _device;

        public MicroBatchManagerTests()
        {
            _device = DeviceManager.DefaultDevice;
        }

        public void Dispose()
        {
        }

        private Tensor CreateBatch(int batchSize, int featureSize)
        {
            var data = new float[batchSize * featureSize];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)i;
            }
            return new Tensor(data, new long[] { batchSize, featureSize });
        }

        private Tensor CreateGradient(int size)
        {
            var data = new float[size];
            for (int i = 0; i < size.Length; i++)
            {
                data[i] = 1.0f;
            }
            return new Tensor(data, new long[] { size });
        }

        [Fact]
        public void Constructor_WithValidInputs_CreatesManager()
        {
            // Act
            var manager = new MicroBatchManager(32, 4, _device);

            // Assert
            Assert.Equal(32, manager.TotalBatchSize);
            Assert.Equal(4, manager.NumMicroBatches);
            Assert.Equal(8, manager.MicroBatchSize);
        }

        [Fact]
        public void Constructor_WithInvalidTotalBatchSize_Throws()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new MicroBatchManager(0, 4, _device));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new MicroBatchManager(-1, 4, _device));
        }

        [Fact]
        public void Constructor_WithInvalidNumMicroBatches_Throws()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new MicroBatchManager(32, 0, _device));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                new MicroBatchManager(32, -1, _device));
        }

        [Fact]
        public void Constructor_WithNullDevice_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new MicroBatchManager(32, 4, null!));
        }

        [Fact]
        public void SplitBatch_WithEvenDivision_CreatesCorrectMicroBatches()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var batch = CreateBatch(32, 10);

            // Act
            var microBatches = manager.SplitBatch(batch);

            // Assert
            Assert.Equal(4, microBatches.Count);
            Assert.All(microBatches, mb => Assert.Equal(8, mb.Shape[0]));
        }

        [Fact]
        public void SplitBatch_WithUnevenDivision_DistributesCorrectly()
        {
            // Arrange
            var manager = new MicroBatchManager(30, 4, _device);
            var batch = CreateBatch(30, 10);

            // Act
            var microBatches = manager.SplitBatch(batch);

            // Assert
            Assert.Equal(4, microBatches.Count);
            Assert.Equal(8, microBatches[0].Shape[0]);
            Assert.Equal(8, microBatches[1].Shape[0]);
            Assert.Equal(8, microBatches[2].Shape[0]);
            Assert.Equal(6, microBatches[3].Shape[0]); // Last gets remainder
        }

        [Fact]
        public void SplitBatch_WithSingleMicroBatch_ReturnsOriginalBatch()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 1, _device);
            var batch = CreateBatch(32, 10);

            // Act
            var microBatches = manager.SplitBatch(batch);

            // Assert
            Assert.Single(microBatches);
            Assert.Equal(32, microBatches[0].Shape[0]);
        }

        [Fact]
        public void SplitBatch_WithNullBatch_ThrowsArgumentNullException()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                manager.SplitBatch(null!));
        }

        [Fact]
        public void SplitBatch_WithWrongBatchSize_ThrowsArgumentException()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var batch = CreateBatch(16, 10); // Wrong size

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                manager.SplitBatch(batch));
        }

        [Fact]
        public void CombineOutputs_WithCorrectInputs_RestoresOriginalShape()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var microBatches = new[]
            {
                CreateBatch(8, 10),
                CreateBatch(8, 10),
                CreateBatch(8, 10),
                CreateBatch(8, 10)
            };

            // Act
            var combined = manager.CombineOutputs(microBatches.ToList());

            // Assert
            Assert.Equal(32, combined.Shape[0]);
            Assert.Equal(10, combined.Shape[1]);
        }

        [Fact]
        public void CombineOutputs_WithUnevenMicroBatches_RestoresOriginalShape()
        {
            // Arrange
            var manager = new MicroBatchManager(30, 4, _device);
            var microBatches = new[]
            {
                CreateBatch(8, 10),
                CreateBatch(8, 10),
                CreateBatch(8, 10),
                CreateBatch(6, 10)
            };

            // Act
            var combined = manager.CombineOutputs(microBatches.ToList());

            // Assert
            Assert.Equal(30, combined.Shape[0]);
            Assert.Equal(10, combined.Shape[1]);
        }

        [Fact]
        public void CombineOutputs_WithNullOutputs_ThrowsArgumentNullException()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                manager.CombineOutputs(null!));
        }

        [Fact]
        public void CombineOutputs_WithWrongCount_ThrowsArgumentException()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var microBatches = new[]
            {
                CreateBatch(8, 10),
                CreateBatch(8, 10)
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                manager.CombineOutputs(microBatches.ToList()));
        }

        [Fact]
        public void AccumulateGradients_AveragesCorrectly()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var gradients = new[]
            {
                CreateGradient(100),
                CreateGradient(100),
                CreateGradient(100),
                CreateGradient(100)
            };

            // Act
            for (int i = 0; i < 4; i++)
            {
                manager.AccumulateGradients(new[] { gradients[i] });
            }

            var accumulated = manager.GetAccumulatedGradients();

            // Assert - average should be 1.0 (each gradient had all 1.0s)
            Assert.Single(accumulated);
            Assert.All(accumulated[0].Data, val => Assert.Equal(1.0f, val));
        }

        [Fact]
        public void AccumulateGradients_WithDifferentValues_AveragesCorrectly()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var gradients = new[]
            {
                CreateGradientWithValues(100, 1.0f),
                CreateGradientWithValues(100, 2.0f),
                CreateGradientWithValues(100, 3.0f),
                CreateGradientWithValues(100, 4.0f)
            };

            // Act
            for (int i = 0; i < 4; i++)
            {
                manager.AccumulateGradients(new[] { gradients[i] });
            }

            var accumulated = manager.GetAccumulatedGradients();

            // Assert - average should be 2.5 (1+2+3+4)/4
            Assert.Single(accumulated);
            Assert.All(accumulated[0].Data, val => Assert.Equal(2.5f, val));
        }

        [Fact]
        public void AccumulateGradients_WithNullGradients_ThrowsArgumentNullException()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                manager.AccumulateGradients(null!));

            Assert.Throws<ArgumentNullException>(() =>
                manager.AccumulateGradients(new[] { null! }));
        }

        [Fact]
        public void AccumulateGradients_WithWrongCount_ThrowsArgumentException()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var grad1 = CreateGradient(100);

            // Act - First accumulation
            manager.AccumulateGradients(new[] { grad1 });

            // Assert - Second accumulation with different count should throw
            var grad2 = CreateGradient(50);
            Assert.Throws<ArgumentException>(() =>
                manager.AccumulateGradients(new[] { grad2 }));
        }

        [Fact]
        public void AccumulateGradients_WithMismatchedShapes_ThrowsArgumentException()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var grad1 = CreateGradient(100);

            // Act - First accumulation
            manager.AccumulateGradients(new[] { grad1 });

            // Assert - Second accumulation with different shape should throw
            var grad2 = CreateGradient(150);
            Assert.Throws<ArgumentException>(() =>
                manager.AccumulateGradients(new[] { grad2 }));
        }

        [Fact]
        public void ResetGradients_ClearsAccumulatedGradients()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var gradients = new[] { CreateGradient(100) };

            manager.AccumulateGradients(gradients);
            Assert.False(manager.IsComplete);

            // Act
            manager.ResetGradients();

            // Assert
            Assert.False(manager.IsComplete);

            // Accumulate again
            manager.AccumulateGradients(gradients);
            Assert.False(manager.IsComplete);
        }

        [Fact]
        public void IsComplete_AfterAccumulatingAll_ReturnsTrue()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var gradients = new[] { CreateGradient(100) };

            // Act & Assert
            for (int i = 0; i < 4; i++)
            {
                Assert.False(manager.IsComplete);
                manager.AccumulateGradients(gradients);
            }

            Assert.True(manager.IsComplete);
        }

        [Fact]
        public void IsComplete_AfterReset_ReturnsFalse()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var gradients = new[] { CreateGradient(100) };

            // Accumulate all
            for (int i = 0; i < 4; i++)
            {
                manager.AccumulateGradients(gradients);
            }
            Assert.True(manager.IsComplete);

            // Act
            manager.ResetGradients();

            // Assert
            Assert.False(manager.IsComplete);
        }

        [Fact]
        public void FullCycle_SplitAccumulateReset_WorksCorrectly()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);
            var batch = CreateBatch(32, 10);

            // Act - Split
            var microBatches = manager.SplitBatch(batch);

            // Accumulate
            foreach (var mb in microBatches)
            {
                var grad = mb.Clone(); // Simulate computing gradient
                manager.AccumulateGradients(new[] { grad });
            }

            // Get gradients
            var accumulated = manager.GetAccumulatedGradients();

            // Reset
            manager.ResetGradients();

            // Assert
            Assert.True(manager.IsComplete); // After accumulation
            manager.ResetGradients();
            Assert.False(manager.IsComplete); // After reset
            Assert.NotNull(accumulated);
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            // Arrange
            var manager = new MicroBatchManager(32, 4, _device);

            // Act - Should not throw
            manager.Dispose();
            manager.Dispose();

            // Assert - No exception
        }

        private Tensor CreateGradientWithValues(int size, float value)
        {
            var data = new float[size];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = value;
            }
            return new Tensor(data, new long[] { size });
        }
    }
}
