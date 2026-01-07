using System;
using System.Linq;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.NN;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    public class LayerPartitionerTests : IDisposable
    {
        private readonly IDevice _device;

        public LayerPartitionerTests()
        {
            _device = DeviceManager.DefaultDevice;
        }

        public void Dispose()
        {
        }

        /// <summary>
        /// Creates a simple sequential model for testing
        /// </summary>
        private Module CreateSimpleModel(int numLayers, int inputSize = 10, int hiddenSize = 10)
        {
            var sequential = new SequentialModule("TestModel");

            for (int i = 0; i < numLayers; i++)
            {
                var layer = new Linear(inputSize, hiddenSize, $"Linear_{i}");
                sequential.Add(layer);
            }

            return sequential;
        }

        [Fact]
        public void Constructor_WithValidInputs_CreatesPartitioner()
        {
            // Arrange & Act
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 4);

            // Assert
            Assert.Equal(PartitionMode.Uniform, partitioner.Mode);
            Assert.Equal(4, partitioner.NumStages);
        }

        [Fact]
        public void Constructor_WithInvalidNumStages_Throws()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => new LayerPartitioner(PartitionMode.Uniform, 0));
            Assert.Throws<ArgumentException>(() => new LayerPartitioner(PartitionMode.Uniform, -1));
        }

        [Fact]
        public void UniformPartition_WithEvenLayers_CreatesEqualStages()
        {
            // Arrange
            var model = CreateSimpleModel(12); // 12 layers
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 4);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.Equal(4, result.Stages.Count);
            Assert.Equal(4, result.StageLayerIndices.Count);

            // Each stage should have 3 layers
            for (int i = 0; i < 4; i++)
            {
                Assert.Equal(3, result.StageLayerIndices[i].Count);
            }
        }

        [Fact]
        public void UniformPartition_WithUnevenLayers_DistributesCorrectly()
        {
            // Arrange
            var model = CreateSimpleModel(10); // 10 layers
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 4);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.Equal(4, result.Stages.Count);

            // First 2 stages should have 3 layers, last 2 should have 2
            // 10 / 4 = 2 remainder 2, so first 2 stages get 3 layers
            Assert.Equal(3, result.StageLayerIndices[0].Count);
            Assert.Equal(3, result.StageLayerIndices[1].Count);
            Assert.Equal(2, result.StageLayerIndices[2].Count);
            Assert.Equal(2, result.StageLayerIndices[3].Count);
        }

        [Fact]
        public void ManualPartition_WithValidInput_CreatesSpecifiedStages()
        {
            // Arrange
            var model = CreateSimpleModel(10);
            var manualPartitions = new System.Collections.Generic.List<System.Collections.Generic.List<int>>
            {
                new System.Collections.Generic.List<int> { 0, 1, 2 },
                new System.Collections.Generic.List<int> { 3, 4, 5, 6 },
                new System.Collections.Generic.List<int> { 7, 8, 9 }
            };
            var partitioner = new LayerPartitioner(PartitionMode.Manual, 3);

            // Act
            var result = partitioner.Partition(model, manualPartitions);

            // Assert
            Assert.Equal(3, result.Stages.Count);
            Assert.Equal(3, result.StageLayerIndices[0].Count);
            Assert.Equal(4, result.StageLayerIndices[1].Count);
            Assert.Equal(3, result.StageLayerIndices[2].Count);

            // Verify layer assignments
            Assert.Equal(new[] { 0, 1, 2 }, result.StageLayerIndices[0].ToArray());
            Assert.Equal(new[] { 3, 4, 5, 6 }, result.StageLayerIndices[1].ToArray());
            Assert.Equal(new[] { 7, 8, 9 }, result.StageLayerIndices[2].ToArray());
        }

        [Fact]
        public void ManualPartition_WithInvalidLayerIndices_Throws()
        {
            // Arrange
            var model = CreateSimpleModel(10);
            var manualPartitions = new System.Collections.Generic.List<System.Collections.Generic.List<int>>
            {
                new System.Collections.Generic.List<int> { 0, 1, 2 },
                new System.Collections.Generic.List<int> { 3, 4, 5 },
                new System.Collections.Generic.List<int> { 100 } // Invalid index
            };
            var partitioner = new LayerPartitioner(PartitionMode.Manual, 3);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                partitioner.Partition(model, manualPartitions));
        }

        [Fact]
        public void ManualPartition_WithGaps_Throws()
        {
            // Arrange
            var model = CreateSimpleModel(10);
            var manualPartitions = new System.Collections.Generic.List<System.Collections.Generic.List<int>>
            {
                new System.Collections.Generic.List<int> { 0, 1, 2 },
                new System.Collections.Generic.List<int> { 4, 5, 6 }, // Missing 3
                new System.Collections.Generic.List<int> { 7, 8, 9 }
            };
            var partitioner = new LayerPartitioner(PartitionMode.Manual, 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                partitioner.Partition(model, manualPartitions));
        }

        [Fact]
        public void ManualPartition_WithUnorderedLayers_Throws()
        {
            // Arrange
            var model = CreateSimpleModel(10);
            var manualPartitions = new System.Collections.Generic.List<System.Collections.Generic.List<int>>
            {
                new System.Collections.Generic.List<int> { 2, 1, 0 }, // Not in order
                new System.Collections.Generic.List<int> { 3, 4, 5 },
                new System.Collections.Generic.List<int> { 6, 7, 8, 9 }
            };
            var partitioner = new LayerPartitioner(PartitionMode.Manual, 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                partitioner.Partition(model, manualPartitions));
        }

        [Fact]
        public void ManualPartition_WithoutPartitions_Throws()
        {
            // Arrange
            var model = CreateSimpleModel(10);
            var partitioner = new LayerPartitioner(PartitionMode.Manual, 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                partitioner.Partition(model, null));
        }

        [Fact]
        public void AutomaticPartition_BalancesMemory()
        {
            // Arrange
            var model = CreateSimpleModel(10);
            var partitioner = new LayerPartitioner(PartitionMode.Automatic, 4);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.Equal(4, result.Stages.Count);
            Assert.Equal(4, result.StageLayerIndices.Count);

            // All layers should be assigned
            int totalLayers = result.StageLayerIndices.Sum(s => s.Count);
            Assert.Equal(10, totalLayers);

            // Load balance should be reasonable (< 2.0)
            Assert.True(result.LoadBalance < 2.0f);

            // Verify no gaps or overlaps
            var allAssigned = new System.Collections.Generic.HashSet<int>();
            for (int stage = 0; stage < result.StageLayerIndices.Count; stage++)
            {
                foreach (int layerIdx in result.StageLayerIndices[stage])
                {
                    Assert.DoesNotContain(layerIdx, allAssigned);
                    allAssigned.Add(layerIdx);
                }
            }
        }

        [Fact]
        public void AutomaticPartition_WithSmallModel_CreatesStages()
        {
            // Arrange
            var model = CreateSimpleModel(5);
            var partitioner = new LayerPartitioner(PartitionMode.Automatic, 2);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.Equal(2, result.Stages.Count);
            Assert.Equal(5, result.StageLayerIndices.Sum(s => s.Count));
        }

        [Fact]
        public void Partition_PreservesLayerOrder()
        {
            // Arrange
            var model = CreateSimpleModel(12);
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 4);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            int expectedLayer = 0;
            for (int stage = 0; stage < result.StageLayerIndices.Count; stage++)
            {
                foreach (int layerIdx in result.StageLayerIndices[stage])
                {
                    Assert.Equal(expectedLayer, layerIdx);
                    expectedLayer++;
                }
            }
        }

        [Fact]
        public void Partition_WithSingleLayerModel_CreatesSingleStage()
        {
            // Arrange
            var model = CreateSimpleModel(1);
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 2);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.Equal(2, result.Stages.Count);
            Assert.Equal(1, result.StageLayerIndices[0].Count);
            Assert.Equal(0, result.StageLayerIndices[0][0]);
        }

        [Fact]
        public void Partition_WithMoreStagesThanLayers_HandlesCorrectly()
        {
            // Arrange
            var model = CreateSimpleModel(3);
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 5);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.Equal(5, result.Stages.Count);
            // First 3 stages should have 1 layer each, last 2 should have 0
            Assert.Equal(1, result.StageLayerIndices[0].Count);
            Assert.Equal(1, result.StageLayerIndices[1].Count);
            Assert.Equal(1, result.StageLayerIndices[2].Count);
            // Last stages may be empty - this is acceptable
        }

        [Fact]
        public void Partition_WithEmptyModel_Throws()
        {
            // Arrange
            var model = new SequentialModule("EmptyModel");
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 4);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                partitioner.Partition(model));
        }

        [Fact]
        public void PartitionResult_ContainsValidStages()
        {
            // Arrange
            var model = CreateSimpleModel(8);
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 2);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.Equal(2, result.Stages.Count);

            // Verify stages have correct ranks
            Assert.Equal(0, result.Stages[0].Rank);
            Assert.Equal(1, result.Stages[1].Rank);
            Assert.Equal(2, result.Stages[0].TotalStages);
            Assert.Equal(2, result.Stages[1].TotalStages);

            // Verify first and last stage properties
            Assert.True(result.Stages[0].IsFirstStage);
            Assert.False(result.Stages[0].IsLastStage);
            Assert.False(result.Stages[1].IsFirstStage);
            Assert.True(result.Stages[1].IsLastStage);
        }

        [Fact]
        public void PartitionResult_ContainsMemoryEstimates()
        {
            // Arrange
            var model = CreateSimpleModel(8);
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 2);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.NotNull(result.MemoryPerStage);
            Assert.Equal(2, result.MemoryPerStage.Length);
            Assert.All(result.MemoryPerStage, memory => Assert.True(memory > 0));
        }

        [Fact]
        public void PartitionResult_ContainsComputationEstimates()
        {
            // Arrange
            var model = CreateSimpleModel(8);
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 2);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.NotNull(result.ComputationPerStage);
            Assert.Equal(2, result.ComputationPerStage.Length);
            Assert.All(result.ComputationPerStage, comp => Assert.True(comp > 0));
        }

        [Fact]
        public void PartitionResult_CalculatesLoadBalance()
        {
            // Arrange
            var model = CreateSimpleModel(8);
            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 2);

            // Act
            var result = partitioner.Partition(model);

            // Assert
            Assert.True(result.LoadBalance >= 1.0f);
            // For uniform partitioning of equal layers, load balance should be close to 1.0
            Assert.True(result.LoadBalance < 1.1f);
        }
    }

    /// <summary>
    /// Simple linear layer for testing
    /// </summary>
    public class Linear : Module
    {
        private readonly int _inputSize;
        private readonly int _outputSize;
        private readonly Parameter _weight;
        private readonly Parameter _bias;

        public int InputSize => _inputSize;
        public int OutputSize => _outputSize;

        public Linear(int inputSize, int outputSize, string name = "Linear") : base(name)
        {
            _inputSize = inputSize;
            _outputSize = outputSize;

            // Create parameters
            _weight = new Parameter(Tensor.Zeros(new long[] { outputSize, inputSize }), "weight");
            _bias = new Parameter(Tensor.Zeros(new long[] { outputSize }), "bias");
        }

        public override Tensor Forward(Tensor input)
        {
            // Simplified forward pass (no actual computation for testing)
            return input;
        }

        public override IEnumerable<Parameter> GetParameters()
        {
            yield return _weight;
            yield return _bias;
        }

        public override IEnumerable<(string Name, Parameter Parameter)> GetNamedParameters()
        {
            yield return ("weight", _weight);
            yield return ("bias", _bias);
        }
    }
}
