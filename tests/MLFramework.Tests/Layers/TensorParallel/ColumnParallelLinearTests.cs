using MLFramework.Layers.TensorParallel;
using MLFramework.Tests.Distributed;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Layers.TensorParallel
{
    /// <summary>
    /// Unit tests for Column Parallel Linear layer.
    /// </summary>
    public class ColumnParallelLinearTests : IDisposable
    {
        private readonly MockProcessGroup _mockProcessGroup;

        public ColumnParallelLinearTests()
        {
            _mockProcessGroup = new MockProcessGroup(worldSize: 4, rank: 0);
            TensorParallel.Initialize(_mockProcessGroup);
        }

        public void Dispose()
        {
            _mockProcessGroup?.Dispose();
        }

        #region Constructor Tests

        [Fact]
        public void Constructor_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new ColumnParallelLinear(
                inputSize: 128,
                outputSize: 256,
                bias: true,
                gatherOutput: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(128, layer.InputSize);
            Assert.Equal(256, layer.OutputSize);
            Assert.Equal(64, layer.LocalOutputSize); // 256 / 4 = 64
            Assert.NotNull(layer.GetLocalWeight());
            Assert.NotNull(layer.GetLocalBias());
        }

        [Fact]
        public void Constructor_WithoutBias_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new ColumnParallelLinear(
                inputSize: 128,
                outputSize: 256,
                bias: false,
                gatherOutput: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Null(layer.GetLocalBias());
        }

        [Fact]
        public void Constructor_OutputNotDivisibleByWorldSize_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new ColumnParallelLinear(
                    inputSize: 128,
                    outputSize: 255, // Not divisible by 4
                    bias: true,
                    gatherOutput: false));
        }

        [Fact]
        public void Constructor_WithCustomProcessGroup_UsesProvidedProcessGroup()
        {
            // Arrange
            var customGroup = new TensorParallelGroup(_mockProcessGroup);

            // Act
            var layer = new ColumnParallelLinear(
                inputSize: 64,
                outputSize: 128,
                bias: true,
                gatherOutput: false,
                processGroup: customGroup);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(64, layer.InputSize);
            Assert.Equal(128, layer.OutputSize);
        }

        #endregion

        #region Forward Pass Tests

        [Fact]
        public void Forward_ProducesCorrectOutputShape_WithoutGather()
        {
            // Arrange
            var layer = new ColumnParallelLinear(
                inputSize: 64,
                outputSize: 128,
                bias: true,
                gatherOutput: false);
            var input = new Tensor(new float[10 * 64], new[] { 10, 64 });

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(new[] { 10, 32 }, output.Shape); // 128 / 4 = 32
        }

        [Fact]
        public void Forward_WithGatherOutput_ProducesFullOutput()
        {
            // Arrange
            var layer = new ColumnParallelLinear(
                inputSize: 64,
                outputSize: 128,
                bias: false,
                gatherOutput: true);
            var input = new Tensor(new float[10 * 64], new[] { 10, 64 });

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(new[] { 10, 128 }, output.Shape); // Full output
        }

        [Fact]
        public void Forward_WithBias_AddsBiasCorrectly()
        {
            // Arrange
            var layer = new ColumnParallelLinear(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: false);
            var input = new Tensor(new float[5 * 32], new[] { 5, 32 });

            // Initialize with known weights and bias
            var weightData = new float[16 * 32]; // [16, 32] (64/4=16)
            for (int i = 0; i < weightData.Length; i++)
            {
                weightData[i] = 0.0f;
            }
            var weight = layer.GetLocalWeight();
            for (int i = 0; i < Math.Min(weightData.Length, weight.Data.Length); i++)
            {
                weight.Data[i] = weightData[i];
            }

            var bias = layer.GetLocalBias();
            if (bias != null)
            {
                for (int i = 0; i < bias.Data.Length; i++)
                {
                    bias.Data[i] = 1.0f; // Set bias to 1.0
                }
            }

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new[] { 5, 16 }, output.Shape);
        }

        [Fact]
        public void Forward_WithDifferentBatchSizes_WorksCorrectly()
        {
            // Arrange
            var layer = new ColumnParallelLinear(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: false);

            // Act & Assert - Test different batch sizes
            var batch1 = new Tensor(new float[1 * 32], new[] { 1, 32 });
            var output1 = layer.Forward(batch1);
            Assert.Equal(new[] { 1, 16 }, output1.Shape);

            var batch10 = new Tensor(new float[10 * 32], new[] { 10, 32 });
            var output10 = layer.Forward(batch10);
            Assert.Equal(new[] { 10, 16 }, output10.Shape);

            var batch100 = new Tensor(new float[100 * 32], new[] { 100, 32 });
            var output100 = layer.Forward(batch100);
            Assert.Equal(new[] { 100, 16 }, output100.Shape);
        }

        #endregion

        #region Weight Sharding Tests

        [Fact]
        public void GetLocalWeightShape_ReturnsCorrectDimensions()
        {
            // Arrange
            var layer = new ColumnParallelLinear(
                inputSize: 128,
                outputSize: 256,
                bias: true);

            // Act
            var (rows, cols) = layer.GetLocalWeightShape();

            // Assert
            Assert.Equal(64, rows); // 256 / 4 = 64
            Assert.Equal(128, cols);
        }

        [Fact]
        public void GetLocalWeight_ReturnsShardedWeight()
        {
            // Arrange
            var layer = new ColumnParallelLinear(
                inputSize: 64,
                outputSize: 128,
                bias: true);

            // Act
            var weight = layer.GetLocalWeight();

            // Assert
            Assert.NotNull(weight);
            Assert.Equal(new[] { 32, 64 }, weight.Shape); // [128/4, 64]
        }

        [Fact]
        public void GetLocalBias_ReturnsShardedBias()
        {
            // Arrange
            var layer = new ColumnParallelLinear(
                inputSize: 64,
                outputSize: 128,
                bias: true);

            // Act
            var bias = layer.GetLocalBias();

            // Assert
            Assert.NotNull(bias);
            Assert.Equal(new[] { 32 }, bias.Shape); // [128/4]
        }

        #endregion

        #region ColumnParallelLinearWithInputGather Tests

        [Fact]
        public void ColumnParallelLinearWithInputGather_Constructor_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new ColumnParallelLinearWithInputGather(
                inputSize: 128,
                outputSize: 256,
                bias: true,
                gatherOutput: false,
                inputIsSharded: true);

            // Assert
            Assert.NotNull(layer);
            Assert.True(layer.InputIsSharded);
            Assert.Equal(128, layer.InputSize);
            Assert.Equal(256, layer.OutputSize);
        }

        [Fact]
        public void ColumnParallelLinearWithInputGather_Forward_WithoutInputGathering_WorksCorrectly()
        {
            // Arrange
            var layer = new ColumnParallelLinearWithInputGather(
                inputSize: 64,
                outputSize: 128,
                bias: false,
                gatherOutput: true,
                inputIsSharded: false);
            var input = new Tensor(new float[10 * 64], new[] { 10, 64 });

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new[] { 10, 128 }, output.Shape);
        }

        #endregion

        #region Factory Tests

        [Fact]
        public void Factory_Create_ReturnsValidLayer()
        {
            // Arrange & Act
            var layer = ColumnParallelLinearFactory.Create(
                inputSize: 64,
                outputSize: 128,
                bias: true,
                gatherOutput: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(64, layer.InputSize);
            Assert.Equal(128, layer.OutputSize);
        }

        [Fact]
        public void Factory_CreateForAttention_CreatesLayerWithCorrectDefaults()
        {
            // Arrange & Act
            var layer = ColumnParallelLinearFactory.CreateForAttention(
                inputSize: 512,
                outputSize: 2048);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(512, layer.InputSize);
            Assert.Equal(2048, layer.OutputSize);
            Assert.Null(layer.GetLocalBias()); // No bias for attention
            Assert.True(layer.GatherOutput); // Gather output for attention
        }

        [Fact]
        public void Factory_CreateForMLPHidden_CreatesLayerWithCorrectDefaults()
        {
            // Arrange & Act
            var layer = ColumnParallelLinearFactory.CreateForMLPHidden(
                inputSize: 768,
                hiddenSize: 3072);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(768, layer.InputSize);
            Assert.Equal(3072, layer.OutputSize);
            Assert.NotNull(layer.GetLocalBias()); // Has bias for MLP
            Assert.False(layer.GatherOutput); // Don't gather - feed to row parallel
        }

        [Fact]
        public void Factory_CreateWithInputGather_ReturnsValidLayer()
        {
            // Arrange & Act
            var layer = ColumnParallelLinearFactory.CreateWithInputGather(
                inputSize: 256,
                outputSize: 512,
                bias: true,
                gatherOutput: false,
                inputIsSharded: true);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(256, layer.InputSize);
            Assert.Equal(512, layer.OutputSize);
            Assert.True(layer.InputIsSharded);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Forward_WithZeroInput_WorksCorrectly()
        {
            // Arrange
            var layer = new ColumnParallelLinear(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: false);
            var input = new Tensor(new float[5 * 32], new[] { 5, 32 });

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new[] { 5, 16 }, output.Shape);
        }

        [Fact]
        public void Constructor_WithSingleWorldSize_WorksCorrectly()
        {
            // Arrange
            var singleGroup = new MockProcessGroup(worldSize: 1, rank: 0);
            TensorParallel.Initialize(singleGroup);

            // Act
            var layer = new ColumnParallelLinear(
                inputSize: 64,
                outputSize: 128,
                bias: true,
                gatherOutput: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(64, layer.InputSize);
            Assert.Equal(128, layer.OutputSize);
            Assert.Equal(128, layer.LocalOutputSize); // No sharding with world size 1

            singleGroup.Dispose();
        }

        #endregion
    }
}
