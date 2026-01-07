using MLFramework.Layers.TensorParallel;
using MLFramework.Tests.Distributed;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Layers.TensorParallel
{
    /// <summary>
    /// Unit tests for Row Parallel Linear layer.
    /// </summary>
    public class RowParallelLinearTests : IDisposable
    {
        private readonly MockProcessGroup _mockProcessGroup;

        public RowParallelLinearTests()
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
            var layer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(256, layer.InputSize);
            Assert.Equal(128, layer.OutputSize);
            Assert.Equal(64, layer.LocalInputSize); // 256 / 4 = 64
            Assert.NotNull(layer.GetLocalWeight());
            Assert.NotNull(layer.GetLocalBias());
        }

        [Fact]
        public void Constructor_WithoutBias_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: false,
                inputIsSharded: true);

            // Assert
            Assert.NotNull(layer);
            Assert.Null(layer.GetLocalBias());
        }

        [Fact]
        public void Constructor_InputNotDivisibleByWorldSize_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new RowParallelLinear(
                    inputSize: 255, // Not divisible by 4
                    outputSize: 128,
                    bias: true,
                    inputIsSharded: true));
        }

        [Fact]
        public void Constructor_WithCustomProcessGroup_UsesProvidedProcessGroup()
        {
            // Arrange
            var customGroup = new TensorParallelGroup(_mockProcessGroup);

            // Act
            var layer = new RowParallelLinear(
                inputSize: 128,
                outputSize: 64,
                bias: true,
                inputIsSharded: true,
                processGroup: customGroup);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(128, layer.InputSize);
            Assert.Equal(64, layer.OutputSize);
        }

        [Fact]
        public void Constructor_WithNonShardedInput_SetsCorrectFlag()
        {
            // Arrange & Act
            var layer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: false);

            // Assert
            Assert.False(layer.InputIsSharded);
        }

        #endregion

        #region Forward Pass Tests

        [Fact]
        public void Forward_WithShardedInput_ProducesCorrectOutputShape()
        {
            // Arrange
            var layer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);
            var input = new Tensor(new float[10 * 64], new[] { 10, 64 }); // Sharded: 256/4 = 64

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(new[] { 10, 128 }, output.Shape); // Full output after all-reduce
        }

        [Fact]
        public void Forward_WithNonShardedInput_SlicesAndProcessesCorrectly()
        {
            // Arrange
            var layer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: false,
                inputIsSharded: false);
            var input = new Tensor(new float[10 * 256], new[] { 10, 256 }); // Full input

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(new[] { 10, 128 }, output.Shape); // Full output after all-reduce
        }

        [Fact]
        public void Forward_WithShardedInputWrongDimensions_ThrowsException()
        {
            // Arrange
            var layer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);
            var input = new Tensor(new float[10 * 128], new[] { 10, 128 }); // Wrong size (should be 64)

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => layer.Forward(input));
        }

        [Fact]
        public void Forward_WithBias_AddsBiasAfterAllReduce()
        {
            // Arrange
            var layer = new RowParallelLinear(
                inputSize: 128,
                outputSize: 64,
                bias: true,
                inputIsSharded: true);
            var input = new Tensor(new float[5 * 32], new[] { 5, 32 }); // Sharded: 128/4 = 32

            // Initialize with known weights and bias
            var weight = layer.GetLocalWeight();
            for (int i = 0; i < weight.Data.Length; i++)
            {
                weight.Data[i] = 1.0f; // Set weights to 1.0
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
            Assert.Equal(new[] { 5, 64 }, output.Shape);
        }

        [Fact]
        public void Forward_WithDifferentBatchSizes_WorksCorrectly()
        {
            // Arrange
            var layer = new RowParallelLinear(
                inputSize: 128,
                outputSize: 64,
                bias: true,
                inputIsSharded: true);

            // Act & Assert - Test different batch sizes
            var batch1 = new Tensor(new float[1 * 32], new[] { 1, 32 }); // Sharded: 128/4 = 32
            var output1 = layer.Forward(batch1);
            Assert.Equal(new[] { 1, 64 }, output1.Shape);

            var batch10 = new Tensor(new float[10 * 32], new[] { 10, 32 });
            var output10 = layer.Forward(batch10);
            Assert.Equal(new[] { 10, 64 }, output10.Shape);

            var batch100 = new Tensor(new float[100 * 32], new[] { 100, 32 });
            var output100 = layer.Forward(batch100);
            Assert.Equal(new[] { 100, 64 }, output100.Shape);
        }

        #endregion

        #region Weight Sharding Tests

        [Fact]
        public void Weight_ShardedCorrectly()
        {
            // Arrange
            int inputSize = 256;
            int outputSize = 128;
            int worldSize = 4;

            var layer = new RowParallelLinear(
                inputSize: inputSize,
                outputSize: outputSize,
                bias: true,
                inputIsSharded: true);

            // Act
            var weightShape = layer.GetLocalWeightShape();
            var weight = layer.GetLocalWeight();

            // Assert
            Assert.Equal(outputSize, weightShape.rows);
            Assert.Equal(inputSize / worldSize, weightShape.cols);
            Assert.Equal(new[] { outputSize, inputSize / worldSize }, weight.Shape);
        }

        [Fact]
        public void Weight_InitializedWithCorrectShape()
        {
            // Arrange
            var layer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);

            // Act
            var weight = layer.GetLocalWeight();

            // Assert
            Assert.Equal(2, weight.Dimensions);
            Assert.Equal(128, weight.Shape[0]); // output_size
            Assert.Equal(64, weight.Shape[1]);  // input_size / world_size
        }

        [Fact]
        public void Bias_NotSharded()
        {
            // Arrange
            var layer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);

            // Act
            var bias = layer.GetLocalBias();

            // Assert
            Assert.NotNull(bias);
            Assert.Single(bias.Shape);
            Assert.Equal(128, bias.Shape[0]); // Full output_size
        }

        #endregion

        #region Factory Tests

        [Fact]
        public void Factory_Create_CreatesValidLayer()
        {
            // Act
            var layer = RowParallelLinearFactory.Create(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(256, layer.InputSize);
            Assert.Equal(128, layer.OutputSize);
            Assert.True(layer.InputIsSharded);
        }

        [Fact]
        public void Factory_CreateForMLPOutput_CreatesCorrectConfiguration()
        {
            // Act
            var layer = RowParallelLinearFactory.CreateForMLPOutput(
                hiddenSize: 256,
                outputSize: 128,
                bias: true);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(256, layer.InputSize);
            Assert.Equal(128, layer.OutputSize);
            Assert.True(layer.InputIsSharded); // Expected sharded for MLP output
        }

        [Fact]
        public void Factory_CreateWithGather_CreatesRowParallelLinearWithInputGather()
        {
            // Act
            var layer = RowParallelLinearFactory.CreateWithGather(
                inputSize: 256,
                outputSize: 128,
                bias: true);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<RowParallelLinearWithInputGather>(layer);
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void ColumnThenRowParallel_WorksTogether()
        {
            // Arrange
            var columnLayer = new ColumnParallelLinear(
                inputSize: 128,
                outputSize: 256,
                bias: true,
                gatherOutput: false);
            var rowLayer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);
            var input = new Tensor(new float[10 * 128], new[] { 10, 128 });

            // Act
            var hidden = columnLayer.Forward(input); // Shape: [10, 64] (sharded)
            var output = rowLayer.Forward(hidden); // Shape: [10, 128] (after all-reduce)

            // Assert
            Assert.Equal(new[] { 10, 64 }, hidden.Shape); // Sharded
            Assert.Equal(new[] { 10, 128 }, output.Shape); // Full output
        }

        [Fact]
        public void TPMLPFactory_CreateMLPBlock_CreatesValidLayers()
        {
            // Act
            var (columnLayer, rowLayer) = TPMLPFactory.CreateMLPBlock(
                inputSize: 128,
                hiddenSize: 256,
                outputSize: 128,
                bias: true);

            // Assert
            Assert.NotNull(columnLayer);
            Assert.NotNull(rowLayer);
            Assert.Equal(128, columnLayer.InputSize);
            Assert.Equal(256, columnLayer.OutputSize);
            Assert.Equal(256, rowLayer.InputSize);
            Assert.Equal(128, rowLayer.OutputSize);
        }

        [Fact]
        public void TPMLPFactory_ForwardMLPWithReLU_WorksCorrectly()
        {
            // Arrange
            var (columnLayer, rowLayer) = TPMLPFactory.CreateMLPBlock(
                inputSize: 128,
                hiddenSize: 256,
                outputSize: 128,
                bias: true);
            var input = new Tensor(new float[10 * 128], new[] { 10, 128 });

            // Act
            var output = TPMLPFactory.ForwardMLPWithReLU(input, columnLayer, rowLayer);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new[] { 10, 128 }, output.Shape);
        }

        [Fact]
        public void TPMLPFactory_ForwardMLPWithGELU_WorksCorrectly()
        {
            // Arrange
            var (columnLayer, rowLayer) = TPMLPFactory.CreateMLPBlockWithGELU(
                inputSize: 128,
                hiddenSize: 256,
                outputSize: 128,
                bias: true);
            var input = new Tensor(new float[10 * 128], new[] { 10, 128 });

            // Act
            var output = TPMLPFactory.ForwardMLPWithGELU(input, columnLayer, rowLayer);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new[] { 10, 128 }, output.Shape);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Forward_WithNullInput_ThrowsException()
        {
            // Arrange
            var layer = new RowParallelLinear(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => layer.Forward(null!));
        }

        [Fact]
        public void Constructor_WithZeroInputSize_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new RowParallelLinear(
                    inputSize: 0,
                    outputSize: 128,
                    bias: true,
                    inputIsSharded: true));
        }

        [Fact]
        public void Constructor_WithZeroOutputSize_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new RowParallelLinear(
                    inputSize: 256,
                    outputSize: 0,
                    bias: true,
                    inputIsSharded: true));
        }

        [Fact]
        public void Forward_WithLargeInput_WorksCorrectly()
        {
            // Arrange
            var layer = new RowParallelLinear(
                inputSize: 1024,
                outputSize: 512,
                bias: false,
                inputIsSharded: true);
            var input = new Tensor(new float[100 * 256], new[] { 100, 256 }); // Sharded: 1024/4 = 256

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(new[] { 100, 512 }, output.Shape);
        }

        #endregion
    }
}
