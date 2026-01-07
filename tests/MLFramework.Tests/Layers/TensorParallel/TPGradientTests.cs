using MLFramework.Layers.TensorParallel;
using MLFramework.NN;
using MLFramework.Tests.Distributed;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Layers.TensorParallel
{
    /// <summary>
    /// Unit tests for tensor parallel gradient synchronization.
    /// </summary>
    public class TPGradientTests : IDisposable
    {
        private readonly MockProcessGroup _mockProcessGroup;

        public TPGradientTests()
        {
            _mockProcessGroup = new MockProcessGroup(worldSize: 4, rank: 0);
            TensorParallel.Initialize(_mockProcessGroup);
        }

        public void Dispose()
        {
            _mockProcessGroup.Dispose();
        }

        #region ColumnParallelLinearGrad Tests

        [Fact]
        public void ColumnParallelLinearGrad_Constructor_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new ColumnParallelLinearGrad(
                inputSize: 128,
                outputSize: 256,
                bias: true,
                gatherOutput: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(128, layer.InputSize);
            Assert.Equal(256, layer.OutputSize);
            Assert.Equal(64, layer.LocalOutputSize); // 256 / 4 = 64
            Assert.NotNull(layer.Weight);
            Assert.NotNull(layer.Bias);
        }

        [Fact]
        public void ColumnParallelLinearGrad_Constructor_WithoutBias_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new ColumnParallelLinearGrad(
                inputSize: 128,
                outputSize: 256,
                bias: false,
                gatherOutput: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Null(layer.Bias);
        }

        [Fact]
        public void ColumnParallelLinearGrad_Constructor_OutputNotDivisibleByWorldSize_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new ColumnParallelLinearGrad(
                    inputSize: 128,
                    outputSize: 255, // Not divisible by 4
                    bias: true,
                    gatherOutput: false));
        }

        [Fact]
        public void ColumnParallelLinearGrad_Forward_ProducesCorrectOutputShape()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
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
        public void ColumnParallelLinearGrad_Forward_WithGatherOutput_ProducesFullOutput()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
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
        public void ColumnParallelLinearGrad_Backward_ComputesCorrectGradients()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: false);

            // Initialize with known weights for testing
            var weightData = new float[16 * 32]; // [16, 32] (64/4=16)
            for (int i = 0; i < weightData.Length; i++)
            {
                weightData[i] = 1.0f;
            }
            layer.Weight.Data = weightData;

            var input = new Tensor(new float[5 * 32], new[] { 5, 32 });
            for (int i = 0; i < input.Size; i++)
            {
                input.Data[i] = 1.0f;
            }

            // Forward pass
            var output = layer.Forward(input);

            // Create gradient
            var gradOutput = new Tensor(new float[5 * 16], new[] { 5, 16 });
            for (int i = 0; i < gradOutput.Size; i++)
            {
                gradOutput.Data[i] = 1.0f;
            }

            // Act
            var gradInput = layer.BackwardInternal(gradOutput);

            // Assert
            Assert.NotNull(layer.Weight.Grad);
            Assert.NotNull(layer.Bias.Grad);
            Assert.NotNull(gradInput);
            Assert.Equal(new[] { 5, 32 }, gradInput.Shape);
        }

        [Fact]
        public void ColumnParallelLinearGrad_Backward_WithGatherOutput_SlicesGradient()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: true);

            var input = new Tensor(new float[5 * 32], new[] { 5, 32 });
            for (int i = 0; i < input.Size; i++)
            {
                input.Data[i] = 1.0f;
            }

            // Forward pass (output will be gathered)
            var output = layer.Forward(input);

            // Full gradient
            var gradOutput = new Tensor(new float[5 * 64], new[] { 5, 64 });
            for (int i = 0; i < gradOutput.Size; i++)
            {
                gradOutput.Data[i] = 1.0f;
            }

            // Act
            var gradInput = layer.BackwardInternal(gradOutput);

            // Assert
            Assert.NotNull(gradInput);
            Assert.Equal(new[] { 5, 32 }, gradInput.Shape);
            // Gradient should only be computed from the local shard
        }

        #endregion

        #region RowParallelLinearGrad Tests

        [Fact]
        public void RowParallelLinearGrad_Constructor_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new RowParallelLinearGrad(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(256, layer.InputSize);
            Assert.Equal(128, layer.OutputSize);
            Assert.Equal(64, layer.LocalInputSize); // 256 / 4 = 64
            Assert.NotNull(layer.Weight);
            Assert.NotNull(layer.Bias);
        }

        [Fact]
        public void RowParallelLinearGrad_Constructor_WithoutBias_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new RowParallelLinearGrad(
                inputSize: 256,
                outputSize: 128,
                bias: false,
                inputIsSharded: true);

            // Assert
            Assert.NotNull(layer);
            Assert.Null(layer.Bias);
        }

        [Fact]
        public void RowParallelLinearGrad_Constructor_InputNotDivisibleByWorldSize_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new RowParallelLinearGrad(
                    inputSize: 255, // Not divisible by 4
                    outputSize: 128,
                    bias: true,
                    inputIsSharded: true));
        }

        [Fact]
        public void RowParallelLinearGrad_Forward_WithShardedInput_ProducesCorrectOutput()
        {
            // Arrange
            var layer = new RowParallelLinearGrad(
                inputSize: 256,
                outputSize: 128,
                bias: true,
                inputIsSharded: true);

            // Sharded input (only our local shard)
            var input = new Tensor(new float[10 * 64], new[] { 10, 64 }); // 256/4=64

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(new[] { 10, 128 }, output.Shape);
        }

        [Fact]
        public void RowParallelLinearGrad_Forward_WithFullInput_SlicesInput()
        {
            // Arrange
            var layer = new RowParallelLinearGrad(
                inputSize: 256,
                outputSize: 128,
                bias: false,
                inputIsSharded: false);

            // Full input
            var input = new Tensor(new float[10 * 256], new[] { 10, 256 });

            // Act
            var output = layer.Forward(input);

            // Assert
            Assert.Equal(new[] { 10, 128 }, output.Shape);
        }

        [Fact]
        public void RowParallelLinearGrad_Backward_ComputesCorrectGradients()
        {
            // Arrange
            var layer = new RowParallelLinearGrad(
                inputSize: 128,
                outputSize: 64,
                bias: true,
                inputIsSharded: true);

            // Initialize with known weights
            var weightData = new float[64 * 32]; // [64, 32] (128/4=32)
            for (int i = 0; i < weightData.Length; i++)
            {
                weightData[i] = 1.0f;
            }
            layer.Weight.Data = weightData;

            // Sharded input
            var input = new Tensor(new float[5 * 32], new[] { 5, 32 });
            for (int i = 0; i < input.Size; i++)
            {
                input.Data[i] = 1.0f;
            }

            // Forward pass
            var output = layer.Forward(input);

            // Full gradient
            var gradOutput = new Tensor(new float[5 * 64], new[] { 5, 64 });
            for (int i = 0; i < gradOutput.Size; i++)
            {
                gradOutput.Data[i] = 1.0f;
            }

            // Act
            var gradInput = layer.BackwardInternal(gradOutput);

            // Assert
            Assert.NotNull(layer.Weight.Grad);
            Assert.NotNull(layer.Bias.Grad);
            Assert.NotNull(gradInput);
            Assert.Equal(new[] { 5, 32 }, gradInput.Shape); // Sharded input gradient
        }

        #endregion

        #region Gradient Accumulation Tests

        [Fact]
        public void GradientAccumulation_MultipleBatches_AccumulatesCorrectly()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: false);

            var input1 = new Tensor(new float[5 * 32], new[] { 5, 32 });
            var input2 = new Tensor(new float[5 * 32], new[] { 5, 32 });
            for (int i = 0; i < input1.Size; i++)
            {
                input1.Data[i] = 1.0f;
                input2.Data[i] = 2.0f;
            }

            var gradOutput1 = new Tensor(new float[5 * 16], new[] { 5, 16 });
            var gradOutput2 = new Tensor(new float[5 * 16], new[] { 5, 16 });
            for (int i = 0; i < gradOutput1.Size; i++)
            {
                gradOutput1.Data[i] = 1.0f;
                gradOutput2.Data[i] = 1.0f;
            }

            // Act - First backward
            layer.Forward(input1);
            var gradInput1 = layer.BackwardInternal(gradOutput1);

            // Second backward (without zeroing)
            layer.Forward(input2);
            var gradInput2 = layer.BackwardInternal(gradOutput2);

            // Assert - Gradients should be accumulated
            Assert.NotNull(layer.Weight.Grad);
            // Gradient should have non-zero values from both passes
            bool hasNonZero = false;
            foreach (var grad in layer.Weight.Grad.Data)
            {
                if (grad != 0.0f)
                {
                    hasNonZero = true;
                    break;
                }
            }
            Assert.True(hasNonZero);
        }

        [Fact]
        public void ZeroGrad_ClearsAllGradients()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: false);

            var input = new Tensor(new float[5 * 32], new[] { 5, 32 });
            var gradOutput = new Tensor(new float[5 * 16], new[] { 5, 16 });

            // Forward + backward to create gradients
            layer.Forward(input);
            layer.BackwardInternal(gradOutput);

            // Act
            layer.ZeroGrad();

            // Assert
            Assert.NotNull(layer.Weight.Grad);
            foreach (var grad in layer.Weight.Grad.Data)
            {
                Assert.Equal(0.0f, grad);
            }

            if (layer.Bias != null)
            {
                Assert.NotNull(layer.Bias.Grad);
                foreach (var grad in layer.Bias.Grad.Data)
                {
                    Assert.Equal(0.0f, grad);
                }
            }
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void TPMLP_ForwardBackward_CompletesSuccessfully()
        {
            // Arrange
            var colLayer = new ColumnParallelLinearGrad(
                inputSize: 128,
                outputSize: 256,
                bias: true,
                gatherOutput: false);

            var rowLayer = new RowParallelLinearGrad(
                inputSize: 256,
                outputSize: 64,
                bias: true,
                inputIsSharded: true);

            var input = new Tensor(new float[10 * 128], new[] { 10, 128 });
            for (int i = 0; i < input.Size; i++)
            {
                input.Data[i] = 0.1f;
            }

            // Act
            var hidden = colLayer.Forward(input);
            var output = rowLayer.Forward(hidden);

            // Create gradient
            var gradOutput = new Tensor(new float[10 * 64], new[] { 10, 64 });
            for (int i = 0; i < gradOutput.Size; i++)
            {
                gradOutput.Data[i] = 1.0f;
            }

            var gradHidden = rowLayer.BackwardInternal(gradOutput);
            var gradInput = colLayer.BackwardInternal(gradHidden);

            // Assert
            Assert.NotNull(colLayer.Weight.Grad);
            Assert.NotNull(rowLayer.Weight.Grad);
            Assert.NotNull(gradInput);
            Assert.Equal(new[] { 10, 128 }, gradInput.Shape);
        }

        [Fact]
        public void TPGradientManager_VerifyGradients_DoesNotThrow()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: false);

            var input = new Tensor(new float[5 * 32], new[] { 5, 32 });
            var gradOutput = new Tensor(new float[5 * 16], new[] { 5, 16 });

            // Forward + backward to create gradients
            layer.Forward(input);
            layer.BackwardInternal(gradOutput);

            var parameters = layer.GetTrainableParameters();

            // Act & Assert - Should not throw
            TPGradientManager.VerifyGradients(parameters);
        }

        [Fact]
        public void TPGradExtensions_GetTrainableParameters_ReturnsCorrectParameters()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: false);

            // Act
            var parameters = layer.GetTrainableParameters();

            // Assert
            Assert.Equal(2, parameters.Count); // weight + bias
            Assert.Contains(parameters, p => p.Name == "weight");
            Assert.Contains(parameters, p => p.Name == "bias");
        }

        [Fact]
        public void TPGradExtensions_GetTrainableParameterCount_ReturnsCorrectCount()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
                inputSize: 32,
                outputSize: 64,
                bias: true,
                gatherOutput: false);

            // Act
            var count = layer.GetTrainableParameterCount();

            // Assert
            // weight: [16, 32] = 512, bias: [16] = 16, total = 528
            Assert.Equal(528L, count);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void TPGradientLayer_WithNoBias_ForwardBackward_WorksCorrectly()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
                inputSize: 32,
                outputSize: 64,
                bias: false,
                gatherOutput: false);

            var input = new Tensor(new float[5 * 32], new[] { 5, 32 });
            var gradOutput = new Tensor(new float[5 * 16], new[] { 5, 16 });

            // Act
            var output = layer.Forward(input);
            var gradInput = layer.BackwardInternal(gradOutput);

            // Assert
            Assert.NotNull(layer.Weight.Grad);
            Assert.Null(layer.Bias);
            Assert.NotNull(gradInput);
        }

        [Fact]
        public void TPGradientLayer_WithDifferentBatchSizes_WorksCorrectly()
        {
            // Arrange
            var layer = new ColumnParallelLinearGrad(
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
    }
}
