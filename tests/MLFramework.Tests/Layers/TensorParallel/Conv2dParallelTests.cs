using MLFramework.Layers.TensorParallel;
using MLFramework.Tests.Distributed;
using RitterFramework.Core.Tensor;
using System;
using System.Linq;
using Xunit;

namespace MLFramework.Tests.Layers.TensorParallel
{
    /// <summary>
    /// Unit tests for tensor-parallel Conv2d layers.
    /// </summary>
    public class Conv2dParallelTests : IDisposable
    {
        private readonly MockProcessGroup _mockProcessGroup;

        public Conv2dParallelTests()
        {
            _mockProcessGroup = new MockProcessGroup(rank: 0, worldSize: 4);
            TensorParallel.Initialize(_mockProcessGroup);
        }

        public void Dispose()
        {
            _mockProcessGroup.Dispose();
        }

        #region Conv2dOutputParallel Tests

        [Fact]
        public void Conv2dOutputParallel_Constructor_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new Conv2dOutputParallel(
                inChannels: 3,
                outChannels: 16,
                kernelSize: 3,
                gatherOutput: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(3, layer.InChannels);
            Assert.Equal(16, layer.OutChannels);
            Assert.Equal(3, layer.KernelSize);
            Assert.NotNull(layer.Weight);
            Assert.NotNull(layer.Bias);

            // Weight shape: [out_channels / world_size, in_channels, kernel_h, kernel_w]
            // [16/4=4, 3, 3, 3]
            Assert.Equal(new[] { 4, 3, 3, 3 }, layer.Weight.Shape);
            // Bias shape: [out_channels / world_size]
            // [16/4=4]
            Assert.Equal(new[] { 4 }, layer.Bias!.Shape);
        }

        [Fact]
        public void Conv2dOutputParallel_Constructor_WithoutBias_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new Conv2dOutputParallel(
                inChannels: 3,
                outChannels: 16,
                kernelSize: 3,
                gatherOutput: false,
                bias: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Null(layer.Bias);
        }

        [Fact]
        public void Conv2dOutputParallel_Constructor_OutChannelsNotDivisibleByWorldSize_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new Conv2dOutputParallel(
                    inChannels: 3,
                    outChannels: 15, // Not divisible by 4
                    kernelSize: 3,
                    gatherOutput: false));
        }

        [Fact]
        public void Conv2dOutputParallel_Forward_ProducesCorrectShardedOutputShape()
        {
            // Arrange
            var layer = new Conv2dOutputParallel(
                inChannels: 3,
                outChannels: 16,
                kernelSize: 3,
                gatherOutput: false,
                padding: 1);
            var input = new Tensor(new float[2 * 3 * 8 * 8], new[] { 2, 3, 8, 8 });

            // Act
            var output = layer.Forward(input);

            // Assert
            // Output shape: [batch, out_channels / world_size, height_out, width_out]
            // [2, 16/4=4, 8, 8] (with padding=1, kernel=3, stride=1)
            Assert.Equal(new[] { 2, 4, 8, 8 }, output.Shape);
        }

        [Fact]
        public void Conv2dOutputParallel_Forward_WithDifferentParameters_ProducesCorrectShape()
        {
            // Arrange
            var layer = new Conv2dOutputParallel(
                inChannels: 3,
                outChannels: 32,
                kernelSize: 5,
                stride: 2,
                padding: 2,
                gatherOutput: false);
            var input = new Tensor(new float[2 * 3 * 16 * 16], new[] { 2, 3, 16, 16 });

            // Act
            var output = layer.Forward(input);

            // Assert
            // Output shape: [batch, out_channels / world_size, height_out, width_out]
            // height_out = (16 + 2*2 - 5) / 2 + 1 = 8
            // [2, 32/4=8, 8, 8]
            Assert.Equal(new[] { 2, 8, 8, 8 }, output.Shape);
        }

        [Fact]
        public void Conv2dOutputParallel_WeightShape_IsCorrect()
        {
            // Arrange & Act
            var layer = new Conv2dOutputParallel(
                inChannels: 64,
                outChannels: 128,
                kernelSize: 3);

            // Assert
            // Weight: [out_channels / world_size, in_channels, kernel_h, kernel_w]
            Assert.Equal(new[] { 32, 64, 3, 3 }, layer.Weight.Shape);
        }

        [Fact]
        public void Conv2dOutputParallel_BiasShape_IsCorrect()
        {
            // Arrange & Act
            var layer = new Conv2dOutputParallel(
                inChannels: 64,
                outChannels: 128,
                kernelSize: 3,
                bias: true);

            // Assert
            // Bias: [out_channels / world_size]
            Assert.Equal(new[] { 32 }, layer.Bias!.Shape);
        }

        [Fact]
        public void Conv2dOutputParallel_ModuleType_IsCorrect()
        {
            // Arrange & Act
            var layer = new Conv2dOutputParallel(3, 16, 3);

            // Assert
            Assert.Equal("Conv2dOutputParallel", layer.ModuleType);
        }

        #endregion

        #region Conv2dInputParallel Tests

        [Fact]
        public void Conv2dInputParallel_Constructor_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new Conv2dInputParallel(
                inChannels: 64,
                outChannels: 16,
                kernelSize: 3);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(64, layer.InChannels);
            Assert.Equal(16, layer.OutChannels);
            Assert.Equal(3, layer.KernelSize);
            Assert.NotNull(layer.Weight);
            Assert.NotNull(layer.Bias);

            // Weight shape: [out_channels, in_channels / world_size, kernel_h, kernel_w]
            // [16, 64/4=16, 3, 3]
            Assert.Equal(new[] { 16, 16, 3, 3 }, layer.Weight.Shape);
            // Bias shape: [out_channels] (not sharded)
            // [16]
            Assert.Equal(new[] { 16 }, layer.Bias!.Shape);
        }

        [Fact]
        public void Conv2dInputParallel_Constructor_WithoutBias_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = new Conv2dInputParallel(
                inChannels: 64,
                outChannels: 16,
                kernelSize: 3,
                bias: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Null(layer.Bias);
        }

        [Fact]
        public void Conv2dInputParallel_Constructor_InChannelsNotDivisibleByWorldSize_ThrowsException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new Conv2dInputParallel(
                    inChannels: 63, // Not divisible by 4
                    outChannels: 16,
                    kernelSize: 3));
        }

        [Fact]
        public void Conv2dInputParallel_Forward_ProducesCorrectOutputShape()
        {
            // Arrange
            var layer = new Conv2dInputParallel(
                inChannels: 64,
                outChannels: 16,
                kernelSize: 3,
                padding: 1);
            // Input should be sharded: [batch, in_channels / world_size, h, w]
            var input = new Tensor(new float[2 * 16 * 8 * 8], new[] { 2, 16, 8, 8 });

            // Act
            var output = layer.Forward(input);

            // Assert
            // Output shape: [batch, out_channels, height_out, width_out]
            // [2, 16, 8, 8] (with padding=1, kernel=3, stride=1)
            Assert.Equal(new[] { 2, 16, 8, 8 }, output.Shape);
        }

        [Fact]
        public void Conv2dInputParallel_Forward_WithWrongShardedInput_ThrowsException()
        {
            // Arrange
            var layer = new Conv2dInputParallel(
                inChannels: 64,
                outChannels: 16,
                kernelSize: 3);
            // Wrong input shape (not sharded properly)
            var input = new Tensor(new float[2 * 64 * 8 * 8], new[] { 2, 64, 8, 8 });

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => layer.Forward(input));
        }

        [Fact]
        public void Conv2dInputParallel_WeightShape_IsCorrect()
        {
            // Arrange & Act
            var layer = new Conv2dInputParallel(
                inChannels: 128,
                outChannels: 64,
                kernelSize: 3);

            // Assert
            // Weight: [out_channels, in_channels / world_size, kernel_h, kernel_w]
            Assert.Equal(new[] { 64, 32, 3, 3 }, layer.Weight.Shape);
        }

        [Fact]
        public void Conv2dInputParallel_BiasShape_IsNotSharded()
        {
            // Arrange & Act
            var layer = new Conv2dInputParallel(
                inChannels: 128,
                outChannels: 64,
                kernelSize: 3,
                bias: true);

            // Assert
            // Bias: [out_channels] (not sharded)
            Assert.Equal(new[] { 64 }, layer.Bias!.Shape);
        }

        [Fact]
        public void Conv2dInputParallel_ModuleType_IsCorrect()
        {
            // Arrange & Act
            var layer = new Conv2dInputParallel(64, 16, 3);

            // Assert
            Assert.Equal("Conv2dInputParallel", layer.ModuleType);
        }

        #endregion

        #region Factory Tests

        [Fact]
        public void TPConv2dFactory_CreateOutputParallel_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = TPConv2dFactory.CreateOutputParallel(
                inChannels: 3,
                outChannels: 16,
                kernelSize: 3,
                gatherOutput: false);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(3, layer.InChannels);
            Assert.Equal(16, layer.OutChannels);
        }

        [Fact]
        public void TPConv2dFactory_CreateInputParallel_CreatesValidLayer()
        {
            // Arrange & Act
            var layer = TPConv2dFactory.CreateInputParallel(
                inChannels: 64,
                outChannels: 16,
                kernelSize: 3);

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(64, layer.InChannels);
            Assert.Equal(16, layer.OutChannels);
        }

        [Fact]
        public void TPConv2dFactory_CreateBottleneckPair_CreatesValidLayers()
        {
            // Arrange & Act
            var (conv1, conv2) = TPConv2dFactory.CreateBottleneckPair(
                inChannels: 64,
                bottleneckChannels: 32,
                outChannels: 64);

            // Assert
            Assert.NotNull(conv1);
            Assert.NotNull(conv2);
            Assert.Equal(64, conv1.InChannels);
            Assert.Equal(32, conv1.OutChannels);
            Assert.Equal(32, conv2.InChannels);
            Assert.Equal(64, conv2.OutChannels);
            Assert.Null(conv1.Bias);
            Assert.Null(conv2.Bias);
        }

        #endregion

        #region Parameter and Gradient Tests

        [Fact]
        public void Conv2dOutputParallel_Parameters_ReturnsCorrectTensors()
        {
            // Arrange
            var layer = new Conv2dOutputParallel(3, 16, 3, bias: true);

            // Act
            var parameters = layer.Parameters.ToList();

            // Assert
            Assert.Equal(2, parameters.Count);
            Assert.Same(layer.Weight, parameters[0]);
            Assert.Same(layer.Bias, parameters[1]);
        }

        [Fact]
        public void Conv2dInputParallel_Parameters_ReturnsCorrectTensors()
        {
            // Arrange
            var layer = new Conv2dInputParallel(64, 16, 3, bias: true);

            // Act
            var parameters = layer.Parameters.ToList();

            // Assert
            Assert.Equal(2, parameters.Count);
            Assert.Same(layer.Weight, parameters[0]);
            Assert.Same(layer.Bias, parameters[1]);
        }

        [Fact]
        public void Conv2dOutputParallel_SetRequiresGrad_SetsFlagOnAllParameters()
        {
            // Arrange
            var layer = new Conv2dOutputParallel(3, 16, 3, bias: true);

            // Act
            layer.SetRequiresGrad(true);

            // Assert
            Assert.True(layer.Weight.RequiresGrad);
            Assert.True(layer.Bias!.RequiresGrad);

            // Act
            layer.SetRequiresGrad(false);

            // Assert
            Assert.False(layer.Weight.RequiresGrad);
            Assert.False(layer.Bias.RequiresGrad);
        }

        [Fact]
        public void Conv2dInputParallel_SetRequiresGrad_SetsFlagOnAllParameters()
        {
            // Arrange
            var layer = new Conv2dInputParallel(64, 16, 3, bias: true);

            // Act
            layer.SetRequiresGrad(true);

            // Assert
            Assert.True(layer.Weight.RequiresGrad);
            Assert.True(layer.Bias!.RequiresGrad);

            // Act
            layer.SetRequiresGrad(false);

            // Assert
            Assert.False(layer.Weight.RequiresGrad);
            Assert.False(layer.Bias.RequiresGrad);
        }

        [Fact]
        public void Conv2dOutputParallel_ApplyToParameters_AppliesActionToAllParameters()
        {
            // Arrange
            var layer = new Conv2dOutputParallel(3, 16, 3, bias: true);
            var actionCalledCount = 0;

            // Act
            layer.ApplyToParameters(_ => actionCalledCount++);

            // Assert
            Assert.Equal(2, actionCalledCount);
        }

        [Fact]
        public void Conv2dInputParallel_ApplyToParameters_AppliesActionToAllParameters()
        {
            // Arrange
            var layer = new Conv2dInputParallel(64, 16, 3, bias: true);
            var actionCalledCount = 0;

            // Act
            layer.ApplyToParameters(_ => actionCalledCount++);

            // Assert
            Assert.Equal(2, actionCalledCount);
        }

        #endregion

        #region Edge Cases

        [Theory]
        [InlineData(1)]
        [InlineData(3)]
        [InlineData(5)]
        [InlineData(7)]
        public void Conv2dOutputParallel_WithDifferentKernelSizes_WorksCorrectly(int kernelSize)
        {
            // Arrange & Act
            var layer = new Conv2dOutputParallel(3, 16, kernelSize, padding: kernelSize / 2);
            var input = new Tensor(new float[2 * 3 * 8 * 8], new[] { 2, 3, 8, 8 });
            var output = layer.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new[] { 2, 4, 8, 8 }, output.Shape);
        }

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        public void Conv2dOutputParallel_WithDifferentStrides_WorksCorrectly(int stride)
        {
            // Arrange
            var layer = new Conv2dOutputParallel(3, 16, 3, stride: stride, padding: 1);
            var input = new Tensor(new float[2 * 3 * 16 * 16], new[] { 2, 3, 16, 16 });

            // Act
            var output = layer.Forward(input);

            // Assert
            // height_out = (16 + 2*1 - 3) / stride + 1 = (15) / stride + 1
            int expectedHeight = (15 + stride - 1) / stride + 1;
            int expectedWidth = expectedHeight;
            Assert.Equal(new[] { 2, 4, expectedHeight, expectedWidth }, output.Shape);
        }

        [Fact]
        public void Conv2dOutputParallel_WithDifferentPadding_WorksCorrectly()
        {
            // Arrange
            var layer = new Conv2dOutputParallel(3, 16, 3, stride: 1, padding: 0);
            var input = new Tensor(new float[2 * 3 * 8 * 8], new[] { 2, 3, 8, 8 });

            // Act
            var output = layer.Forward(input);

            // Assert
            // height_out = (8 + 2*0 - 3) / 1 + 1 = 6
            Assert.Equal(new[] { 2, 4, 6, 6 }, output.Shape);
        }

        #endregion
    }
}
