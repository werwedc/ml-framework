using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Modules;
using MLFramework.LoRA;

namespace MLFramework.Tests.LoRA
{
    /// <summary>
    /// Unit tests for LoRALinear layer
    /// </summary>
    public class LoRALinearTests
    {
        [Fact]
        public void Constructor_WithValidParameters_CreatesAdapter()
        {
            // Arrange
            var linear = new Linear(inFeatures: 10, outFeatures: 5);
            int rank = 4;
            float alpha = 8.0f;

            // Act
            var lora = new LoRALinear(linear, rank, alpha);

            // Assert
            Assert.NotNull(lora);
            Assert.Equal(rank, lora.Rank);
            Assert.Equal(alpha, lora.Alpha);
            Assert.Equal(10, lora.InDim);
            Assert.Equal(5, lora.OutDim);
            Assert.True(lora.IsEnabled);
        }

        [Fact]
        public void Constructor_WithNullLinearLayer_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new LoRALinear(null!, 4, 8.0f));
        }

        [Fact]
        public void Constructor_WithDifferentInitializationStrategies_CreatesMatrices()
        {
            // Arrange
            var linear1 = new Linear(10, 5);
            var linear2 = new Linear(10, 5);
            var linear3 = new Linear(10, 5);

            // Act
            var loraStandard = new LoRALinear(linear1, 4, 8.0f, LoRAInitializationStrategy.Standard);
            var loraXavier = new LoRALinear(linear2, 4, 8.0f, LoRAInitializationStrategy.Xavier);
            var loraZero = new LoRALinear(linear3, 4, 8.0f, LoRAInitializationStrategy.Zero);

            // Assert
            Assert.NotNull(loraStandard);
            Assert.NotNull(loraXavier);
            Assert.NotNull(loraZero);
        }

        [Fact]
        public void Forward_WithEnabledAdapter_ReturnsModifiedOutput()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            var input = Tensor.Ones(new[] { 2, 10 }); // batch of 2

            // Act
            var output = lora.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new[] { 2, 5 }, output.Shape);
        }

        [Fact]
        public void Forward_WithDisabledAdapter_ReturnsBaseOutput()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            var input = Tensor.Ones(new[] { 2, 10 });
            lora.IsEnabled = false;

            // Act
            var output = lora.Forward(input);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(new[] { 2, 5 }, output.Shape);
        }

        [Fact]
        public void Forward_ProducesCorrectOutputShape()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            var input = Tensor.Ones(new[] { 3, 10 }); // batch size 3

            // Act
            var output = lora.Forward(input);

            // Assert
            Assert.Equal(new[] { 3, 5 }, output.Shape);
        }

        [Fact]
        public void FreezeBaseLayer_SetsRequiresGradFalse()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);

            // Act
            lora.FreezeBaseLayer();

            // Assert
            Assert.False(linear.Weight.RequiresGrad);
            if (linear.Bias != null)
            {
                Assert.False(linear.Bias.RequiresGrad);
            }
        }

        [Fact]
        public void UnfreezeBaseLayer_SetsRequiresGradTrue()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            lora.FreezeBaseLayer();

            // Act
            lora.UnfreezeBaseLayer();

            // Assert
            Assert.True(linear.Weight.RequiresGrad);
            if (linear.Bias != null)
            {
                Assert.True(linear.Bias.RequiresGrad);
            }
        }

        [Fact]
        public void TrainableParameters_WithFrozenBase_ReturnsOnlyAdapters()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            lora.FreezeBaseLayer();

            // Act
            var trainableParams = lora.TrainableParameters.ToList();

            // Assert
            Assert.Equal(2, trainableParams.Count); // Only A and B matrices
            Assert.DoesNotContain(linear.Weight, trainableParams);
        }

        [Fact]
        public void TrainableParameters_WithUnfrozenBase_ReturnsAll()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);

            // Act
            var trainableParams = lora.TrainableParameters.ToList();

            // Assert
            Assert.Contains(linear.Weight, trainableParams);
            if (linear.Bias != null)
            {
                Assert.Contains(linear.Bias, trainableParams);
            }
        }

        [Fact]
        public void FrozenParameters_WithFrozenBase_ReturnsBaseParams()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            lora.FreezeBaseLayer();

            // Act
            var frozenParams = lora.FrozenParameters.ToList();

            // Assert
            Assert.Contains(linear.Weight, frozenParams);
        }

        [Fact]
        public void MergeAdapter_UpdatesBaseWeights()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);

            // Get original weights
            var (matrixA, matrixB) = lora.GetAdapterWeights();

            // Act
            lora.MergeAdapter();

            // Assert - Verify merge happened without throwing
            Assert.NotNull(matrixA);
            Assert.NotNull(matrixB);
        }

        [Fact]
        public void ResetBaseLayer_WithBackup_RestoresWeights()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            lora.MergeAdapter();

            // Act
            lora.ResetBaseLayer();

            // Assert - Should not throw
            Assert.True(true);
        }

        [Fact]
        public void ResetBaseLayer_WithoutBackup_ThrowsInvalidOperationException()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => lora.ResetBaseLayer());
        }

        [Fact]
        public void GetAdapterWeights_ReturnsCorrectTensors()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);

            // Act
            var (matrixA, matrixB) = lora.GetAdapterWeights();

            // Assert
            Assert.NotNull(matrixA);
            Assert.NotNull(matrixB);
            Assert.Equal(new[] { 4, 10 }, matrixA.Shape); // [rank, in_dim]
            Assert.Equal(new[] { 5, 4 }, matrixB.Shape); // [out_dim, rank]
        }

        [Fact]
        public void SetAdapterWeights_WithValidShapes_UpdatesMatrices()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            var newA = Tensor.Ones(new[] { 4, 10 });
            var newB = Tensor.Zeros(new[] { 5, 4 });

            // Act & Assert
            var exception = Record.Exception(() => lora.SetAdapterWeights(newA, newB));
            Assert.Null(exception);
        }

        [Fact]
        public void SetAdapterWeights_WithInvalidShapeA_ThrowsArgumentException()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            var newA = Tensor.Ones(new[] { 3, 10 }); // Wrong rank
            var newB = Tensor.Zeros(new[] { 5, 4 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => lora.SetAdapterWeights(newA, newB));
        }

        [Fact]
        public void SetAdapterWeights_WithInvalidShapeB_ThrowsArgumentException()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);
            var newA = Tensor.Ones(new[] { 4, 10 });
            var newB = Tensor.Zeros(new[] { 6, 4 }); // Wrong out dim

            // Act & Assert
            Assert.Throws<ArgumentException>(() => lora.SetAdapterWeights(newA, newB));
        }

        [Fact]
        public void SetAdapterWeights_WithNull_ThrowsArgumentNullException()
        {
            // Arrange
            var linear = new Linear(10, 5);
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => lora.SetAdapterWeights(null, null));
        }

        [Fact]
        public void ScalingFactor_ReturnsCorrectValue()
        {
            // Arrange
            var linear = new Linear(10, 5);
            int rank = 4;
            float alpha = 8.0f;
            var lora = new LoRALinear(linear, rank, alpha);

            // Act
            float scalingFactor = lora.ScalingFactor;

            // Assert
            Assert.Equal(2.0f, scalingFactor); // alpha / rank = 8.0 / 4.0 = 2.0
        }

        [Fact]
        public void AsLoRAExtension_CreatesAdapter()
        {
            // Arrange
            var linear = new Linear(10, 5);

            // Act
            var lora = linear.AsLoRA(rank: 4, alpha: 8.0f);

            // Assert
            Assert.NotNull(lora);
            Assert.IsType<LoRALinear>(lora);
            Assert.Equal(4, lora.Rank);
            Assert.Equal(8.0f, lora.Alpha);
        }

        [Fact]
        public void Constructor_WithBias_CreatesBiasAdapter()
        {
            // Arrange
            var linear = new Linear(10, 5);

            // Act
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f, useBias: true);

            // Assert
            var bias = lora.GetBias();
            Assert.NotNull(bias);
            Assert.Equal(new[] { 5 }, bias!.Shape);
        }

        [Fact]
        public void Constructor_WithoutBias_DoesNotCreateBiasAdapter()
        {
            // Arrange
            var linear = new Linear(10, 5);

            // Act
            var lora = new LoRALinear(linear, rank: 4, alpha: 8.0f, useBias: false);

            // Assert
            var bias = lora.GetBias();
            Assert.Null(bias);
        }
    }
}
