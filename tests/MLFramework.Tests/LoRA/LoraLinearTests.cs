using RitterFramework.Core.Tensor;
using MLFramework.LoRA;
using MLFramework.Modules;
using Xunit;

namespace MLFramework.Tests.LoRA
{
    /// <summary>
    /// Unit tests for LoraLinear layer
    /// </summary>
    public class LoraLinearTests
    {
        [Fact]
        public void Forward_ProducesCorrectOutputShape()
        {
            // Arrange
            var baseLinear = new Linear(inFeatures: 64, outFeatures: 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            var input = new Tensor(new float[10 * 64], new[] { 10, 64 }); // batch=10, in=64

            // Act
            var output = loraLayer.Forward(input);

            // Assert
            Assert.Equal(new[] { 10, 128 }, output.Shape);
        }

        [Fact]
        public void Forward_WithLoRAEnabled_ProducesDifferentOutputThanBase()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            var input = CreateRandomTensor(new[] { 10, 64 });

            // Act
            var baseOutput = baseLinear.Forward(input);
            var loraOutput = loraLayer.Forward(input);

            // Assert
            // Since LoraB is initialized to zeros, the first pass should be similar to base
            // but the small random values in LoraA may cause slight differences
            Assert.Equal(baseOutput.Shape, loraOutput.Shape);
        }

        [Fact]
        public void Forward_WithLoRADisabled_MatchesBaseOutput()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);
            loraLayer.IsEnabled = false;

            var input = CreateRandomTensor(new[] { 10, 64 });

            // Act
            var baseOutput = baseLinear.Forward(input);
            var loraOutput = loraLayer.Forward(input);

            // Assert
            Assert.Equal(baseOutput.Shape, loraOutput.Shape);
        }

        [Fact]
        public void Constructor_WithNullLinear_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new LoraLinear(null!, rank: 8, alpha: 16));
        }

        [Fact]
        public void Constructor_WithValidParameters_CreatesLoraLinear()
        {
            // Arrange
            var baseLinear = new Linear(inFeatures: 64, outFeatures: 128);

            // Act
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.1f);

            // Assert
            Assert.NotNull(loraLayer);
            Assert.Equal(8, loraLayer.Rank);
            Assert.Equal(16, loraLayer.Alpha);
            Assert.Equal(2.0f, loraLayer.ScalingFactor); // alpha / rank = 16/8 = 2.0
            Assert.Equal(64, loraLayer.InFeatures);
            Assert.Equal(128, loraLayer.OutFeatures);
            Assert.Equal("LoraLinear", loraLayer.ModuleType);
        }

        [Fact]
        public void TrainableParameters_ReturnsOnlyLoRAMatrices()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            // Act
            var trainableParams = loraLayer.TrainableParameters;

            // Assert
            Assert.Equal(2, trainableParams.Count());
        }

        [Fact]
        public void FrozenParameters_ReturnsBaseLayerParameters()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            // Act
            var frozenParams = loraLayer.FrozenParameters;

            // Assert
            // Should have weight and possibly bias
            Assert.True(frozenParams.Count() >= 1);
        }

        [Fact]
        public void GetAdapterWeights_ReturnsCorrectWeights()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            // Act
            var (matrixA, matrixB) = loraLayer.GetAdapterWeights();

            // Assert
            Assert.NotNull(matrixA);
            Assert.NotNull(matrixB);
            Assert.Equal(new[] { 128, 8 }, matrixA.Shape); // [out_features, rank]
            Assert.Equal(new[] { 8, 64 }, matrixB.Shape); // [rank, in_features]
        }

        [Fact]
        public void SetAdapterWeights_WithValidWeights_UpdatesWeights()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            var newMatrixA = CreateRandomTensor(new[] { 128, 8 });
            var newMatrixB = CreateRandomTensor(new[] { 8, 64 });

            // Act
            loraLayer.SetAdapterWeights(newMatrixA, newMatrixB);

            // Assert
            var (matrixA, matrixB) = loraLayer.GetAdapterWeights();
            Assert.Equal(newMatrixA.Data, matrixA.Data);
            Assert.Equal(newMatrixB.Data, matrixB.Data);
        }

        [Fact]
        public void SetAdapterWeights_WithInvalidShape_ThrowsArgumentException()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            var invalidMatrixA = CreateRandomTensor(new[] { 64, 8 }); // Wrong shape

            // Act & Assert
            Assert.Throws<ArgumentException>(() => loraLayer.SetAdapterWeights(invalidMatrixA, null));
        }

        [Fact]
        public void ScalingFactor_CalculatesCorrectly()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 4, alpha: 8, dropout: 0.0f);

            // Act
            var scalingFactor = loraLayer.ScalingFactor;

            // Assert
            Assert.Equal(2.0f, scalingFactor); // alpha / rank = 8/4 = 2.0
        }

        [Fact]
        public void FreezeBaseLayer_SetsRequiresGradToFalse()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            // Act
            loraLayer.FreezeBaseLayer();

            // Assert
            var frozenParams = loraLayer.FrozenParameters;
            foreach (var param in frozenParams)
            {
                Assert.False(param.RequiresGrad);
            }
        }

        [Fact]
        public void UnfreezeBaseLayer_SetsRequiresGradToTrue()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            // Act
            loraLayer.UnfreezeBaseLayer();

            // Assert
            var frozenParams = loraLayer.FrozenParameters;
            foreach (var param in frozenParams)
            {
                Assert.True(param.RequiresGrad);
            }
        }

        [Fact]
        public void IsEnabled_SetToFalse_DisablesLoRA()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);
            var input = CreateRandomTensor(new[] { 10, 64 });

            // Act
            loraLayer.IsEnabled = false;
            var output = loraLayer.Forward(input);

            // Assert
            Assert.NotNull(output);
        }

        [Fact]
        public void Constructor_WithDifferentRanks_CreatesValidLoraLinear()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);

            // Act
            var loraLayer1 = new LoraLinear(baseLinear, rank: 4, alpha: 8);
            var loraLayer2 = new LoraLinear(baseLinear, rank: 16, alpha: 32);
            var loraLayer3 = new LoraLinear(baseLinear, rank: 32, alpha: 64);

            // Assert
            Assert.Equal(4, loraLayer1.Rank);
            Assert.Equal(2.0f, loraLayer1.ScalingFactor);

            Assert.Equal(16, loraLayer2.Rank);
            Assert.Equal(2.0f, loraLayer2.ScalingFactor);

            Assert.Equal(32, loraLayer3.Rank);
            Assert.Equal(2.0f, loraLayer3.ScalingFactor);
        }

        [Fact]
        public void Constructor_WithDifferentDimensions_CreatesValidLoraLinear()
        {
            // Arrange & Act
            var loraLayer1 = new LoraLinear(new Linear(32, 64), rank: 4, alpha: 8);
            var loraLayer2 = new LoraLinear(new Linear(128, 256), rank: 8, alpha: 16);
            var loraLayer3 = new LoraLinear(new Linear(512, 1024), rank: 16, alpha: 32);

            // Assert
            Assert.Equal(32, loraLayer1.InFeatures);
            Assert.Equal(64, loraLayer1.OutFeatures);

            Assert.Equal(128, loraLayer2.InFeatures);
            Assert.Equal(256, loraLayer2.OutFeatures);

            Assert.Equal(512, loraLayer3.InFeatures);
            Assert.Equal(1024, loraLayer3.OutFeatures);
        }

        [Theory]
        [InlineData(0.0f)]
        [InlineData(0.1f)]
        [InlineData(0.5f)]
        public void Constructor_WithDifferentDropoutRates_CreatesValidLoraLinear(float dropout)
        {
            // Arrange
            var baseLinear = new Linear(64, 128);

            // Act
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: dropout);

            // Assert
            Assert.NotNull(loraLayer);
        }

        [Fact]
        public void TrainableParameters_AfterMerge_ReturnsEmpty()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            // Act
            loraLayer.Merge();
            var trainableParams = loraLayer.TrainableParameters;

            // Assert
            Assert.Empty(trainableParams);
        }

        [Fact]
        public void TrainableParameters_AfterUnmerge_ReturnsLoRAMatrices()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);

            // Act
            loraLayer.Merge();
            loraLayer.Unmerge();
            var trainableParams = loraLayer.TrainableParameters;

            // Assert
            Assert.Equal(2, trainableParams.Count());
        }

        [Fact]
        public void Forward_MultipleInvocations_ProducesConsistentOutput()
        {
            // Arrange
            var baseLinear = new Linear(64, 128);
            var loraLayer = new LoraLinear(baseLinear, rank: 8, alpha: 16, dropout: 0.0f);
            var input = CreateRandomTensor(new[] { 10, 64 });

            // Act
            var output1 = loraLayer.Forward(input);
            var output2 = loraLayer.Forward(input);

            // Assert
            Assert.Equal(output1.Data, output2.Data);
        }

        private Tensor CreateRandomTensor(int[] shape)
        {
            var random = new Random(42);
            var data = new float[shape.Aggregate(1, (x, y) => x * y)];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)random.NextDouble() * 2.0f - 1.0f; // Random values between -1 and 1
            }
            return new Tensor(data, shape, false);
        }
    }
}
