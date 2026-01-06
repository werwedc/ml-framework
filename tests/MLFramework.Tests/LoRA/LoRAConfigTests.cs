using Xunit;
using MLFramework.LoRA;

namespace MLFramework.Tests.LoRA
{
    public class LoRAConfigTests
    {
        [Fact]
        public void DefaultValues_AreCorrect()
        {
            // Act
            var config = new LoRAConfig();

            // Assert
            Assert.Equal(8, config.Rank);
            Assert.Equal(16.0f, config.Alpha);
            Assert.Null(config.TargetModules);
            Assert.False(config.UseBias);
            Assert.Equal(LoRAInitializationStrategy.Standard, config.Initialization);
            Assert.Equal(0.0f, config.Dropout);
            Assert.False(config.UseFusedKernels);
            Assert.Null(config.TargetLayerTypes);
        }

        [Theory]
        [InlineData(-1)]
        [InlineData(0)]
        public void Constructor_WithInvalidRank_ThrowsException(int invalidRank)
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new LoRAConfig(rank: invalidRank));
        }

        [Theory]
        [InlineData(-1.0f)]
        [InlineData(0.0f)]
        public void Constructor_WithInvalidAlpha_ThrowsException(float invalidAlpha)
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new LoRAConfig(alpha: invalidAlpha));
        }

        [Fact]
        public void Constructor_WithInvalidDropout_ThrowsException()
        {
            // Arrange
            var config = new LoRAConfig();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => config.Dropout = -0.1f);
            Assert.Throws<ArgumentException>(() => config.Dropout = 1.0f);
        }

        [Fact]
        public void Constructor_WithValidParameters_SetsValuesCorrectly()
        {
            // Arrange
            int rank = 16;
            float alpha = 32.0f;

            // Act
            var config = new LoRAConfig(rank, alpha);

            // Assert
            Assert.Equal(rank, config.Rank);
            Assert.Equal(alpha, config.Alpha);
        }

        [Fact]
        public void TargetModules_CanBeSet()
        {
            // Arrange
            var config = new LoRAConfig();
            var modules = new[] { "attn.q_proj", "attn.v_proj", "mlp.fc_in" };

            // Act
            config.TargetModules = modules;

            // Assert
            Assert.Equal(modules, config.TargetModules);
        }

        [Fact]
        public void TargetLayerTypes_CanBeSet()
        {
            // Arrange
            var config = new LoRAConfig();
            var types = new[] { "Linear", "Conv2d" };

            // Act
            config.TargetLayerTypes = types;

            // Assert
            Assert.Equal(types, config.TargetLayerTypes);
        }

        [Theory]
        [InlineData(0.0f)]
        [InlineData(0.1f)]
        [InlineData(0.5f)]
        [InlineData(0.9f)]
        public void Dropout_ValidValues_AreSetCorrectly(float validDropout)
        {
            // Arrange
            var config = new LoRAConfig();

            // Act
            config.Dropout = validDropout;

            // Assert
            Assert.Equal(validDropout, config.Dropout);
        }

        [Theory]
        [InlineData(LoRAInitializationStrategy.Standard)]
        [InlineData(LoRAInitializationStrategy.Xavier)]
        [InlineData(LoRAInitializationStrategy.Zero)]
        public void InitializationStrategy_CanBeSet(LoRAInitializationStrategy strategy)
        {
            // Arrange
            var config = new LoRAConfig();

            // Act
            config.Initialization = strategy;

            // Assert
            Assert.Equal(strategy, config.Initialization);
        }

        [Fact]
        public void UseFusedKernels_CanBeSet()
        {
            // Arrange
            var config = new LoRAConfig();

            // Act
            config.UseFusedKernels = true;

            // Assert
            Assert.True(config.UseFusedKernels);
        }

        [Fact]
        public void UseBias_CanBeSet()
        {
            // Arrange
            var config = new LoRAConfig();

            // Act
            config.UseBias = true;

            // Assert
            Assert.True(config.UseBias);
        }
    }
}
