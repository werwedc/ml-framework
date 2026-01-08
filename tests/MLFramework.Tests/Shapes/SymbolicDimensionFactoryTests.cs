using Xunit;
using MLFramework.Shapes;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for SymbolicDimensionFactory class.
    /// </summary>
    public class SymbolicDimensionFactoryTests
    {
        [Fact]
        public void Create_WithName_ReturnsSymbolicDimension()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.Create("batch_size");

            // Assert
            Assert.NotNull(dim);
            Assert.Equal("batch_size", dim.Name);
            Assert.Null(dim.Value);
        }

        [Fact]
        public void Create_WithNameAndValue_ReturnsKnownDimension()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.Create("batch_size", 32);

            // Assert
            Assert.Equal("batch_size", dim.Name);
            Assert.Equal(32, dim.Value);
            Assert.True(dim.IsKnown());
        }

        [Fact]
        public void Create_WithoutValue_ReturnsUnknownDimension()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.Create("seq_len");

            // Assert
            Assert.Equal("seq_len", dim.Name);
            Assert.Null(dim.Value);
            Assert.False(dim.IsKnown());
        }

        [Fact]
        public void CreateBounded_WithValidBounds_ReturnsBoundedDimension()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.CreateBounded("batch_size", 1, 128);

            // Assert
            Assert.Equal("batch_size", dim.Name);
            Assert.Equal(1, dim.MinValue);
            Assert.Equal(128, dim.MaxValue);
            Assert.True(dim.IsBounded());
        }

        [Fact]
        public void CreateBounded_WithInvalidBounds_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                SymbolicDimensionFactory.CreateBounded("batch_size", 100, 50));
        }

        [Fact]
        public void CreateRange_WithMinValue_ReturnsRangeWithUnboundedMax()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.CreateRange("batch_size", 1);

            // Assert
            Assert.Equal("batch_size", dim.Name);
            Assert.Equal(1, dim.MinValue);
            Assert.Null(dim.MaxValue);
            Assert.False(dim.IsBounded());
        }

        [Fact]
        public void CreateRange_WithNegativeMin_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() =>
                SymbolicDimensionFactory.CreateRange("batch_size", -1));
        }

        [Fact]
        public void CreateKnown_WithValue_ReturnsKnownDimension()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.CreateKnown("features", 512);

            // Assert
            Assert.Equal("features", dim.Name);
            Assert.Equal(512, dim.Value);
            Assert.True(dim.IsKnown());
        }

        [Fact]
        public void BatchSize_ReturnsDimensionWithNameBatchSize()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.BatchSize();

            // Assert
            Assert.Equal("batch_size", dim.Name);
            Assert.Null(dim.Value);
            Assert.Equal(1, dim.MinValue);
            Assert.Null(dim.MaxValue);
        }

        [Fact]
        public void SequenceLength_ReturnsDimensionWithNameSeqLen()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.SequenceLength();

            // Assert
            Assert.Equal("seq_len", dim.Name);
            Assert.Null(dim.Value);
            Assert.Equal(1, dim.MinValue);
            Assert.Null(dim.MaxValue);
        }

        [Fact]
        public void Features_WithValue_ReturnsKnownDimension()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.Features(256);

            // Assert
            Assert.Equal("features", dim.Name);
            Assert.Equal(256, dim.Value);
            Assert.True(dim.IsKnown());
        }

        [Fact]
        public void Channels_WithValue_ReturnsKnownDimension()
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.Channels(3);

            // Assert
            Assert.Equal("channels", dim.Name);
            Assert.Equal(3, dim.Value);
            Assert.True(dim.IsKnown());
        }

        [Fact]
        public void MultipleBatchSizeCalls_CreateEqualDimensions()
        {
            // Arrange
            var dim1 = SymbolicDimensionFactory.BatchSize();
            var dim2 = SymbolicDimensionFactory.BatchSize();

            // Act & Assert
            Assert.Equal(dim1, dim2);
            Assert.Equal(dim1.GetHashCode(), dim2.GetHashCode());
        }

        [Fact]
        public void MultipleSequenceLengthCalls_CreateEqualDimensions()
        {
            // Arrange
            var dim1 = SymbolicDimensionFactory.SequenceLength();
            var dim2 = SymbolicDimensionFactory.SequenceLength();

            // Act & Assert
            Assert.Equal(dim1, dim2);
            Assert.Equal(dim1.GetHashCode(), dim2.GetHashCode());
        }

        [Fact]
        public void FeaturesCallsWithSameValue_CreateEqualDimensions()
        {
            // Arrange
            var dim1 = SymbolicDimensionFactory.Features(256);
            var dim2 = SymbolicDimensionFactory.Features(256);

            // Act & Assert
            Assert.Equal(dim1, dim2);
            Assert.Equal(dim1.GetHashCode(), dim2.GetHashCode());
        }

        [Fact]
        public void FeaturesCallsWithDifferentValue_CreateDifferentDimensions()
        {
            // Arrange
            var dim1 = SymbolicDimensionFactory.Features(256);
            var dim2 = SymbolicDimensionFactory.Features(512);

            // Act & Assert
            Assert.NotEqual(dim1, dim2);
        }

        [Fact]
        public void ChannelsCallsWithSameValue_CreateEqualDimensions()
        {
            // Arrange
            var dim1 = SymbolicDimensionFactory.Channels(3);
            var dim2 = SymbolicDimensionFactory.Channels(3);

            // Act & Assert
            Assert.Equal(dim1, dim2);
            Assert.Equal(dim1.GetHashCode(), dim2.GetHashCode());
        }

        [Fact]
        public void FactoryMethods_CreateImmutableInstances()
        {
            // Arrange
            var dim = SymbolicDimensionFactory.Create("test", 32);

            // Act - Try to modify properties (not possible as they're read-only)
            // This test verifies the immutability design

            // Assert - Should not be able to modify the instance
            Assert.Equal("test", dim.Name);
            Assert.Equal(32, dim.Value);
        }

        [Theory]
        [InlineData("batch_size", null)]
        [InlineData("seq_len", null)]
        [InlineData("features", 512)]
        [InlineData("channels", 3)]
        [InlineData("hidden_dim", 768)]
        public void Create_CreatesDimensionWithCorrectName(string name, int? value)
        {
            // Arrange & Act
            var dim = SymbolicDimensionFactory.Create(name, value);

            // Assert
            Assert.Equal(name, dim.Name);
            Assert.Equal(value, dim.Value);
        }
    }
}
