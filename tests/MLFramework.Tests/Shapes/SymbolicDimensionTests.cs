using Xunit;
using MLFramework.Shapes;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for SymbolicDimension class.
    /// </summary>
    public class SymbolicDimensionTests
    {
        [Fact]
        public void Create_WithValidName_CreatesInstance()
        {
            // Arrange & Act
            var dim = new SymbolicDimension("batch_size");

            // Assert
            Assert.Equal("batch_size", dim.Name);
            Assert.Null(dim.Value);
            Assert.Equal(0, dim.MinValue);
            Assert.Null(dim.MaxValue);
        }

        [Fact]
        public void Create_WithNullName_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => new SymbolicDimension(null!));
        }

        [Fact]
        public void Create_WithEmptyName_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => new SymbolicDimension(""));
        }

        [Fact]
        public void Create_WithWhitespaceName_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => new SymbolicDimension("   "));
        }

        [Fact]
        public void Create_WithValue_SetsValue()
        {
            // Arrange & Act
            var dim = new SymbolicDimension("batch_size", 32);

            // Assert
            Assert.Equal("batch_size", dim.Name);
            Assert.Equal(32, dim.Value);
        }

        [Fact]
        public void Create_WithMinValue_SetsMinValue()
        {
            // Arrange & Act
            var dim = new SymbolicDimension("batch_size", minValue: 1);

            // Assert
            Assert.Equal("batch_size", dim.Name);
            Assert.Equal(1, dim.MinValue);
        }

        [Fact]
        public void Create_WithMaxValue_SetsMaxValue()
        {
            // Arrange & Act
            var dim = new SymbolicDimension("batch_size", maxValue: 128);

            // Assert
            Assert.Equal("batch_size", dim.Name);
            Assert.Equal(128, dim.MaxValue);
        }

        [Fact]
        public void Create_WithNegativeMinValue_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => new SymbolicDimension("batch_size", minValue: -1));
        }

        [Fact]
        public void Create_WithInvalidMaxValue_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => new SymbolicDimension("batch_size", minValue: 100, maxValue: 50));
        }

        [Fact]
        public void Create_WithValueOutsideBounds_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => new SymbolicDimension("batch_size", value: 200, minValue: 10, maxValue: 100));
        }

        [Fact]
        public void Create_WithBoundedValue_SetsAllProperties()
        {
            // Arrange & Act
            var dim = new SymbolicDimension("seq_len", value: 256, minValue: 1, maxValue: 512);

            // Assert
            Assert.Equal("seq_len", dim.Name);
            Assert.Equal(256, dim.Value);
            Assert.Equal(1, dim.MinValue);
            Assert.Equal(512, dim.MaxValue);
        }

        [Fact]
        public void IsKnown_WithValue_ReturnsTrue()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size", 32);

            // Act & Assert
            Assert.True(dim.IsKnown());
        }

        [Fact]
        public void IsKnown_WithNullValue_ReturnsFalse()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act & Assert
            Assert.False(dim.IsKnown());
        }

        [Fact]
        public void IsBounded_WithMaxValue_ReturnsTrue()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size", maxValue: 128);

            // Act & Assert
            Assert.True(dim.IsBounded());
        }

        [Fact]
        public void IsBounded_WithNullMaxValue_ReturnsFalse()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act & Assert
            Assert.False(dim.IsBounded());
        }

        [Fact]
        public void Clone_PreservesAllProperties()
        {
            // Arrange
            var original = new SymbolicDimension("seq_len", value: 256, minValue: 1, maxValue: 512);

            // Act
            var cloned = original.Clone();

            // Assert
            Assert.NotSame(original, cloned);
            Assert.Equal(original.Name, cloned.Name);
            Assert.Equal(original.Value, cloned.Value);
            Assert.Equal(original.MinValue, cloned.MinValue);
            Assert.Equal(original.MaxValue, cloned.MaxValue);
        }

        [Fact]
        public void WithConstraints_UpdatesMinConstraint()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act
            var constrained = dim.WithConstraints(min: 1, max: null);

            // Assert
            Assert.Equal(1, constrained.MinValue);
            Assert.Null(dim.MinValue); // Original unchanged due to immutability
        }

        [Fact]
        public void WithConstraints_UpdatesMaxConstraint()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act
            var constrained = dim.WithConstraints(min: null, max: 128);

            // Assert
            Assert.Equal(128, constrained.MaxValue);
            Assert.Null(dim.MaxValue); // Original unchanged
        }

        [Fact]
        public void WithConstraints_UpdatesBothConstraints()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size", value: 64);

            // Act
            var constrained = dim.WithConstraints(min: 1, max: 128);

            // Assert
            Assert.Equal(1, constrained.MinValue);
            Assert.Equal(128, constrained.MaxValue);
            Assert.Equal(0, dim.MinValue); // Original unchanged
        }

        [Fact]
        public void WithConstraints_WithInvalidBounds_ThrowsArgumentException()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => dim.WithConstraints(min: 100, max: 50));
        }

        [Fact]
        public void Equals_SameProperties_ReturnsTrue()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", value: 32, minValue: 1, maxValue: 128);
            var dim2 = new SymbolicDimension("batch_size", value: 32, minValue: 1, maxValue: 128);

            // Act & Assert
            Assert.True(dim1.Equals(dim2));
        }

        [Fact]
        public void Equals_DifferentName_ReturnsFalse()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", value: 32);
            var dim2 = new SymbolicDimension("seq_len", value: 32);

            // Act & Assert
            Assert.False(dim1.Equals(dim2));
        }

        [Fact]
        public void Equals_DifferentValue_ReturnsFalse()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", value: 32);
            var dim2 = new SymbolicDimension("batch_size", value: 64);

            // Act & Assert
            Assert.False(dim1.Equals(dim2));
        }

        [Fact]
        public void Equals_DifferentMinValue_ReturnsFalse()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", minValue: 0);
            var dim2 = new SymbolicDimension("batch_size", minValue: 1);

            // Act & Assert
            Assert.False(dim1.Equals(dim2));
        }

        [Fact]
        public void Equals_DifferentMaxValue_ReturnsFalse()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", maxValue: 128);
            var dim2 = new SymbolicDimension("batch_size", maxValue: 256);

            // Act & Assert
            Assert.False(dim1.Equals(dim2));
        }

        [Fact]
        public void Equals_Null_ReturnsFalse()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act & Assert
            Assert.False(dim.Equals(null));
        }

        [Fact]
        public void Equals_SameInstance_ReturnsTrue()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act & Assert
            Assert.True(dim.Equals(dim));
        }

        [Fact]
        public void EqualityOperator_SameProperties_ReturnsTrue()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", value: 32);
            var dim2 = new SymbolicDimension("batch_size", value: 32);

            // Act & Assert
            Assert.True(dim1 == dim2);
        }

        [Fact]
        public void InequalityOperator_DifferentProperties_ReturnsTrue()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", value: 32);
            var dim2 = new SymbolicDimension("batch_size", value: 64);

            // Act & Assert
            Assert.True(dim1 != dim2);
        }

        [Fact]
        public void GetHashCode_SameProperties_ReturnsSameHashCode()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", value: 32, minValue: 1, maxValue: 128);
            var dim2 = new SymbolicDimension("batch_size", value: 32, minValue: 1, maxValue: 128);

            // Act
            var hash1 = dim1.GetHashCode();
            var hash2 = dim2.GetHashCode();

            // Assert
            Assert.Equal(hash1, hash2);
        }

        [Fact]
        public void GetHashCode_DifferentProperties_ReturnsDifferentHashCode()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", value: 32);
            var dim2 = new SymbolicDimension("seq_len", value: 32);

            // Act
            var hash1 = dim1.GetHashCode();
            var hash2 = dim2.GetHashCode();

            // Assert
            Assert.NotEqual(hash1, hash2);
        }

        [Fact]
        public void ToString_UnknownDimension_ReturnsNameWithUnboundedRange()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act & Assert
            Assert.Equal("batch_size[0..∞]", dim.ToString());
        }

        [Fact]
        public void ToString_KnownDimension_ReturnsNameWithValue()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size", 32);

            // Act & Assert
            Assert.Equal("batch_size=32", dim.ToString());
        }

        [Fact]
        public void ToString_BoundedDimension_ReturnsNameWithRange()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size", minValue: 1, maxValue: 128);

            // Act & Assert
            Assert.Equal("batch_size[1..128]", dim.ToString());
        }

        [Fact]
        public void ToString_RangeOnly_ReturnsNameWithMinAndInfinity()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size", minValue: 1);

            // Act & Assert
            Assert.Equal("batch_size[1..∞]", dim.ToString());
        }
    }
}
