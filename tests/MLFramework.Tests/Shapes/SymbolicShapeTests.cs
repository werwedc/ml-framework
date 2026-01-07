using Xunit;
using MLFramework.Shapes;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for SymbolicShape class.
    /// </summary>
    public class SymbolicShapeTests
    {
        [Fact]
        public void Constructor_WithDimensions_CreatesInstance()
        {
            // Arrange & Act
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("seq_len"),
                new SymbolicDimension("hidden", 512));

            // Assert
            Assert.Equal(3, shape.Rank);
            Assert.Equal("batch_size", shape.GetDimension(0).Name);
            Assert.Equal("seq_len", shape.GetDimension(1).Name);
            Assert.Equal(512, shape.GetDimension(2).Value);
        }

        [Fact]
        public void Constructor_WithNullDimensions_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => new SymbolicShape(null!));
        }

        [Fact]
        public void Constructor_WithEmptyDimensions_CreatesScalar()
        {
            // Arrange & Act
            var shape = new SymbolicShape(Array.Empty<SymbolicDimension>());

            // Assert
            Assert.Equal(0, shape.Rank);
        }

        [Fact]
        public void Constructor_WithEnumerable_CreatesInstance()
        {
            // Arrange
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20)
            };

            // Act
            var shape = new SymbolicShape(dims);

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.Equal(10, shape.GetDimension(0).Value);
            Assert.Equal(20, shape.GetDimension(1).Value);
        }

        [Fact]
        public void GetDimension_ValidIndex_ReturnsDimension()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("seq_len"));

            // Act
            var dim = shape.GetDimension(0);

            // Assert
            Assert.Equal("batch_size", dim.Name);
        }

        [Fact]
        public void GetDimension_NegativeIndex_ReturnsDimension()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("seq_len"));

            // Act
            var dim = shape.GetDimension(-1);

            // Assert
            Assert.Equal("seq_len", dim.Name);
        }

        [Fact]
        public void GetDimension_InvalidIndex_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("batch_size"));

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => shape.GetDimension(10));
        }

        [Fact]
        public void GetDimension_InvalidNegativeIndex_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("batch_size"));

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() => shape.GetDimension(-10));
        }

        [Fact]
        public void SetDimension_ValidIndex_ReturnsNewShape()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("seq_len"));

            // Act
            var newShape = shape.SetDimension(0, new SymbolicDimension("new_dim", 100));

            // Assert
            Assert.NotSame(shape, newShape);
            Assert.Equal("batch_size", shape.GetDimension(0).Name); // Original unchanged
            Assert.Equal("new_dim", newShape.GetDimension(0).Name);
        }

        [Fact]
        public void SetDimension_NegativeIndex_ReturnsNewShape()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("seq_len"));

            // Act
            var newShape = shape.SetDimension(-1, new SymbolicDimension("new_dim", 100));

            // Assert
            Assert.Equal("new_dim", newShape.GetDimension(1).Name);
        }

        [Fact]
        public void SetDimension_NullDimension_ThrowsArgumentNullException()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("batch_size"));

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => shape.SetDimension(0, null!));
        }

        [Fact]
        public void IsFullyKnown_AllDimensionsKnown_ReturnsTrue()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act & Assert
            Assert.True(shape.IsFullyKnown());
        }

        [Fact]
        public void IsFullyKnown_SomeDimensionsUnknown_ReturnsFalse()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("hidden", 512));

            // Act & Assert
            Assert.False(shape.IsFullyKnown());
        }

        [Fact]
        public void IsFullyKnown_AllDimensionsUnknown_ReturnsFalse()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("seq_len"));

            // Act & Assert
            Assert.False(shape.IsFullyKnown());
        }

        [Fact]
        public void IsPartiallyKnown_AtLeastOneDimensionKnown_ReturnsTrue()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("hidden", 512));

            // Act & Assert
            Assert.True(shape.IsPartiallyKnown());
        }

        [Fact]
        public void IsPartiallyKnown_AllDimensionsUnknown_ReturnsFalse()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("seq_len"));

            // Act & Assert
            Assert.False(shape.IsPartiallyKnown());
        }

        [Fact]
        public void ToConcrete_AllDimensionsKnown_ReturnsConcreteArray()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act
            var concrete = shape.ToConcrete();

            // Assert
            Assert.Equal(2, concrete.Length);
            Assert.Equal(10, concrete[0]);
            Assert.Equal(20, concrete[1]);
        }

        [Fact]
        public void ToConcrete_NotAllDimensionsKnown_ThrowsInvalidOperationException()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("hidden", 512));

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => shape.ToConcrete());
        }

        [Fact]
        public void Clone_CreatesNewInstance()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act
            var cloned = shape.Clone();

            // Assert
            Assert.NotSame(shape, cloned);
            Assert.Equal(shape, cloned);
        }

        [Fact]
        public void Equals_SameDimensions_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act & Assert
            Assert.True(shape1.Equals(shape2));
        }

        [Fact]
        public void Equals_DifferentDimensions_ReturnsFalse()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 30));

            // Act & Assert
            Assert.False(shape1.Equals(shape2));
        }

        [Fact]
        public void Equals_DifferentRank_ReturnsFalse()
        {
            // Arrange
            var shape1 = new SymbolicShape(new SymbolicDimension("dim0", 10));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act & Assert
            Assert.False(shape1.Equals(shape2));
        }

        [Fact]
        public void Equals_Null_ReturnsFalse()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("dim0", 10));

            // Act & Assert
            Assert.False(shape.Equals(null));
        }

        [Fact]
        public void Equals_SameInstance_ReturnsTrue()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("dim0", 10));

            // Act & Assert
            Assert.True(shape.Equals(shape));
        }

        [Fact]
        public void EqualityOperator_SameProperties_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(new SymbolicDimension("dim0", 10));
            var shape2 = new SymbolicShape(new SymbolicDimension("dim0", 10));

            // Act & Assert
            Assert.True(shape1 == shape2);
        }

        [Fact]
        public void InequalityOperator_DifferentProperties_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(new SymbolicDimension("dim0", 10));
            var shape2 = new SymbolicShape(new SymbolicDimension("dim0", 20));

            // Act & Assert
            Assert.True(shape1 != shape2);
        }

        [Fact]
        public void GetHashCode_SameProperties_ReturnsSameHashCode()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act
            var hash1 = shape1.GetHashCode();
            var hash2 = shape2.GetHashCode();

            // Assert
            Assert.Equal(hash1, hash2);
        }

        [Fact]
        public void GetHashCode_DifferentProperties_ReturnsDifferentHashCode()
        {
            // Arrange
            var shape1 = new SymbolicShape(new SymbolicDimension("dim0", 10));
            var shape2 = new SymbolicShape(new SymbolicDimension("dim0", 20));

            // Act
            var hash1 = shape1.GetHashCode();
            var hash2 = shape2.GetHashCode();

            // Assert
            Assert.NotEqual(hash1, hash2);
        }

        [Fact]
        public void UnknownDimension_ReturnsFormattedString()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("seq_len"),
                new SymbolicDimension("hidden", 512));

            // Act
            var result = shape.ToString();

            // Assert
            Assert.Equal("[batch_size[0..∞], seq_len[0..∞], hidden=512]", result);
        }

        [Fact]
        public void KnownDimensions_ReturnsFormattedString()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act
            var result = shape.ToString();

            // Assert
            Assert.Equal("[dim0=10, dim1=20]", result);
        }

        [Fact]
        public void ScalarShape_ReturnsEmptyBrackets()
        {
            // Arrange
            var shape = new SymbolicShape(Array.Empty<SymbolicDimension>());

            // Act
            var result = shape.ToString();

            // Assert
            Assert.Equal("[]", result);
        }

        [Fact]
        public void Immutability_SetDimension_DoesNotModifyOriginal()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act
            shape.SetDimension(0, new SymbolicDimension("new_dim", 100));

            // Assert
            Assert.Equal("dim0", shape.GetDimension(0).Name);
            Assert.Equal(10, shape.GetDimension(0).Value);
        }

        [Fact]
        public void LargeRank_CreatesSuccessfully()
        {
            // Arrange & Act
            var dims = Enumerable.Range(0, 10)
                .Select(i => new SymbolicDimension($"dim{i}", i + 1))
                .ToArray();
            var shape = new SymbolicShape(dims);

            // Assert
            Assert.Equal(10, shape.Rank);
            Assert.Equal(1, shape.GetDimension(0).Value);
            Assert.Equal(10, shape.GetDimension(9).Value);
        }
    }
}
