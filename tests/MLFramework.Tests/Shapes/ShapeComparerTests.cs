using Xunit;
using MLFramework.Shapes;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for ShapeComparer class.
    /// </summary>
    public class ShapeComparerTests
    {
        #region AreCompatible Tests

        [Fact]
        public void AreCompatible_SameShapes_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act
            bool result = ShapeComparer.AreCompatible(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void AreCompatible_ScalarAndAnyShape_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(Array.Empty<SymbolicDimension>());
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act
            bool result = ShapeComparer.AreCompatible(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void AreCompatible_VectorAndMatrix_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim1", 4));

            // Act
            bool result = ShapeComparer.AreCompatible(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void AreCompatible_OneDimensionIsOne_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 5),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 1),
                new SymbolicDimension("dim1", 4));

            // Act
            bool result = ShapeComparer.AreCompatible(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void AreCompatible_SymbolicDimensions_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("dim1", 4));

            // Act
            bool result = ShapeComparer.AreCompatible(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void AreCompatible_DifferentConcreteValues_ReturnsFalse()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 5),
                new SymbolicDimension("dim1", 4));

            // Act
            bool result = ShapeComparer.AreCompatible(shape1, shape2);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void AreCompatible_DifferentRanksCompatible_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim1", 4));

            // Act
            bool result = ShapeComparer.AreCompatible(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void AreCompatible_MixedKnownAndUnknown_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 4));

            // Act
            bool result = ShapeComparer.AreCompatible(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void AreCompatible_NullShape_ThrowsArgumentNullException()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("dim0", 10));

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => ShapeComparer.AreCompatible(null!, shape));
        }

        #endregion

        #region GetBroadcastShape Tests

        [Fact]
        public void GetBroadcastShape_SameShapes_ReturnsSameShape()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20));

            // Act
            var result = ShapeComparer.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(2, result.Rank);
            Assert.Equal(10, result.GetDimension(0).Value);
            Assert.Equal(20, result.GetDimension(1).Value);
        }

        [Fact]
        public void GetBroadcastShape_ScalarAndMatrix_ReturnsMatrix()
        {
            // Arrange
            var shape1 = new SymbolicShape(Array.Empty<SymbolicDimension>());
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 4));

            // Act
            var result = ShapeComparer.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(2, result.Rank);
            Assert.Equal(3, result.GetDimension(0).Value);
            Assert.Equal(4, result.GetDimension(1).Value);
        }

        [Fact]
        public void GetBroadcastShape_VectorAndMatrix_ReturnsMatrix()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim1", 4));

            // Act
            var result = ShapeComparer.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(2, result.Rank);
            Assert.Equal(3, result.GetDimension(0).Value);
            Assert.Equal(4, result.GetDimension(1).Value);
        }

        [Fact]
        public void GetBroadcastShape_OneDimensionIsOne_ReturnsLargerDimension()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 5),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 1),
                new SymbolicDimension("dim1", 4));

            // Act
            var result = ShapeComparer.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(2, result.Rank);
            Assert.Equal(5, result.GetDimension(0).Value);
            Assert.Equal(4, result.GetDimension(1).Value);
        }

        [Fact]
        public void GetBroadcastShape_SymbolicDimensions_ReturnsSymbolicShape()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("dim1", 4));

            // Act
            var result = ShapeComparer.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(2, result.Rank);
            Assert.False(result.GetDimension(0).Value.HasValue);
        }

        [Fact]
        public void GetBroadcastShape_IncompatibleShapes_ThrowsArgumentException()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 5),
                new SymbolicDimension("dim1", 4));

            // Act & Assert
            Assert.Throws<ArgumentException>(() => ShapeComparer.GetBroadcastShape(shape1, shape2));
        }

        [Fact]
        public void GetBroadcastShape_MultipleOnes_BroadcastsCorrectly()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", 1),
                new SymbolicDimension("dim1", 1),
                new SymbolicDimension("dim2", 5));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 4),
                new SymbolicDimension("dim2", 5));

            // Act
            var result = ShapeComparer.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(3, result.Rank);
            Assert.Equal(3, result.GetDimension(0).Value);
            Assert.Equal(4, result.GetDimension(1).Value);
            Assert.Equal(5, result.GetDimension(2).Value);
        }

        [Fact]
        public void GetBroadcastShape_UnknownDimensionCombinesBounds()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("dim0", minValue: 0, maxValue: 100));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("dim0", minValue: 50, maxValue: 200));

            // Act
            var result = ShapeComparer.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(1, result.Rank);
            Assert.Equal(50, result.GetDimension(0).MinValue);
            Assert.Equal(200, result.GetDimension(0).MaxValue);
        }

        #endregion

        #region CanReshape Tests

        [Fact]
        public void CanReshape_SameSize_ReturnsTrue()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 3));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 2));

            // Act
            bool result = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanReshape_DifferentSize_ReturnsFalse()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 3));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", 4),
                new SymbolicDimension("dim1", 3));

            // Act
            bool result = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void CanReshape_SingleUnknownDimension_ReturnsTrue()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 3));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", -1),
                new SymbolicDimension("dim1", 3));

            // Act
            bool result = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanReshape_MultipleUnknownDimensions_ReturnsFalse()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 3));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", -1),
                new SymbolicDimension("dim1", -1));

            // Act
            bool result = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void CanReshape_BothPartiallyKnown_ReturnsTrue()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("dim1", 3));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", -1),
                new SymbolicDimension("dim1", 3));

            // Act
            bool result = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanReshape_DifferentRank_ReturnsTrue()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 3),
                new SymbolicDimension("dim2", 4));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", 24));

            // Act
            bool result = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanReshape_NullShape_ThrowsArgumentNullException()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("dim0", 10));

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => ShapeComparer.CanReshape(null!, shape));
        }

        #endregion

        #region Reshape Tests

        [Fact]
        public void Reshape_KnownToKnown_ReturnsTarget()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 3));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", 3),
                new SymbolicDimension("dim1", 2));

            // Act
            var result = ShapeComparer.Reshape(from, to);

            // Assert
            Assert.Equal(to, result);
        }

        [Fact]
        public void Reshape_WithInferredDimension_ComputesCorrectly()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 6));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", -1),
                new SymbolicDimension("dim1", 3));

            // Act
            var result = ShapeComparer.Reshape(from, to);

            // Assert
            Assert.Equal(4, result.GetDimension(0).Value);
        }

        [Fact]
        public void Reshape_MultipleInferredDimensions_ThrowsArgumentException()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 6));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", -1),
                new SymbolicDimension("dim1", -1));

            // Act & Assert
            Assert.Throws<ArgumentException>(() => ShapeComparer.Reshape(from, to));
        }

        [Fact]
        public void Reshape_NonDivisible_ThrowsArgumentException()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 5));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", -1),
                new SymbolicDimension("dim1", 3));

            // Act & Assert
            Assert.Throws<ArgumentException>(() => ShapeComparer.Reshape(from, to));
        }

        [Fact]
        public void Reshape_InvalidReshape_ThrowsArgumentException()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 2),
                new SymbolicDimension("dim1", 3));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", 4),
                new SymbolicDimension("dim1", 3));

            // Act & Assert
            Assert.Throws<ArgumentException>(() => ShapeComparer.Reshape(from, to));
        }

        [Fact]
        public void Reshape_PartitionKnown_UnknownSource_ReturnsTarget()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("dim1", 6));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", -1),
                new SymbolicDimension("dim1", 3));

            // Act
            var result = ShapeComparer.Reshape(from, to);

            // Assert
            Assert.False(result.GetDimension(0).Value.HasValue);
        }

        [Fact]
        public void Reshape_InferredDimensionNamedCorrectly()
        {
            // Arrange
            var from = new SymbolicShape(
                new SymbolicDimension("dim0", 12));
            var to = new SymbolicShape(
                new SymbolicDimension("dim0", -1),
                new SymbolicDimension("dim1", 4));

            // Act
            var result = ShapeComparer.Reshape(from, to);

            // Assert
            Assert.Equal("inferred_0", result.GetDimension(0).Name);
            Assert.Equal(3, result.GetDimension(0).Value);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void AreCompatible_ScalarAndScalar_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(Array.Empty<SymbolicDimension>());
            var shape2 = new SymbolicShape(Array.Empty<SymbolicDimension>());

            // Act
            bool result = ShapeComparer.AreCompatible(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void GetBroadcastShape_ScalarAndScalar_ReturnsScalar()
        {
            // Arrange
            var shape1 = new SymbolicShape(Array.Empty<SymbolicDimension>());
            var shape2 = new SymbolicShape(Array.Empty<SymbolicDimension>());

            // Act
            var result = ShapeComparer.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(0, result.Rank);
        }

        [Fact]
        public void CanReshape_ScalarToVector_ReturnsTrue()
        {
            // Arrange
            var from = new SymbolicShape(Array.Empty<SymbolicDimension>());
            var to = new SymbolicShape(new SymbolicDimension("dim0", 1));

            // Act
            bool result = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanReshape_VectorToScalar_ReturnsTrue()
        {
            // Arrange
            var from = new SymbolicShape(new SymbolicDimension("dim0", 1));
            var to = new SymbolicShape(Array.Empty<SymbolicDimension>());

            // Act
            bool result = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanReshape_VectorToNonScalar_False()
        {
            // Arrange
            var from = new SymbolicShape(new SymbolicDimension("dim0", 5));
            var to = new SymbolicShape(Array.Empty<SymbolicDimension>());

            // Act
            bool result = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.False(result);
        }

        #endregion
    }
}
