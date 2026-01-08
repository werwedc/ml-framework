using Xunit;
using MLFramework.Shapes;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for SymbolicShapeFactory class.
    /// </summary>
    public class SymbolicShapeFactoryTests
    {
        [Fact]
        public void Create_WithDimensions_ReturnsShape()
        {
            // Arrange
            var dims = new SymbolicDimension[]
            {
                new SymbolicDimension("dim0", 10),
                new SymbolicDimension("dim1", 20)
            };

            // Act
            var shape = SymbolicShapeFactory.Create(dims);

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.Equal(10, shape.GetDimension(0).Value);
            Assert.Equal(20, shape.GetDimension(1).Value);
        }

        [Fact]
        public void FromConcrete_WithDimensions_ReturnsShapeWithKnownDimensions()
        {
            // Arrange
            int[] dims = { 10, 20, 30 };

            // Act
            var shape = SymbolicShapeFactory.FromConcrete(dims);

            // Assert
            Assert.Equal(3, shape.Rank);
            Assert.True(shape.IsFullyKnown());
            Assert.Equal(10, shape.GetDimension(0).Value);
            Assert.Equal(20, shape.GetDimension(1).Value);
            Assert.Equal(30, shape.GetDimension(2).Value);
        }

        [Fact]
        public void FromConcrete_WithNull_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => SymbolicShapeFactory.FromConcrete(null!));
        }

        [Fact]
        public void FromConcrete_SingleDimension_ReturnsVector()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.FromConcrete(10);

            // Assert
            Assert.Equal(1, shape.Rank);
            Assert.Equal(10, shape.GetDimension(0).Value);
        }

        [Fact]
        public void FromConcrete_MultipleDimensions_ReturnsCorrectShape()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.FromConcrete(2, 3, 4);

            // Assert
            Assert.Equal(3, shape.Rank);
            Assert.Equal(2, shape.GetDimension(0).Value);
            Assert.Equal(3, shape.GetDimension(1).Value);
            Assert.Equal(4, shape.GetDimension(2).Value);
        }

        [Fact]
        public void Scalar_ReturnsRankZeroShape()
        {
            // Act
            var shape = SymbolicShapeFactory.Scalar();

            // Assert
            Assert.Equal(0, shape.Rank);
            Assert.True(shape.IsFullyKnown());
        }

        [Fact]
        public void Vector_WithLength_ReturnsVectorWithKnownLength()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.Vector(10);

            // Assert
            Assert.Equal(1, shape.Rank);
            Assert.True(shape.IsFullyKnown());
            Assert.Equal(10, shape.GetDimension(0).Value);
        }

        [Fact]
        public void Vector_WithName_ReturnsVectorWithSymbolicLength()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.Vector("seq_len");

            // Assert
            Assert.Equal(1, shape.Rank);
            Assert.Equal("seq_len", shape.GetDimension(0).Name);
            Assert.False(shape.IsFullyKnown());
        }

        [Fact]
        public void Matrix_WithKnownDimensions_ReturnsMatrix()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.Matrix(3, 4);

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.True(shape.IsFullyKnown());
            Assert.Equal(3, shape.GetDimension(0).Value);
            Assert.Equal(4, shape.GetDimension(1).Value);
        }

        [Fact]
        public void Matrix_WithNamedDimensions_ReturnsSymbolicMatrix()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.Matrix("rows", "cols");

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.Equal("rows", shape.GetDimension(0).Name);
            Assert.Equal("cols", shape.GetDimension(1).Name);
            Assert.False(shape.IsFullyKnown());
        }

        [Fact]
        public void Batched_WithSymbolicBatch_AddsBatchDimension()
        {
            // Arrange
            var innerShape = SymbolicShapeFactory.Matrix(10, 20);

            // Act
            var shape = SymbolicShapeFactory.Batched("batch_size", innerShape);

            // Assert
            Assert.Equal(3, shape.Rank);
            Assert.Equal("batch_size", shape.GetDimension(0).Name);
            Assert.Equal(10, shape.GetDimension(1).Value);
            Assert.Equal(20, shape.GetDimension(2).Value);
        }

        [Fact]
        public void Batched_WithConcreteBatch_AddsBatchDimension()
        {
            // Arrange
            var innerShape = SymbolicShapeFactory.Matrix(10, 20);

            // Act
            var shape = SymbolicShapeFactory.Batched(32, innerShape);

            // Assert
            Assert.Equal(3, shape.Rank);
            Assert.Equal(32, shape.GetDimension(0).Value);
            Assert.Equal(10, shape.GetDimension(1).Value);
            Assert.Equal(20, shape.GetDimension(2).Value);
        }

        [Fact]
        public void Batched_WithNullInnerShape_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => SymbolicShapeFactory.Batched("batch", null!));
        }

        [Fact]
        public void Batched_WithScalar_AddsBatchDimension()
        {
            // Arrange
            var innerShape = SymbolicShapeFactory.Scalar();

            // Act
            var shape = SymbolicShapeFactory.Batched(10, innerShape);

            // Assert
            Assert.Equal(1, shape.Rank);
            Assert.Equal(10, shape.GetDimension(0).Value);
        }

        [Fact]
        public void Batched_WithVector_AddsBatchDimension()
        {
            // Arrange
            var innerShape = SymbolicShapeFactory.Vector(20);

            // Act
            var shape = SymbolicShapeFactory.Batched(10, innerShape);

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.Equal(10, shape.GetDimension(0).Value);
            Assert.Equal(20, shape.GetDimension(1).Value);
        }

        [Fact]
        public void Create_WithNoDimensions_CreatesScalar()
        {
            // Act
            var shape = SymbolicShapeFactory.Create(Array.Empty<SymbolicDimension>());

            // Assert
            Assert.Equal(0, shape.Rank);
            Assert.True(shape.IsFullyKnown());
        }

        [Fact]
        public void Create_WithSymbolicDimensions_ReturnsSymbolicShape()
        {
            // Arrange
            var dims = new SymbolicDimension[]
            {
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("seq_len"),
                new SymbolicDimension("hidden", 512)
            };

            // Act
            var shape = SymbolicShapeFactory.Create(dims);

            // Assert
            Assert.Equal(3, shape.Rank);
            Assert.False(shape.IsFullyKnown());
            Assert.True(shape.IsPartiallyKnown());
            Assert.Equal("batch_size", shape.GetDimension(0).Name);
            Assert.Equal("seq_len", shape.GetDimension(1).Name);
            Assert.Equal(512, shape.GetDimension(2).Value);
        }

        [Fact]
        public void Create_WithMixedKnownAndUnknown_ReturnsMixedShape()
        {
            // Arrange
            var dims = new SymbolicDimension[]
            {
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("features", 512)
            };

            // Act
            var shape = SymbolicShapeFactory.Create(dims);

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.False(shape.IsFullyKnown());
            Assert.True(shape.IsPartiallyKnown());
        }

        [Fact]
        public void FromConcrete_LargeDimensions_HandlesCorrectly()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.FromConcrete(1000, 2000, 3000);

            // Assert
            Assert.Equal(3, shape.Rank);
            Assert.Equal(1000, shape.GetDimension(0).Value);
            Assert.Equal(2000, shape.GetDimension(1).Value);
            Assert.Equal(3000, shape.GetDimension(2).Value);
        }

        [Fact]
        public void Matrix_SquareMatrix_ReturnsCorrectShape()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.Matrix(10, 10);

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.Equal(10, shape.GetDimension(0).Value);
            Assert.Equal(10, shape.GetDimension(1).Value);
        }

        [Fact]
        public void Vector_ZeroLength_ReturnsCorrectShape()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.Vector(0);

            // Assert
            Assert.Equal(1, shape.Rank);
            Assert.Equal(0, shape.GetDimension(0).Value);
        }

        [Fact]
        public void Batched_WithDeeplyNestedShape_AddsBatchDimension()
        {
            // Arrange
            var innerShape = SymbolicShapeFactory.FromConcrete(10, 20, 30, 40);

            // Act
            var shape = SymbolicShapeFactory.Batched(5, innerShape);

            // Assert
            Assert.Equal(5, shape.Rank);
            Assert.Equal(5, shape.GetDimension(0).Value);
            Assert.Equal(10, shape.GetDimension(1).Value);
            Assert.Equal(20, shape.GetDimension(2).Value);
            Assert.Equal(30, shape.GetDimension(3).Value);
            Assert.Equal(40, shape.GetDimension(4).Value);
        }

        [Fact]
        public void Create_DimensionNamesArePreserved()
        {
            // Arrange
            var dims = new SymbolicDimension[]
            {
                new SymbolicDimension("batch_size", 32),
                new SymbolicDimension("seq_len", 128)
            };

            // Act
            var shape = SymbolicShapeFactory.Create(dims);

            // Assert
            Assert.Equal("batch_size", shape.GetDimension(0).Name);
            Assert.Equal("seq_len", shape.GetDimension(1).Name);
        }

        [Fact]
        public void FromConcrete_GeneratesDimensionNames()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.FromConcrete(10, 20);

            // Assert
            Assert.Equal("dim_0", shape.GetDimension(0).Name);
            Assert.Equal("dim_1", shape.GetDimension(1).Name);
        }

        [Fact]
        public void FromConcrete_SingleElement_GeneratesCorrectName()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.FromConcrete(10);

            // Assert
            Assert.Equal("dim_0", shape.GetDimension(0).Name);
        }
    }
}
