using Xunit;
using MLFramework.Tensors;
using MLFramework.Shapes;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Tensors
{
    /// <summary>
    /// Unit tests for TensorSymbolic extension methods.
    /// </summary>
    public class TensorSymbolicTests
    {
        [Fact]
        public void Symbolic_WithSingleDimension_CreatesSymbolicTensor()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size", 32);

            // Act
            var tensor = TensorSymbolic.Symbolic(dim);

            // Assert
            Assert.NotNull(tensor);
            Assert.Equal(1, tensor.SymbolicShape.Rank);
            Assert.Equal("batch_size", tensor.SymbolicShape.Dimensions[0].Name);
            Assert.Equal(32, tensor.SymbolicShape.Dimensions[0].Value);
        }

        [Fact]
        public void Symbolic_WithMultipleDimensions_CreatesSymbolicTensor()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size");
            var dim2 = new SymbolicDimension("seq_len", 128);
            var dim3 = new SymbolicDimension("hidden", 512);

            // Act
            var tensor = TensorSymbolic.Symbolic(dim1, dim2, dim3);

            // Assert
            Assert.Equal(3, tensor.SymbolicShape.Rank);
            Assert.Equal("batch_size", tensor.SymbolicShape.Dimensions[0].Name);
            Assert.Equal("seq_len", tensor.SymbolicShape.Dimensions[1].Name);
            Assert.Equal("hidden", tensor.SymbolicShape.Dimensions[2].Name);
        }

        [Fact]
        public void Symbolic_WithNoDimensions_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => TensorSymbolic.Symbolic());
        }

        [Fact]
        public void Symbolic_WithNullDimensions_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => TensorSymbolic.Symbolic(null!));
        }

        [Fact]
        public void Symbolic_WithShape_CreatesSymbolicTensor()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size", 32),
                new SymbolicDimension("seq_len", 128)
            );

            // Act
            var tensor = TensorSymbolic.Symbolic(shape);

            // Assert
            Assert.Equal(2, tensor.SymbolicShape.Rank);
            Assert.Equal(shape, tensor.SymbolicShape);
        }

        [Fact]
        public void Symbolic_WithNullShape_ThrowsArgumentNullException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => TensorSymbolic.Symbolic((SymbolicShape)null!));
        }

        [Fact]
        public void Zeros_WithKnownDimensions_CreatesSymbolicTensor()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", 32);
            var dim2 = new SymbolicDimension("hidden", 512);

            // Act
            var tensor = TensorSymbolic.Zeros(dim1, dim2);

            // Assert
            Assert.NotNull(tensor);
            Assert.Equal(TensorFillType.Zeros, tensor.FillType);
            Assert.Equal(2, tensor.SymbolicShape.Rank);
        }

        [Fact]
        public void Zeros_WithUnknownDimension_ThrowsInvalidOperationException()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => TensorSymbolic.Zeros(dim));
        }

        [Fact]
        public void Ones_WithKnownDimensions_CreatesSymbolicTensor()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", 32);
            var dim2 = new SymbolicDimension("hidden", 512);

            // Act
            var tensor = TensorSymbolic.Ones(dim1, dim2);

            // Assert
            Assert.NotNull(tensor);
            Assert.Equal(TensorFillType.Ones, tensor.FillType);
            Assert.Equal(2, tensor.SymbolicShape.Rank);
        }

        [Fact]
        public void Ones_WithUnknownDimension_ThrowsInvalidOperationException()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => TensorSymbolic.Ones(dim));
        }

        [Fact]
        public void Random_WithKnownDimensions_CreatesSymbolicTensor()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", 32);
            var dim2 = new SymbolicDimension("hidden", 512);

            // Act
            var tensor = TensorSymbolic.Random(dim1, dim2);

            // Assert
            Assert.NotNull(tensor);
            Assert.Equal(TensorFillType.Random, tensor.FillType);
            Assert.Equal(2, tensor.SymbolicShape.Rank);
        }

        [Fact]
        public void Random_WithUnknownDimension_ThrowsInvalidOperationException()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => TensorSymbolic.Random(dim));
        }

        [Fact]
        public void ShapeHint_WithValidConstraint_AddsConstraint()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");
            var tensor = TensorSymbolic.Symbolic(dim);
            var constraint = new RangeConstraint(1, 128);

            // Act
            var result = tensor.ShapeHint("batch_size", constraint);

            // Assert
            Assert.NotSame(tensor, result); // Should return new instance
            Assert.True(result.Constraints.ContainsKey("batch_size"));
            Assert.Single(result.Constraints["batch_size"]);
        }

        [Fact]
        public void ShapeHint_WithNullTensor_ThrowsArgumentNullException()
        {
            // Arrange
            var constraint = new RangeConstraint(1, 128);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                TensorSymbolic.ShapeHint(null!, "batch_size", constraint));
        }

        [Fact]
        public void ShapeHint_WithNullConstraint_ThrowsArgumentNullException()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");
            var tensor = TensorSymbolic.Symbolic(dim);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                tensor.ShapeHint("batch_size", null!));
        }

        [Fact]
        public void WithBounds_WithValidBounds_AddsRangeConstraint()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");
            var tensor = TensorSymbolic.Symbolic(dim);

            // Act
            var result = tensor.WithBounds("batch_size", 1, 128);

            // Assert
            Assert.NotSame(tensor, result);
            Assert.True(result.Constraints.ContainsKey("batch_size"));
            var constraints = result.Constraints["batch_size"];
            Assert.Single(constraints);
            Assert.IsType<RangeConstraint>(constraints[0]);
        }

        [Fact]
        public void WithBounds_WithOnlyMinimum_AddsRangeConstraint()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");
            var tensor = TensorSymbolic.Symbolic(dim);

            // Act
            var result = tensor.WithBounds("batch_size", 1, null);

            // Assert
            Assert.NotSame(tensor, result);
            Assert.True(result.Constraints.ContainsKey("batch_size"));
            var constraint = Assert.IsType<RangeConstraint>(result.Constraints["batch_size"][0]);
            Assert.Equal(1, constraint.Min);
            Assert.Null(constraint.Max);
        }

        [Fact]
        public void GetShape_ReturnsSymbolicShape()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size");
            var dim2 = new SymbolicDimension("seq_len", 128);
            var tensor = TensorSymbolic.Symbolic(dim1, dim2);

            // Act
            var shape = tensor.GetShape();

            // Assert
            Assert.NotNull(shape);
            Assert.Equal(2, shape.Rank);
            Assert.Equal(tensor.SymbolicShape, shape);
        }

        [Fact]
        public void GetShape_WithNullTensor_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                TensorSymbolic.GetShape(null!));
        }

        [Fact]
        public void ResizeTo_WithConcreteDimensions_CreatesConcreteTensor()
        {
            // Arrange
            var tensor = TensorSymbolic.Zeros(
                new SymbolicDimension("batch_size", 32),
                new SymbolicDimension("hidden", 512)
            );

            // Act
            var concrete = tensor.ResizeTo(32, 512);

            // Assert
            Assert.NotNull(concrete);
            Assert.Equal(2, concrete.Dimensions);
            Assert.Equal(32, concrete.Shape[0]);
            Assert.Equal(512, concrete.Shape[1]);
            Assert.Equal(32 * 512, concrete.Size);
        }

        [Fact]
        public void ResizeTo_WithSymbolicDimensions_CreatesNewSymbolicTensor()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size");
            var dim2 = new SymbolicDimension("seq_len");
            var tensor = TensorSymbolic.Symbolic(dim1, dim2);

            // Act
            var result = tensor.ResizeTo(
                new SymbolicDimension("batch_size", 64),
                new SymbolicDimension("seq_len", 128)
            );

            // Assert
            Assert.NotSame(tensor, result);
            Assert.Equal(64, result.SymbolicShape.Dimensions[0].Value);
            Assert.Equal(128, result.SymbolicShape.Dimensions[1].Value);
        }

        [Fact]
        public void GetDimension_WithValidName_ReturnsDimension()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size", 32);
            var dim2 = new SymbolicDimension("seq_len", 128);
            var tensor = TensorSymbolic.Symbolic(dim1, dim2);

            // Act
            var result = tensor.GetDimension("batch_size");

            // Assert
            Assert.NotNull(result);
            Assert.Equal("batch_size", result.Name);
            Assert.Equal(32, result.Value);
        }

        [Fact]
        public void GetDimension_WithInvalidName_ThrowsInvalidOperationException()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");
            var tensor = TensorSymbolic.Symbolic(dim);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                tensor.GetDimension("invalid_name"));
        }

        [Fact]
        public void ResizeTo_WithZerosFillType_CreatesTensorFilledWithZeros()
        {
            // Arrange
            var tensor = TensorSymbolic.Zeros(
                new SymbolicDimension("batch_size", 2),
                new SymbolicDimension("hidden", 3)
            );

            // Act
            var concrete = tensor.ResizeTo(2, 3);

            // Assert
            Assert.Equal(6, concrete.Size);
            foreach (var value in concrete.Data)
            {
                Assert.Equal(0, value);
            }
        }

        [Fact]
        public void ResizeTo_WithOnesFillType_CreatesTensorFilledWithOnes()
        {
            // Arrange
            var tensor = TensorSymbolic.Ones(
                new SymbolicDimension("batch_size", 2),
                new SymbolicDimension("hidden", 3)
            );

            // Act
            var concrete = tensor.ResizeTo(2, 3);

            // Assert
            Assert.Equal(6, concrete.Size);
            foreach (var value in concrete.Data)
            {
                Assert.Equal(1, value);
            }
        }
    }
}
