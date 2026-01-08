using Xunit;
using MLFramework.Tensors;
using MLFramework.Shapes;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Tensors
{
    /// <summary>
    /// Unit tests for SymbolicTensor class.
    /// </summary>
    public class SymbolicTensorTests
    {
        [Fact]
        public void Constructor_WithShape_CreatesInstance()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size", 32),
                new SymbolicDimension("seq_len", 128)
            );

            // Act
            var tensor = new SymbolicTensor(shape);

            // Assert
            Assert.Equal(shape, tensor.SymbolicShape);
            Assert.Empty(tensor.Constraints);
            Assert.Equal(TensorFillType.None, tensor.FillType);
        }

        [Fact]
        public void Constructor_WithNullShape_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new SymbolicTensor(null!));
        }

        [Fact]
        public void Constructor_WithFillType_SetsFillType()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size", 32)
            );

            // Act
            var tensor = new SymbolicTensor(shape, TensorFillType.Zeros);

            // Assert
            Assert.Equal(TensorFillType.Zeros, tensor.FillType);
        }

        [Fact]
        public void ValidateShape_WithNoConstraints_ReturnsTrue()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("batch_size", 32));
            var tensor = new SymbolicTensor(shape);

            // Act
            var result = tensor.ValidateShape();

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void ValidateShape_WithSatisfiedConstraint_ReturnsTrue()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size", 32);
            var shape = new SymbolicShape(dim);
            var tensor = new SymbolicTensor(shape);
            var constraint = new RangeConstraint(1, 128);
            tensor = tensor.WithConstraint("batch_size", constraint);

            // Act
            var result = tensor.ValidateShape();

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void ValidateShape_WithUnsatisfiedConstraint_ReturnsFalse()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size", 256);
            var shape = new SymbolicShape(dim);
            var tensor = new SymbolicTensor(shape);
            var constraint = new RangeConstraint(1, 128);
            tensor = tensor.WithConstraint("batch_size", constraint);

            // Act
            var result = tensor.ValidateShape();

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ValidateShape_WithMissingDimension_ReturnsFalse()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("seq_len", 128));
            var tensor = new SymbolicTensor(shape);
            var constraint = new RangeConstraint(1, 128);
            tensor = tensor.WithConstraint("batch_size", constraint);

            // Act
            var result = tensor.ValidateShape();

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void GetConcreteShape_WithValidValues_ReturnsConcreteArray()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("hidden", 512)
            );
            var tensor = new SymbolicTensor(shape);

            // Act
            var result = tensor.GetConcreteShape(32, 512);

            // Assert
            Assert.Equal(2, result.Length);
            Assert.Equal(32, result[0]);
            Assert.Equal(512, result[1]);
        }

        [Fact]
        public void GetConcreteShape_WithWrongNumberOfValues_ThrowsInvalidOperationException()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("batch_size"));
            var tensor = new SymbolicTensor(shape);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => tensor.GetConcreteShape(32, 64));
        }

        [Fact]
        public void GetConcreteShape_WithNullValues_ThrowsInvalidOperationException()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("batch_size"));
            var tensor = new SymbolicTensor(shape);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => tensor.GetConcreteShape(null!));
        }

        [Fact]
        public void GetConcreteShape_WithUnsatisfiedConstraints_ThrowsInvalidOperationException()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");
            var shape = new SymbolicShape(dim);
            var tensor = new SymbolicTensor(shape);
            var constraint = new RangeConstraint(1, 32);
            tensor = tensor.WithConstraint("batch_size", constraint);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => tensor.GetConcreteShape(64));
        }

        [Fact]
        public void CanInstantiateWith_WithValidDimensions_ReturnsTrue()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                new SymbolicDimension("hidden", 512)
            );
            var tensor = new SymbolicTensor(shape);

            // Act
            var result = tensor.CanInstantiateWith(32, 512);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanInstantiateWith_WithWrongNumberOfDimensions_ReturnsFalse()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("batch_size"));
            var tensor = new SymbolicTensor(shape);

            // Act
            var result = tensor.CanInstantiateWith(32, 64);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void CanInstantiateWith_WithUnsatisfiedConstraints_ReturnsFalse()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");
            var shape = new SymbolicShape(dim);
            var tensor = new SymbolicTensor(shape);
            var constraint = new RangeConstraint(1, 32);
            tensor = tensor.WithConstraint("batch_size", constraint);

            // Act
            var result = tensor.CanInstantiateWith(64);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ToConcrete_WithZerosFillType_CreatesZerosTensor()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size", 32),
                new SymbolicDimension("hidden", 512)
            );
            var tensor = new SymbolicTensor(shape, TensorFillType.Zeros);

            // Act
            var result = tensor.ToConcrete(32, 512);

            // Assert
            Assert.Equal(2, result.Dimensions);
            Assert.Equal(32 * 512, result.Size);
            foreach (var value in result.Data)
            {
                Assert.Equal(0, value);
            }
        }

        [Fact]
        public void ToConcrete_WithOnesFillType_CreatesOnesTensor()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size", 2),
                new SymbolicDimension("hidden", 3)
            );
            var tensor = new SymbolicTensor(shape, TensorFillType.Ones);

            // Act
            var result = tensor.ToConcrete(2, 3);

            // Assert
            Assert.Equal(6, result.Size);
            foreach (var value in result.Data)
            {
                Assert.Equal(1, value);
            }
        }

        [Fact]
        public void ToConcrete_WithRandomFillType_CreatesRandomTensor()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size", 2),
                new SymbolicDimension("hidden", 3)
            );
            var tensor = new SymbolicTensor(shape, TensorFillType.Random);

            // Act
            var result = tensor.ToConcrete(2, 3);

            // Assert
            Assert.Equal(6, result.Size);
            // Check that values are in valid range (0-1)
            foreach (var value in result.Data)
            {
                Assert.InRange(value, 0, 1);
            }
        }

        [Fact]
        public void ToConcrete_WithNoneFillType_ThrowsInvalidOperationException()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size", 32),
                new SymbolicDimension("hidden", 512)
            );
            var tensor = new SymbolicTensor(shape);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => tensor.ToConcrete(32, 512));
        }

        [Fact]
        public void ToConcrete_WithUnsatisfiedConstraints_ThrowsInvalidOperationException()
        {
            // Arrange
            var dim = new SymbolicDimension("batch_size");
            var shape = new SymbolicShape(dim);
            var tensor = new SymbolicTensor(shape, TensorFillType.Zeros);
            var constraint = new RangeConstraint(1, 32);
            tensor = tensor.WithConstraint("batch_size", constraint);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => tensor.ToConcrete(64));
        }

        [Fact]
        public void WithConstraint_ReturnsNewInstance()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("batch_size"));
            var tensor = new SymbolicTensor(shape);
            var constraint = new RangeConstraint(1, 128);

            // Act
            var result = tensor.WithConstraint("batch_size", constraint);

            // Assert
            Assert.NotSame(tensor, result);
            Assert.Empty(tensor.Constraints);
            Assert.Single(result.Constraints);
        }

        [Fact]
        public void WithConstraint_AddsConstraintToExistingOnes()
        {
            // Arrange
            var shape = new SymbolicShape(new SymbolicDimension("batch_size"));
            var tensor = new SymbolicTensor(shape);
            var constraint1 = new RangeConstraint(1, 128);
            var constraint2 = new RangeConstraint(2, 64);

            // Act
            var result = tensor
                .WithConstraint("batch_size", constraint1)
                .WithConstraint("batch_size", constraint2);

            // Assert
            Assert.Equal(2, result.Constraints["batch_size"].Count);
        }

        [Fact]
        public void WithShape_ReturnsNewInstanceWithNewShape()
        {
            // Arrange
            var shape1 = new SymbolicShape(new SymbolicDimension("batch_size"));
            var tensor = new SymbolicTensor(shape1);
            var shape2 = new SymbolicShape(
                new SymbolicDimension("batch_size", 64),
                new SymbolicDimension("seq_len", 128)
            );

            // Act
            var result = tensor.WithShape(shape2);

            // Assert
            Assert.NotSame(tensor, result);
            Assert.Equal(2, result.SymbolicShape.Rank);
            Assert.Equal(1, tensor.SymbolicShape.Rank);
        }

        [Fact]
        public void ToString_ReturnsDescriptiveString()
        {
            // Arrange
            var shape = new SymbolicShape(
                new SymbolicDimension("batch_size", 32),
                new SymbolicDimension("hidden", 512)
            );
            var tensor = new SymbolicTensor(shape, TensorFillType.Zeros);

            // Act
            var result = tensor.ToString();

            // Assert
            Assert.Contains("SymbolicTensor", result);
            Assert.Contains("[batch_size=32, hidden=512]", result);
            Assert.Contains("Fill: Zeros", result);
        }
    }
}
