using Xunit;
using MLFramework.Shapes;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for ModuloConstraint class.
    /// </summary>
    public class ModuloConstraintTests
    {
        [Fact]
        public void Create_WithValidDivisor_CreatesInstance()
        {
            // Arrange & Act
            var constraint = new ModuloConstraint(8);

            // Assert
            Assert.Equal(8, constraint.Divisor);
        }

        [Fact]
        public void Create_WithZeroDivisor_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => new ModuloConstraint(0));
        }

        [Fact]
        public void Create_WithNegativeDivisor_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => new ModuloConstraint(-1));
        }

        [Fact]
        public void Validate_WithDivisibleValue_ReturnsTrue()
        {
            // Arrange
            var constraint = new ModuloConstraint(8);
            var dim = new SymbolicDimension("test", 32);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void Validate_WithNonDivisibleValue_ReturnsFalse()
        {
            // Arrange
            var constraint = new ModuloConstraint(8);
            var dim = new SymbolicDimension("test", 33);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void Validate_WithZeroValue_ReturnsTrue()
        {
            // Arrange
            var constraint = new ModuloConstraint(8);
            var dim = new SymbolicDimension("test", 0);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void Validate_WithUnknownValue_ReturnsFalse()
        {
            // Arrange
            var constraint = new ModuloConstraint(8);
            var dim = new SymbolicDimension("test");

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void Validate_WithNullDimension_ReturnsFalse()
        {
            // Arrange
            var constraint = new ModuloConstraint(8);

            // Act
            var result = constraint.Validate(null!);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ToString_ReturnsExpectedString()
        {
            // Arrange
            var constraint = new ModuloConstraint(8);

            // Act
            var result = constraint.ToString();

            // Assert
            Assert.Equal("Modulo 8", result);
        }

        [Fact]
        public void Equals_WithSameDivisor_ReturnsTrue()
        {
            // Arrange
            var constraint1 = new ModuloConstraint(8);
            var constraint2 = new ModuloConstraint(8);

            // Act
            var result = constraint1 == constraint2;

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void Equals_WithDifferentDivisor_ReturnsFalse()
        {
            // Arrange
            var constraint1 = new ModuloConstraint(8);
            var constraint2 = new ModuloConstraint(16);

            // Act
            var result = constraint1 == constraint2;

            // Assert
            Assert.False(result);
        }
    }
}
