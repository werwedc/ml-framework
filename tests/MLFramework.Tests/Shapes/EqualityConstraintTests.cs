using Xunit;
using MLFramework.Shapes;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for EqualityConstraint class.
    /// </summary>
    public class EqualityConstraintTests
    {
        [Fact]
        public void Create_WithValidValue_CreatesInstance()
        {
            // Arrange & Act
            var constraint = new EqualityConstraint(42);

            // Assert
            Assert.Equal(42, constraint.TargetValue);
        }

        [Fact]
        public void Create_WithNegativeValue_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => new EqualityConstraint(-1));
        }

        [Fact]
        public void Validate_WithMatchingValue_ReturnsTrue()
        {
            // Arrange
            var constraint = new EqualityConstraint(42);
            var dim = new SymbolicDimension("test", 42);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void Validate_WithDifferentValue_ReturnsFalse()
        {
            // Arrange
            var constraint = new EqualityConstraint(42);
            var dim = new SymbolicDimension("test", 50);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void Validate_WithUnknownValue_ReturnsFalse()
        {
            // Arrange
            var constraint = new EqualityConstraint(42);
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
            var constraint = new EqualityConstraint(42);

            // Act
            var result = constraint.Validate(null!);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ToString_ReturnsExpectedString()
        {
            // Arrange
            var constraint = new EqualityConstraint(42);

            // Act
            var result = constraint.ToString();

            // Assert
            Assert.Equal("Equals 42", result);
        }

        [Fact]
        public void Equals_WithSameValue_ReturnsTrue()
        {
            // Arrange
            var constraint1 = new EqualityConstraint(42);
            var constraint2 = new EqualityConstraint(42);

            // Act
            var result = constraint1 == constraint2;

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void Equals_WithDifferentValue_ReturnsFalse()
        {
            // Arrange
            var constraint1 = new EqualityConstraint(42);
            var constraint2 = new EqualityConstraint(50);

            // Act
            var result = constraint1 == constraint2;

            // Assert
            Assert.False(result);
        }
    }
}
