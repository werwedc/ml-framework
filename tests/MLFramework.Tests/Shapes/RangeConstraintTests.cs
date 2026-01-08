using Xunit;
using MLFramework.Shapes;
using System.Collections.Generic;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for RangeConstraint class.
    /// </summary>
    public class RangeConstraintTests
    {
        [Fact]
        public void Create_WithValidValues_CreatesInstance()
        {
            // Arrange & Act
            var constraint = new RangeConstraint(0, 100);

            // Assert
            Assert.Equal(0, constraint.MinValue);
            Assert.Equal(100, constraint.MaxValue);
        }

        [Fact]
        public void Create_WithNegativeMin_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => new RangeConstraint(-1, 100));
        }

        [Fact]
        public void Create_WithMaxLessThanMin_ThrowsArgumentException()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => new RangeConstraint(100, 50));
        }

        [Fact]
        public void Validate_WithValueInRange_ReturnsTrue()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);
            var dim = new SymbolicDimension("test", 50);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void Validate_WithValueAtMinBoundary_ReturnsTrue()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);
            var dim = new SymbolicDimension("test", 10);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void Validate_WithValueAtMaxBoundary_ReturnsTrue()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);
            var dim = new SymbolicDimension("test", 100);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void Validate_WithValueBelowRange_ReturnsFalse()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);
            var dim = new SymbolicDimension("test", 5);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void Validate_WithValueAboveRange_ReturnsFalse()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);
            var dim = new SymbolicDimension("test", 150);

            // Act
            var result = constraint.Validate(dim);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void Validate_WithUnknownValue_ReturnsFalse()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);
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
            var constraint = new RangeConstraint(10, 100);

            // Act
            var result = constraint.Validate(null!);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void Clamp_WithValueAboveRange_ClampsToMax()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);

            // Act
            var result = constraint.Clamp(150);

            // Assert
            Assert.Equal(100, result);
        }

        [Fact]
        public void Clamp_WithValueBelowRange_ClampsToMin()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);

            // Act
            var result = constraint.Clamp(5);

            // Assert
            Assert.Equal(10, result);
        }

        [Fact]
        public void Clamp_WithValueInRange_ReturnsUnchanged()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);

            // Act
            var result = constraint.Clamp(50);

            // Assert
            Assert.Equal(50, result);
        }

        [Fact]
        public void ToString_ReturnsExpectedString()
        {
            // Arrange
            var constraint = new RangeConstraint(10, 100);

            // Act
            var result = constraint.ToString();

            // Assert
            Assert.Equal("Range [10, 100]", result);
        }

        [Fact]
        public void Equals_WithSameValues_ReturnsTrue()
        {
            // Arrange
            var constraint1 = new RangeConstraint(10, 100);
            var constraint2 = new RangeConstraint(10, 100);

            // Act
            var result = constraint1 == constraint2;

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void Equals_WithDifferentValues_ReturnsFalse()
        {
            // Arrange
            var constraint1 = new RangeConstraint(10, 100);
            var constraint2 = new RangeConstraint(20, 200);

            // Act
            var result = constraint1 == constraint2;

            // Assert
            Assert.False(result);
        }
    }
}
