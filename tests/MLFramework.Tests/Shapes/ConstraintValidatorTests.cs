using Xunit;
using MLFramework.Shapes;
using System.Collections.Generic;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for ConstraintValidator class.
    /// </summary>
    public class ConstraintValidatorTests
    {
        [Fact]
        public void ValidateAll_WithEmptyDimensions_ReturnsTrue()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>();
            var constraints = new Dictionary<string, List<IShapeConstraint>>();

            // Act
            var result = validator.ValidateAll(dims, constraints);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void ValidateAll_WithNoConstraints_ReturnsTrue()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 32)
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>();

            // Act
            var result = validator.ValidateAll(dims, constraints);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void ValidateAll_WithSatisfiedConstraints_ReturnsTrue()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 32)
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>
            {
                ["batch_size"] = new List<IShapeConstraint>
                {
                    new RangeConstraint(16, 64),
                    new ModuloConstraint(8)
                }
            };

            // Act
            var result = validator.ValidateAll(dims, constraints);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void ValidateAll_WithViolatedConstraints_ReturnsFalse()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 33)
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>
            {
                ["batch_size"] = new List<IShapeConstraint>
                {
                    new ModuloConstraint(8)
                }
            };

            // Act
            var result = validator.ValidateAll(dims, constraints);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ValidateAll_WithUnknownValue_ReturnsFalse()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size")
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>
            {
                ["batch_size"] = new List<IShapeConstraint>
                {
                    new EqualityConstraint(32)
                }
            };

            // Act
            var result = validator.ValidateAll(dims, constraints);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ValidateAll_WithNullDimensions_ThrowsArgumentNullException()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var constraints = new Dictionary<string, List<IShapeConstraint>>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => validator.ValidateAll(null!, constraints));
        }

        [Fact]
        public void ValidateAll_WithNullConstraints_ThrowsArgumentNullException()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 32)
            };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => validator.ValidateAll(dims, null!));
        }

        [Fact]
        public void GetViolations_WithNoViolations_ReturnsEmptyList()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 32)
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>
            {
                ["batch_size"] = new List<IShapeConstraint>
                {
                    new EqualityConstraint(32)
                }
            };

            // Act
            var violations = validator.GetViolations(dims, constraints);

            // Assert
            Assert.Empty(violations);
        }

        [Fact]
        public void GetViolations_WithViolations_ReturnsDescriptiveMessages()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 33)
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>
            {
                ["batch_size"] = new List<IShapeConstraint>
                {
                    new ModuloConstraint(8)
                }
            };

            // Act
            var violations = validator.GetViolations(dims, constraints);

            // Assert
            Assert.Single(violations);
            Assert.Contains("batch_size", violations[0]);
            Assert.Contains("Modulo 8", violations[0]);
        }

        [Fact]
        public void GetViolations_WithUnknownValue_ReturnsDescriptiveMessage()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size")
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>
            {
                ["batch_size"] = new List<IShapeConstraint>
                {
                    new EqualityConstraint(32)
                }
            };

            // Act
            var violations = validator.GetViolations(dims, constraints);

            // Assert
            Assert.Single(violations);
            Assert.Contains("unknown value", violations[0]);
        }

        [Fact]
        public void GetViolations_WithMultipleViolations_ReturnsAllMessages()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 33),
                new SymbolicDimension("seq_len", 1000)
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>
            {
                ["batch_size"] = new List<IShapeConstraint>
                {
                    new ModuloConstraint(8),
                    new RangeConstraint(16, 64)
                },
                ["seq_len"] = new List<IShapeConstraint>
                {
                    new RangeConstraint(0, 512)
                }
            };

            // Act
            var violations = validator.GetViolations(dims, constraints);

            // Assert
            Assert.Equal(2, violations.Count);
        }

        [Fact]
        public void GetViolations_WithNullDimension_ReturnsDescriptiveMessage()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension?> { null };
            var constraints = new Dictionary<string, List<IShapeConstraint>>();

            // Act
            var violations = validator.GetViolations(dims!, constraints);

            // Assert
            Assert.Single(violations);
            Assert.Contains("Null dimension", violations[0]);
        }

        [Fact]
        public void GetViolations_WithNullConstraints_ThrowsArgumentNullException()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 32)
            };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => validator.GetViolations(dims, null!));
        }

        [Fact]
        public void ValidateDimension_WithNoConstraints_ReturnsTrue()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dim = new SymbolicDimension("batch_size", 32);
            var constraints = new List<IShapeConstraint>();

            // Act
            var result = validator.ValidateDimension(dim, constraints);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void ValidateDimension_WithSatisfiedConstraints_ReturnsTrue()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dim = new SymbolicDimension("batch_size", 32);
            var constraints = new List<IShapeConstraint>
            {
                new EqualityConstraint(32),
                new ModuloConstraint(8)
            };

            // Act
            var result = validator.ValidateDimension(dim, constraints);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void ValidateDimension_WithViolatedConstraint_ReturnsFalse()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dim = new SymbolicDimension("batch_size", 33);
            var constraints = new List<IShapeConstraint>
            {
                new ModuloConstraint(8)
            };

            // Act
            var result = validator.ValidateDimension(dim, constraints);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ValidateDimension_WithNullDimension_ReturnsFalse()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var constraints = new List<IShapeConstraint>
            {
                new EqualityConstraint(32)
            };

            // Act
            var result = validator.ValidateDimension(null!, constraints);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ValidateDimension_WithNullConstraint_ReturnsFalse()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dim = new SymbolicDimension("batch_size", 32);
            var constraints = new List<IShapeConstraint?> { null };

            // Act
            var result = validator.ValidateDimension(dim, constraints!);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void ClearCache_ClearsValidationCache()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 33)
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>
            {
                ["batch_size"] = new List<IShapeConstraint>
                {
                    new ModuloConstraint(8)
                }
            };

            // Act
            var violations1 = validator.GetViolations(dims, constraints);
            validator.ClearCache();
            var violations2 = validator.GetViolations(dims, constraints);

            // Assert
            Assert.Equal(1, violations1.Count);
            Assert.Equal(1, violations2.Count);
        }

        [Fact]
        public void GetViolations_UsesCache()
        {
            // Arrange
            var validator = new ConstraintValidator();
            var dims = new List<SymbolicDimension>
            {
                new SymbolicDimension("batch_size", 33)
            };
            var constraints = new Dictionary<string, List<IShapeConstraint>>
            {
                ["batch_size"] = new List<IShapeConstraint>
                {
                    new ModuloConstraint(8)
                }
            };

            // Act
            var violations1 = validator.GetViolations(dims, constraints);
            var violations2 = validator.GetViolations(dims, constraints);

            // Assert
            Assert.Equal(1, violations1.Count);
            Assert.Equal(1, violations2.Count);
        }
    }
}
