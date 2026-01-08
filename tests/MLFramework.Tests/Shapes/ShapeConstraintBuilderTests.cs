using Xunit;
using MLFramework.Shapes;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for ShapeConstraintBuilder class.
    /// </summary>
    public class ShapeConstraintBuilderTests
    {
        [Fact]
        public void Create_InitializesWithEmptyConstraints()
        {
            // Arrange & Act
            var builder = new ShapeConstraintBuilder();
            var constraints = builder.Build();

            // Assert
            Assert.NotNull(constraints);
            Assert.Empty(constraints);
        }

        [Fact]
        public void Min_AddsMinConstraint()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();

            // Act
            builder.Min(10);
            var constraints = builder.Build();

            // Assert
            Assert.Single(constraints);
            Assert.IsType<RangeConstraint>(constraints[0]);
            var rangeConstraint = (RangeConstraint)constraints[0];
            Assert.Equal(10, rangeConstraint.MinValue);
        }

        [Fact]
        public void Min_WithNegativeValue_ThrowsArgumentException()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => builder.Min(-1));
        }

        [Fact]
        public void Max_AddsMaxConstraint()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();

            // Act
            builder.Max(100);
            var constraints = builder.Build();

            // Assert
            Assert.Single(constraints);
            Assert.IsType<RangeConstraint>(constraints[0]);
            var rangeConstraint = (RangeConstraint)constraints[0];
            Assert.Equal(100, rangeConstraint.MaxValue);
        }

        [Fact]
        public void Max_WithNegativeValue_ThrowsArgumentException()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => builder.Max(-1));
        }

        [Fact]
        public void Range_AddsRangeConstraint()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();

            // Act
            builder.Range(10, 100);
            var constraints = builder.Build();

            // Assert
            Assert.Single(constraints);
            Assert.IsType<RangeConstraint>(constraints[0]);
            var rangeConstraint = (RangeConstraint)constraints[0];
            Assert.Equal(10, rangeConstraint.MinValue);
            Assert.Equal(100, rangeConstraint.MaxValue);
        }

        [Fact]
        public void Equal_AddsEqualityConstraint()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();

            // Act
            builder.Equal(42);
            var constraints = builder.Build();

            // Assert
            Assert.Single(constraints);
            Assert.IsType<EqualityConstraint>(constraints[0]);
            var equalityConstraint = (EqualityConstraint)constraints[0];
            Assert.Equal(42, equalityConstraint.TargetValue);
        }

        [Fact]
        public void Modulo_AddsModuloConstraint()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();

            // Act
            builder.Modulo(8);
            var constraints = builder.Build();

            // Assert
            Assert.Single(constraints);
            Assert.IsType<ModuloConstraint>(constraints[0]);
            var moduloConstraint = (ModuloConstraint)constraints[0];
            Assert.Equal(8, moduloConstraint.Divisor);
        }

        [Fact]
        public void AddConstraint_AddsCustomConstraint()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();
            var customConstraint = new EqualityConstraint(42);

            // Act
            builder.AddConstraint(customConstraint);
            var constraints = builder.Build();

            // Assert
            Assert.Single(constraints);
            Assert.Same(customConstraint, constraints[0]);
        }

        [Fact]
        public void AddConstraint_WithNull_ThrowsArgumentNullException()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => builder.AddConstraint(null!));
        }

        [Fact]
        public void FluentAPI_SupportsMethodChaining()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();

            // Act
            var constraints = builder
                .Min(10)
                .Max(100)
                .Modulo(8)
                .Build();

            // Assert
            Assert.Equal(3, constraints.Count);
        }

        [Fact]
        public void Build_ReturnsNewList()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();
            builder.Min(10);
            var constraints1 = builder.Build();

            // Act
            constraints1.Clear();
            var constraints2 = builder.Build();

            // Assert
            Assert.Empty(constraints1);
            Assert.Single(constraints2);
        }

        [Fact]
        public void Reset_ClearsAllConstraints()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();
            builder.Min(10).Max(100);

            // Act
            builder.Reset();
            var constraints = builder.Build();

            // Assert
            Assert.Empty(constraints);
        }

        [Fact]
        public void Build_AfterReset_ReturnsNewConstraints()
        {
            // Arrange
            var builder = new ShapeConstraintBuilder();
            builder.Min(10).Max(100);

            // Act
            builder.Reset().Equal(42);
            var constraints = builder.Build();

            // Assert
            Assert.Single(constraints);
            Assert.IsType<EqualityConstraint>(constraints[0]);
        }
    }
}
