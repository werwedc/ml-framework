using System;
using System.Linq;
using Xunit;
using MLFramework.Shapes;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Shapes
{
    /// <summary>
    /// Unit tests for symbolic broadcasting functionality.
    /// </summary>
    public class SymbolicBroadcastingTests
    {
        private readonly SymbolicBroadcastingEngine _engine;

        public SymbolicBroadcastingTests()
        {
            _engine = new SymbolicBroadcastingEngine();
        }

        #region BroadcastingRule Tests

        [Fact]
        public void BroadcastingRule_Apply_EqualDimensions_ReturnsSameDimension()
        {
            // Arrange
            var dim1 = SymbolicDimensionFactory.CreateKnown(5);
            var dim2 = SymbolicDimensionFactory.CreateKnown(5);

            // Act
            var result = BroadcastingRule.Apply(dim1, dim2);

            // Assert
            Assert.Equal(5, result.Value);
            Assert.Equal(dim1.Name, result.Name);
        }

        [Fact]
        public void BroadcastingRule_Apply_OneDimensionIsOne_ReturnsOtherDimension()
        {
            // Arrange
            var dim1 = SymbolicDimensionFactory.CreateKnown(1);
            var dim2 = SymbolicDimensionFactory.CreateKnown(5);

            // Act
            var result = BroadcastingRule.Apply(dim1, dim2);

            // Assert
            Assert.Equal(5, result.Value);
            Assert.Equal(dim2.Name, result.Name);
        }

        [Fact]
        public void BroadcastingRule_Apply_BothDimensionsAreOne_ReturnsOne()
        {
            // Arrange
            var dim1 = SymbolicDimensionFactory.CreateKnown(1);
            var dim2 = SymbolicDimensionFactory.CreateKnown(1);

            // Act
            var result = BroadcastingRule.Apply(dim1, dim2);

            // Assert
            Assert.Equal(1, result.Value);
        }

        [Fact]
        public void BroadcastingRule_Apply_SymbolicDimension_ReturnsSymbolicWithConstraint()
        {
            // Arrange
            var dim1 = new SymbolicDimension("batch_size");
            var dim2 = SymbolicDimensionFactory.CreateKnown(5);

            // Act
            var result = BroadcastingRule.Apply(dim1, dim2);

            // Assert
            Assert.False(result.Value.HasValue);
            Assert.Equal("batch_size", result.Name);
            Assert.Equal(1, result.MinValue);
        }

        [Fact]
        public void BroadcastingRule_Apply_IncompatibleDimensions_ThrowsException()
        {
            // Arrange
            var dim1 = SymbolicDimensionFactory.CreateKnown(3);
            var dim2 = SymbolicDimensionFactory.CreateKnown(5);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => BroadcastingRule.Apply(dim1, dim2));
        }

        #endregion

        #region CanBroadcast Tests

        [Fact]
        public void CanBroadcast_SameShapes_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            bool result = _engine.CanBroadcast(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanBroadcast_ScalarAndMatrix_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(); // scalar

            // Act
            bool result = _engine.CanBroadcast(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanBroadcast_VectorAndMatrix_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(4)); // (4,) broadcasts to (3, 4)

            // Act
            bool result = _engine.CanBroadcast(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanBroadcast_OneDimension_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(5),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(1),
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            bool result = _engine.CanBroadcast(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanBroadcast_SymbolicDimensions_ReturnsTrue()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            bool result = _engine.CanBroadcast(shape1, shape2);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void CanBroadcast_IncompatibleShapes_ReturnsFalse()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(4),
                SymbolicDimensionFactory.CreateKnown(3));

            // Act
            bool result = _engine.CanBroadcast(shape1, shape2);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void CanBroadcast_DifferentConcreteValues_ReturnsFalse()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(5),
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            bool result = _engine.CanBroadcast(shape1, shape2);

            // Assert
            Assert.False(result);
        }

        #endregion

        #region GetBroadcastShape Tests

        [Fact]
        public void GetBroadcastShape_SameShapes_ReturnsSameShape()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            var result = _engine.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(2, result.Rank);
            Assert.Equal(3, result.GetDimension(0).Value);
            Assert.Equal(4, result.GetDimension(1).Value);
        }

        [Fact]
        public void GetBroadcastShape_ScalarAndMatrix_ReturnsMatrix()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(); // scalar

            // Act
            var result = _engine.GetBroadcastShape(shape1, shape2);

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
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            var result = _engine.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(2, result.Rank);
            Assert.Equal(3, result.GetDimension(0).Value);
            Assert.Equal(4, result.GetDimension(1).Value);
        }

        [Fact]
        public void GetBroadcastShape_WithOneDimension_ReturnsLargerDimension()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(5),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(1),
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            var result = _engine.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.Equal(2, result.Rank);
            Assert.Equal(5, result.GetDimension(0).Value);
            Assert.Equal(4, result.GetDimension(1).Value);
        }

        [Fact]
        public void GetBroadcastShape_IncompatibleShapes_ThrowsException()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(5),
                SymbolicDimensionFactory.CreateKnown(4));

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => _engine.GetBroadcastShape(shape1, shape2));
        }

        #endregion

        #region GetBroadcastPlan Tests

        [Fact]
        public void GetBroadcastPlan_SameShapes_ReturnsPlanWithEqualRules()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            var plan = _engine.GetBroadcastPlan(shape1, shape2);

            // Assert
            Assert.Equal(2, plan.Count);
            Assert.True(plan[0].IsBroadcastable);
            Assert.True(plan[1].IsBroadcastable);
        }

        [Fact]
        public void GetBroadcastPlan_ScalarAndMatrix_ReturnsPlanWithBroadcastRules()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(); // scalar

            // Act
            var plan = _engine.GetBroadcastPlan(shape1, shape2);

            // Assert
            Assert.Equal(2, plan.Count);
            Assert.All(plan, rule => Assert.True(rule.IsBroadcastable));
        }

        [Fact]
        public void GetBroadcastPlan_CachesPlan_ReturnsSameInstance()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            var plan1 = _engine.GetBroadcastPlan(shape1, shape2);
            var plan2 = _engine.GetBroadcastPlan(shape1, shape2);

            // Assert
            Assert.Same(plan1, plan2);
        }

        #endregion

        #region InferBroadcastConstraints Tests

        [Fact]
        public void InferBroadcastConstraints_ConcreteShapes_ReturnsEmpty()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            var constraints = _engine.InferBroadcastConstraints(shape1, shape2);

            // Assert
            Assert.Empty(constraints);
        }

        [Fact]
        public void InferBroadcastConstraints_SymbolicDimensions_ReturnsConstraints()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                SymbolicDimensionFactory.CreateKnown(4));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                SymbolicDimensionFactory.CreateKnown(4));

            // Act
            var constraints = _engine.InferBroadcastConstraints(shape1, shape2);

            // Assert
            Assert.NotEmpty(constraints);
        }

        #endregion

        #region BroadcastedTensor Tests

        [Fact]
        public void BroadcastedTensor_Materialize_SameShape_ReturnsOriginalTensor()
        {
            // Arrange
            var data = new float[] { 1, 2, 3, 4, 5, 6 };
            var tensor = new Tensor(data, new int[] { 2, 3 });
            var shape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(2),
                SymbolicDimensionFactory.CreateKnown(3));
            var plan = _engine.GetBroadcastPlan(shape, shape);
            var broadcasted = new BroadcastedTensor(tensor, shape, shape, plan);

            // Act
            var result = broadcasted.Materialize();

            // Assert
            Assert.Same(tensor, result);
        }

        [Fact]
        public void BroadcastedTensor_Materialize_ScalarToMatrix_BroadcastsCorrectly()
        {
            // Arrange
            var data = new float[] { 5 };
            var tensor = new Tensor(data, new int[] { 1 });
            var originalShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(1));
            var broadcastShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(2),
                SymbolicDimensionFactory.CreateKnown(3));
            var plan = _engine.GetBroadcastPlan(originalShape, broadcastShape);
            var broadcasted = new BroadcastedTensor(tensor, originalShape, broadcastShape, plan);

            // Act
            var result = broadcasted.Materialize();

            // Assert
            Assert.Equal(new int[] { 2, 3 }, result.Shape);
            Assert.Equal(6, result.Data.Length);
            Assert.All(result.Data, value => Assert.Equal(5, value));
        }

        [Fact]
        public void BroadcastedTensor_Materialize_VectorToMatrix_BroadcastsCorrectly()
        {
            // Arrange
            var data = new float[] { 1, 2, 3 };
            var tensor = new Tensor(data, new int[] { 3 });
            var originalShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3));
            var broadcastShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(2),
                SymbolicDimensionFactory.CreateKnown(3));
            var plan = _engine.GetBroadcastPlan(originalShape, broadcastShape);
            var broadcasted = new BroadcastedTensor(tensor, originalShape, broadcastShape, plan);

            // Act
            var result = broadcasted.Materialize();

            // Assert
            Assert.Equal(new int[] { 2, 3 }, result.Shape);
            Assert.Equal(6, result.Data.Length);
            Assert.Equal(1, result.Data[0]);
            Assert.Equal(2, result.Data[1]);
            Assert.Equal(3, result.Data[2]);
            Assert.Equal(1, result.Data[3]);
            Assert.Equal(2, result.Data[4]);
            Assert.Equal(3, result.Data[5]);
        }

        [Fact]
        public void BroadcastedTensor_GetStrides_ReturnsCorrectStrides()
        {
            // Arrange
            var data = new float[] { 1, 2, 3 };
            var tensor = new Tensor(data, new int[] { 3 });
            var originalShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3));
            var broadcastShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(2),
                SymbolicDimensionFactory.CreateKnown(3));
            var plan = _engine.GetBroadcastPlan(originalShape, broadcastShape);
            var broadcasted = new BroadcastedTensor(tensor, originalShape, broadcastShape, plan);

            // Act
            var strides = broadcasted.GetStrides();

            // Assert
            Assert.Equal(2, strides.Length);
            Assert.Equal(0, strides[0]); // Broadcast dimension
            Assert.Equal(1, strides[1]); // Non-broadcast dimension
        }

        [Fact]
        public void BroadcastedTensor_GetBroadcastSources_ReturnsCorrectSources()
        {
            // Arrange
            var data = new float[] { 1, 2, 3 };
            var tensor = new Tensor(data, new int[] { 3 });
            var originalShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3));
            var broadcastShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(2),
                SymbolicDimensionFactory.CreateKnown(3));
            var plan = _engine.GetBroadcastPlan(originalShape, broadcastShape);
            var broadcasted = new BroadcastedTensor(tensor, originalShape, broadcastShape, plan);

            // Act
            var sources = broadcasted.GetBroadcastSources();

            // Assert
            Assert.Single(sources);
            Assert.Equal(0, sources[0]);
        }

        [Fact]
        public void BroadcastedTensor_RequiresMaterialization_NoBroadcast_ReturnsFalse()
        {
            // Arrange
            var data = new float[] { 1, 2, 3, 4 };
            var tensor = new Tensor(data, new int[] { 2, 2 });
            var shape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(2),
                SymbolicDimensionFactory.CreateKnown(2));
            var plan = _engine.GetBroadcastPlan(shape, shape);
            var broadcasted = new BroadcastedTensor(tensor, shape, shape, plan);

            // Act
            bool result = broadcasted.RequiresMaterialization();

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void BroadcastedTensor_RequiresMaterialization_WithBroadcast_ReturnsTrue()
        {
            // Arrange
            var data = new float[] { 1, 2, 3 };
            var tensor = new Tensor(data, new int[] { 3 });
            var originalShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3));
            var broadcastShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(2),
                SymbolicDimensionFactory.CreateKnown(3));
            var plan = _engine.GetBroadcastPlan(originalShape, broadcastShape);
            var broadcasted = new BroadcastedTensor(tensor, originalShape, broadcastShape, plan);

            // Act
            bool result = broadcasted.RequiresMaterialization();

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void BroadcastedTensor_Materialize_SymbolicShape_ThrowsException()
        {
            // Arrange
            var data = new float[] { 1, 2, 3 };
            var tensor = new Tensor(data, new int[] { 3 });
            var originalShape = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3));
            var broadcastShape = new SymbolicShape(
                new SymbolicDimension("batch_size"),
                SymbolicDimensionFactory.CreateKnown(3));
            var plan = _engine.GetBroadcastPlan(originalShape, broadcastShape);
            var broadcasted = new BroadcastedTensor(tensor, originalShape, broadcastShape, plan);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => broadcasted.Materialize());
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void Broadcasting_EmptyShape_Scalar_ReturnsScalar()
        {
            // Arrange
            var shape1 = new SymbolicShape(); // scalar
            var shape2 = new SymbolicShape(); // scalar

            // Act
            bool canBroadcast = _engine.CanBroadcast(shape1, shape2);
            var broadcastShape = _engine.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.True(canBroadcast);
            Assert.Equal(0, broadcastShape.Rank);
        }

        [Fact]
        public void Broadcasting_OneDimension_To_MultipleDimensions()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(1),
                SymbolicDimensionFactory.CreateKnown(1),
                SymbolicDimensionFactory.CreateKnown(5));
            var shape2 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(3),
                SymbolicDimensionFactory.CreateKnown(4),
                SymbolicDimensionFactory.CreateKnown(5));

            // Act
            bool canBroadcast = _engine.CanBroadcast(shape1, shape2);
            var broadcastShape = _engine.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.True(canBroadcast);
            Assert.Equal(3, broadcastShape.Rank);
            Assert.Equal(3, broadcastShape.GetDimension(0).Value);
            Assert.Equal(4, broadcastShape.GetDimension(1).Value);
            Assert.Equal(5, broadcastShape.GetDimension(2).Value);
        }

        [Fact]
        public void Broadcasting_MixedSymbolicAndConcrete_WorksCorrectly()
        {
            // Arrange
            var shape1 = new SymbolicShape(
                SymbolicDimensionFactory.CreateKnown(1),
                new SymbolicDimension("seq_len", 100));
            var shape2 = new SymbolicShape(
                new SymbolicDimension("batch_size", 32),
                new SymbolicDimension("seq_len", 100));

            // Act
            bool canBroadcast = _engine.CanBroadcast(shape1, shape2);
            var broadcastShape = _engine.GetBroadcastShape(shape1, shape2);

            // Assert
            Assert.True(canBroadcast);
            Assert.Equal(2, broadcastShape.Rank);
            Assert.Equal(32, broadcastShape.GetDimension(0).Value);
            Assert.Equal(100, broadcastShape.GetDimension(1).Value);
        }

        #endregion
    }
}
