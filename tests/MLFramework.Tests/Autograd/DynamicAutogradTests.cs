using MLFramework.Autograd;
using MLFramework.Autograd.Operations;
using MLFramework.Shapes;
using Xunit;

namespace MLFramework.Tests.Autograd
{
    /// <summary>
    /// Unit tests for dynamic autograd functionality with symbolic shapes.
    /// </summary>
    public class DynamicAutogradTests
    {
        #region SymbolicShape Tests

        [Fact]
        public void SymbolicShape_CreateWithKnownDimensions()
        {
            // Arrange
            var dims = new[]
            {
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("features", 128)
            };

            // Act
            var shape = new SymbolicShape(dims);

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.True(shape.IsFullyKnown());
            Assert.True(shape.IsPartiallyKnown());
        }

        [Fact]
        public void SymbolicShape_CreateWithUnknownDimensions()
        {
            // Arrange
            var dims = new[]
            {
                new SymbolicDimension("batch"),
                new SymbolicDimension("features", 128)
            };

            // Act
            var shape = new SymbolicShape(dims);

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.False(shape.IsFullyKnown());
            Assert.True(shape.IsPartiallyKnown());
        }

        [Fact]
        public void SymbolicShape_GetDimension_WithPositiveIndex()
        {
            // Arrange
            var dims = new[]
            {
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("features", 128)
            };
            var shape = new SymbolicShape(dims);

            // Act
            var dim = shape.GetDimension(0);

            // Assert
            Assert.Equal("batch", dim.Name);
            Assert.Equal(32, dim.Value);
        }

        [Fact]
        public void SymbolicShape_GetDimension_WithNegativeIndex()
        {
            // Arrange
            var dims = new[]
            {
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("features", 128)
            };
            var shape = new SymbolicShape(dims);

            // Act
            var dim = shape.GetDimension(-1);

            // Assert
            Assert.Equal("features", dim.Name);
            Assert.Equal(128, dim.Value);
        }

        [Fact]
        public void SymbolicShape_ToConcrete_ThrowsWhenNotFullyKnown()
        {
            // Arrange
            var dims = new[]
            {
                new SymbolicDimension("batch"),
                new SymbolicDimension("features", 128)
            };
            var shape = new SymbolicShape(dims);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => shape.ToConcrete());
        }

        [Fact]
        public void SymbolicShape_ToConcrete_ReturnsArrayWhenFullyKnown()
        {
            // Arrange
            var dims = new[]
            {
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("features", 128)
            };
            var shape = new SymbolicShape(dims);

            // Act
            var concrete = shape.ToConcrete();

            // Assert
            Assert.Equal(2, concrete.Length);
            Assert.Equal(32, concrete[0]);
            Assert.Equal(128, concrete[1]);
        }

        [Fact]
        public void SymbolicShape_Clone_CreatesIndependentCopy()
        {
            // Arrange
            var dims = new[]
            {
                new SymbolicDimension("batch", 32)
            };
            var shape = new SymbolicShape(dims);

            // Act
            var cloned = shape.Clone();

            // Assert
            Assert.NotSame(shape, cloned);
            Assert.Equal(shape, cloned);
        }

        #endregion

        #region SymbolicShapeFactory Tests

        [Fact]
        public void SymbolicShapeFactory_FromConcrete_CreatesSymbolicShape()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.FromConcrete(32, 128, 256);

            // Assert
            Assert.Equal(3, shape.Rank);
            Assert.True(shape.IsFullyKnown());
            Assert.Equal(32, shape.GetDimension(0).Value);
            Assert.Equal(128, shape.GetDimension(1).Value);
            Assert.Equal(256, shape.GetDimension(2).Value);
        }

        [Fact]
        public void SymbolicShapeFactory_Scalar_ReturnsRankZeroShape()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.Scalar();

            // Assert
            Assert.Equal(0, shape.Rank);
            Assert.True(shape.IsFullyKnown());
        }

        [Fact]
        public void SymbolicShapeFactory_Vector_ReturnsRankOneShape()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.Vector(128);

            // Assert
            Assert.Equal(1, shape.Rank);
            Assert.True(shape.IsFullyKnown());
            Assert.Equal(128, shape.GetDimension(0).Value);
        }

        [Fact]
        public void SymbolicShapeFactory_Matrix_ReturnsRankTwoShape()
        {
            // Arrange & Act
            var shape = SymbolicShapeFactory.Matrix(32, 128);

            // Assert
            Assert.Equal(2, shape.Rank);
            Assert.True(shape.IsFullyKnown());
            Assert.Equal(32, shape.GetDimension(0).Value);
            Assert.Equal(128, shape.GetDimension(1).Value);
        }

        [Fact]
        public void SymbolicShapeFactory_Batched_AddsBatchDimension()
        {
            // Arrange
            var innerShape = SymbolicShapeFactory.Matrix(128, 256);

            // Act
            var batchedShape = SymbolicShapeFactory.Batched("batch", innerShape);

            // Assert
            Assert.Equal(3, batchedShape.Rank);
            Assert.Equal("batch", batchedShape.GetDimension(0).Name);
            Assert.Equal(128, batchedShape.GetDimension(1).Value);
            Assert.Equal(256, batchedShape.GetDimension(2).Value);
        }

        #endregion

        #region ShapeComparer Tests

        [Fact]
        public void ShapeComparer_AreCompatible_ReturnsTrueForSameShapes()
        {
            // Arrange
            var shapeA = SymbolicShapeFactory.Matrix(32, 128);
            var shapeB = SymbolicShapeFactory.Matrix(32, 128);

            // Act
            var compatible = ShapeComparer.AreCompatible(shapeA, shapeB);

            // Assert
            Assert.True(compatible);
        }

        [Fact]
        public void ShapeComparer_AreCompatible_ReturnsTrueForBroadcastableShapes()
        {
            // Arrange
            var shapeA = SymbolicShapeFactory.Matrix(32, 1);
            var shapeB = SymbolicShapeFactory.Matrix(32, 128);

            // Act
            var compatible = ShapeComparer.AreCompatible(shapeA, shapeB);

            // Assert
            Assert.True(compatible);
        }

        [Fact]
        public void ShapeComparer_AreCompatible_ReturnsFalseForIncompatibleShapes()
        {
            // Arrange
            var shapeA = SymbolicShapeFactory.Matrix(32, 64);
            var shapeB = SymbolicShapeFactory.Matrix(32, 128);

            // Act
            var compatible = ShapeComparer.AreCompatible(shapeA, shapeB);

            // Assert
            Assert.False(compatible);
        }

        [Fact]
        public void ShapeComparer_GetBroadcastShape_ReturnsCorrectShape()
        {
            // Arrange
            var shapeA = SymbolicShapeFactory.Matrix(1, 128);
            var shapeB = SymbolicShapeFactory.Matrix(32, 1);

            // Act
            var broadcastShape = ShapeComparer.GetBroadcastShape(shapeA, shapeB);

            // Assert
            Assert.Equal(32, broadcastShape.GetDimension(0).Value);
            Assert.Equal(128, broadcastShape.GetDimension(1).Value);
        }

        [Fact]
        public void ShapeComparer_CanReshape_ReturnsTrueForSameTotalSize()
        {
            // Arrange
            var from = SymbolicShapeFactory.Matrix(32, 128);  // 4096 total
            var to = SymbolicShapeFactory.Vector(4096);

            // Act
            var canReshape = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.True(canReshape);
        }

        [Fact]
        public void ShapeComparer_CanReshape_ReturnsFalseForDifferentTotalSize()
        {
            // Arrange
            var from = SymbolicShapeFactory.Matrix(32, 128);  // 4096 total
            var to = SymbolicShapeFactory.Vector(4095);

            // Act
            var canReshape = ShapeComparer.CanReshape(from, to);

            // Assert
            Assert.False(canReshape);
        }

        #endregion

        #region DynamicGradientTensor Tests

        [Fact]
        public void DynamicGradientTensor_AccumulateGradient_StoresGradient()
        {
            // Arrange
            var shape = SymbolicShapeFactory.Vector(128);
            var tensor = CreateMockTensor();
            var dynamicTensor = new DynamicGradientTensor(tensor, shape, true);
            var gradient = CreateMockTensor();

            // Act
            dynamicTensor.AccumulateGradient(gradient);

            // Assert
            Assert.True(dynamicTensor.HasGradient);
            Assert.NotNull(dynamicTensor.Gradient);
        }

        [Fact]
        public void DynamicGradientTensor_AccumulateGradient_ThrowsWhenNotRequired()
        {
            // Arrange
            var shape = SymbolicShapeFactory.Vector(128);
            var tensor = CreateMockTensor();
            var dynamicTensor = new DynamicGradientTensor(tensor, shape, false);
            var gradient = CreateMockTensor();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                dynamicTensor.AccumulateGradient(gradient));
        }

        [Fact]
        public void DynamicGradientTensor_ClearGradient_RemovesGradient()
        {
            // Arrange
            var shape = SymbolicShapeFactory.Vector(128);
            var tensor = CreateMockTensor();
            var dynamicTensor = new DynamicGradientTensor(tensor, shape, true);
            var gradient = CreateMockTensor();
            dynamicTensor.AccumulateGradient(gradient);

            // Act
            dynamicTensor.ClearGradient();

            // Assert
            Assert.False(dynamicTensor.HasGradient);
            Assert.Null(dynamicTensor.Gradient);
        }

        [Fact]
        public void DynamicGradientTensor_Detach_DisablesGradientTracking()
        {
            // Arrange
            var shape = SymbolicShapeFactory.Vector(128);
            var tensor = CreateMockTensor();
            var dynamicTensor = new DynamicGradientTensor(tensor, shape, true);

            // Act
            var detached = dynamicTensor.Detach();

            // Assert
            Assert.False(detached.GradientRequired);
            Assert.True(dynamicTensor.GradientRequired);  // Original unchanged
        }

        #endregion

        #region DynamicAutogradContext Tests

        [Fact]
        public void DynamicAutogradContext_SaveForBackward_StoresTensors()
        {
            // Arrange
            var context = new DynamicAutogradContext();
            var tensor = CreateMockTensor();

            // Act
            context.SaveForBackward(tensor);

            // Assert
            Assert.Equal(1, context.SavedTensorCount);
            Assert.Same(tensor, context.GetSavedTensor(0));
        }

        [Fact]
        public void DynamicAutogradContext_SaveInputShape_StoresShapes()
        {
            // Arrange
            var context = new DynamicAutogradContext();
            var shape = SymbolicShapeFactory.Vector(128);

            // Act
            context.SaveInputShape(shape);

            // Assert
            Assert.Equal(1, context.InputShapeCount);
            Assert.Same(shape, context.GetInputShape(0));
        }

        [Fact]
        public void DynamicAutogradContext_RegisterGradientShape_StoresForValidation()
        {
            // Arrange
            var context = new DynamicAutogradContext();
            var shape = SymbolicShapeFactory.Vector(128);

            // Act
            context.RegisterGradientShape(0, shape);

            // Assert
            Assert.Same(shape, context.GetGradientShape(0));
        }

        [Fact]
        public void DynamicAutogradContext_ValidateGradient_PassesForMatchingShapes()
        {
            // Arrange
            var context = new DynamicAutogradContext();
            var shape = SymbolicShapeFactory.Vector(128);
            context.RegisterGradientShape(0, shape);

            // Act & Assert (should not throw)
            context.ValidateGradient(0, shape);
        }

        [Fact]
        public void DynamicAutogradContext_ValidateGradient_ThrowsForMismatchingShapes()
        {
            // Arrange
            var context = new DynamicAutogradContext();
            var shapeA = SymbolicShapeFactory.Vector(128);
            var shapeB = SymbolicShapeFactory.Vector(256);
            context.RegisterGradientShape(0, shapeA);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                context.ValidateGradient(0, shapeB));
        }

        [Fact]
        public void DynamicAutogradContext_Clear_RemovesAllSavedData()
        {
            // Arrange
            var context = new DynamicAutogradContext();
            context.SaveForBackward(CreateMockTensor());
            context.SaveInputShape(SymbolicShapeFactory.Vector(128));

            // Act
            context.Clear();

            // Assert
            Assert.Equal(0, context.SavedTensorCount);
            Assert.Equal(0, context.InputShapeCount);
        }

        #endregion

        #region GradientAccumulatorDynamic Tests

        [Fact]
        public void GradientAccumulatorDynamic_Accumulate_StoresGradient()
        {
            // Arrange
            var accumulator = new GradientAccumulatorDynamic();
            var shape = SymbolicShapeFactory.Vector(128);
            var gradient = CreateMockTensor();

            // Act
            accumulator.Accumulate("param1", gradient, shape);

            // Assert
            Assert.True(accumulator.HasGradient("param1"));
            Assert.NotNull(accumulator.GetAccumulated("param1"));
        }

        [Fact]
        public void GradientAccumulatorDynamic_Accumulate_IncrementsCount()
        {
            // Arrange
            var accumulator = new GradientAccumulatorDynamic();
            var shape = SymbolicShapeFactory.Vector(128);
            var gradient = CreateMockTensor();

            // Act
            accumulator.Accumulate("param1", gradient, shape);
            accumulator.Accumulate("param1", gradient, shape);

            // Assert
            Assert.Equal(2, accumulator.GetAccumulationCount("param1"));
        }

        [Fact]
        public void GradientAccumulatorDynamic_AccumulateBatched_StoresMultipleGradients()
        {
            // Arrange
            var accumulator = new GradientAccumulatorDynamic();
            var shape = SymbolicShapeFactory.Vector(128);
            var grads = new List<RitterFramework.Core.Tensor>
            {
                CreateMockTensor(),
                CreateMockTensor()
            };
            var shapes = new List<SymbolicShape> { shape, shape };

            // Act
            accumulator.AccumulateBatched("param1", grads, shapes);

            // Assert
            Assert.Equal(2, accumulator.GetAccumulationCount("param1"));
        }

        [Fact]
        public void GradientAccumulatorDynamic_Remove_RemovesGradient()
        {
            // Arrange
            var accumulator = new GradientAccumulatorDynamic();
            var shape = SymbolicShapeFactory.Vector(128);
            var gradient = CreateMockTensor();
            accumulator.Accumulate("param1", gradient, shape);

            // Act
            var removed = accumulator.Remove("param1");

            // Assert
            Assert.True(removed);
            Assert.False(accumulator.HasGradient("param1"));
        }

        [Fact]
        public void GradientAccumulatorDynamic_Reset_ClearsAllGradients()
        {
            // Arrange
            var accumulator = new GradientAccumulatorDynamic();
            var shape = SymbolicShapeFactory.Vector(128);
            accumulator.Accumulate("param1", CreateMockTensor(), shape);
            accumulator.Accumulate("param2", CreateMockTensor(), shape);

            // Act
            accumulator.Reset();

            // Assert
            Assert.Equal(0, accumulator.Count);
        }

        #endregion

        #region DynamicMatMulBackward Tests

        [Fact]
        public void DynamicMatMulBackward_GetOutputShape_ComputesCorrectShape()
        {
            // Arrange
            var backward = new DynamicMatMulBackward();
            var shapeA = SymbolicShapeFactory.Matrix(32, 64);  // [32, 64]
            var shapeB = SymbolicShapeFactory.Matrix(64, 128);  // [64, 128]
            var inputShapes = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputShape = backward.GetOutputShape(inputShapes);

            // Assert
            Assert.Equal(2, outputShape.Rank);
            Assert.Equal(32, outputShape.GetDimension(0).Value);  // M from A
            Assert.Equal(128, outputShape.GetDimension(1).Value); // N from B
        }

        [Fact]
        public void DynamicMatMulBackward_GetOutputShape_WithBatchDimensions()
        {
            // Arrange
            var backward = new DynamicMatMulBackward();
            var shapeA = new SymbolicShape(
                new SymbolicDimension("batch"),
                new SymbolicDimension("M"),
                new SymbolicDimension("K"));
            var shapeB = new SymbolicShape(
                new SymbolicDimension("batch"),
                new SymbolicDimension("K"),
                new SymbolicDimension("N"));
            var inputShapes = new List<SymbolicShape> { shapeA, shapeB };

            // Act
            var outputShape = backward.GetOutputShape(inputShapes);

            // Assert
            Assert.Equal(3, outputShape.Rank);
            Assert.Equal("batch", outputShape.GetDimension(0).Name);
            Assert.Equal("M", outputShape.GetDimension(1).Name);
            Assert.Equal("N", outputShape.GetDimension(2).Name);
        }

        [Fact]
        public void DynamicMatMulBackward_ValidateGradientShape_DoesNotThrowForValidShape()
        {
            // Arrange
            var backward = new DynamicMatMulBackward();
            var shape = SymbolicShapeFactory.Matrix(32, 128);

            // Act & Assert (should not throw)
            backward.ValidateGradientShape(shape);
        }

        [Fact]
        public void DynamicMatMulBackward_ValidateGradientShape_ThrowsForInvalidRank()
        {
            // Arrange
            var backward = new DynamicMatMulBackward();
            var shape = SymbolicShapeFactory.Vector(128);  // Rank 1

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                backward.ValidateGradientShape(shape));
        }

        #endregion

        #region DynamicConv2DBackward Tests

        [Fact]
        public void DynamicConv2DBackward_GetOutputShape_ComputesCorrectShape()
        {
            // Arrange
            var backward = new DynamicConv2DBackward(
                kernelSize: new[] { 3, 3 },
                stride: new[] { 1, 1 },
                padding: new[] { 0, 0 });
            var inputShape = new SymbolicShape(
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("in_channels", 3),
                new SymbolicDimension("height", 32),
                new SymbolicDimension("width", 32));
            var weightShape = new SymbolicShape(
                new SymbolicDimension("out_channels", 64),
                new SymbolicDimension("in_channels", 3),
                new SymbolicDimension("kernel_h", 3),
                new SymbolicDimension("kernel_w", 3));
            var inputShapes = new List<SymbolicShape> { inputShape, weightShape };

            // Act
            var outputShape = backward.GetOutputShape(inputShapes);

            // Assert
            Assert.Equal(4, outputShape.Rank);
            Assert.Equal(32, outputShape.GetDimension(0).Value);  // batch
            Assert.Equal(64, outputShape.GetDimension(1).Value); // out_channels
            Assert.Equal(30, outputShape.GetDimension(2).Value);  // height (32 - 3 + 1)
            Assert.Equal(30, outputShape.GetDimension(3).Value);  // width (32 - 3 + 1)
        }

        [Fact]
        public void DynamicConv2DBackward_ValidateGradientShape_DoesNotThrowForValidShape()
        {
            // Arrange
            var backward = new DynamicConv2DBackward(new[] { 3, 3 });
            var shape = new SymbolicShape(
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("out_channels", 64),
                new SymbolicDimension("height", 30),
                new SymbolicDimension("width", 30));

            // Act & Assert (should not throw)
            backward.ValidateGradientShape(shape);
        }

        #endregion

        #region DynamicReshapeBackward Tests

        [Fact]
        public void DynamicReshapeBackward_GetOutputShape_ReturnsOriginalShape()
        {
            // Arrange
            var originalShape = SymbolicShapeFactory.Matrix(32, 128);
            var backward = new DynamicReshapeBackward(originalShape);
            var inputShape = SymbolicShapeFactory.Vector(4096);
            var inputShapes = new List<SymbolicShape> { inputShape };

            // Act
            var outputShape = backward.GetOutputShape(inputShapes);

            // Assert
            Assert.Same(originalShape, outputShape);
        }

        [Fact]
        public void DynamicReshapeBackward_ValidateGradientShape_PassesForCompatibleShape()
        {
            // Arrange
            var originalShape = SymbolicShapeFactory.Matrix(32, 128);  // 4096 total
            var backward = new DynamicReshapeBackward(originalShape);
            var gradientShape = SymbolicShapeFactory.Vector(4096);   // 4096 total

            // Act & Assert (should not throw)
            backward.ValidateGradientShape(gradientShape);
        }

        [Fact]
        public void DynamicReshapeBackward_ValidateGradientShape_ThrowsForIncompatibleShape()
        {
            // Arrange
            var originalShape = SymbolicShapeFactory.Matrix(32, 128);  // 4096 total
            var backward = new DynamicReshapeBackward(originalShape);
            var gradientShape = SymbolicShapeFactory.Vector(4095);   // Different size

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                backward.ValidateGradientShape(gradientShape));
        }

        #endregion

        #region DynamicBroadcastBackward Tests

        [Fact]
        public void DynamicBroadcastBackward_GetOutputShape_ReturnsBroadcastedShape()
        {
            // Arrange
            var inputShape = SymbolicShapeFactory.Matrix(1, 128);
            var outputShape = SymbolicShapeFactory.Matrix(32, 128);
            var backward = new DynamicBroadcastBackward(inputShape, outputShape);
            var inputShapes = new List<SymbolicShape> { inputShape };

            // Act
            var result = backward.GetOutputShape(inputShapes);

            // Assert
            Assert.Same(outputShape, result);
        }

        [Fact]
        public void DynamicBroadcastBackward_ValidateGradientShape_PassesForMatchingShape()
        {
            // Arrange
            var inputShape = SymbolicShapeFactory.Matrix(1, 128);
            var outputShape = SymbolicShapeFactory.Matrix(32, 128);
            var backward = new DynamicBroadcastBackward(inputShape, outputShape);

            // Act & Assert (should not throw)
            backward.ValidateGradientShape(outputShape);
        }

        #endregion

        #region Helper Methods

        private RitterFramework.Core.Tensor CreateMockTensor()
        {
            // This is a placeholder - in practice, you'd create a real tensor
            // For now, we'll just create a mock object
            return null!;
        }

        #endregion
    }
}
