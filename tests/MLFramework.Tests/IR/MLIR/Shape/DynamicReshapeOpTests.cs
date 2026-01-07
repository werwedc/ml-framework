using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.MLIR.Shape;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.MLIR.Shape
{
    [TestFixture]
    public class DynamicReshapeOpTests
    {
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _context = new IRContext();
        }

        [Test]
        public void DynamicReshapeOp_CreatesCorrectOperation()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Int32, new[] { 3 });
            var input = _context.CreateValue(inputType, "input");
            var shape = _context.CreateValue(shapeType, "shape");
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType, "result");

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.AreEqual(input, reshapeOp.Input);
            Assert.AreEqual(shape, reshapeOp.Shape);
            Assert.AreEqual(result, reshapeOp.Result);
        }

        [Test]
        public void DynamicReshapeOp_Validate_DoesNotThrowForValidReshape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Int32, new[] { 3 });
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(shapeType);
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType);

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.DoesNotThrow(() => reshapeOp.Validate());
        }

        [Test]
        public void DynamicReshapeOp_Validate_SupportsDifferentDimensions()
        {
            var inputShapes = new[]
            {
                new[] { 2, 3, 4 },
                new[] { 12 },
                new[] { 2, 6 },
                new[] { 4, 6 },
                new[] { 24, 1 }
            };

            foreach (var inputShape in inputShapes)
            {
                var inputType = new TensorType(DataType.Float32, inputShape);
                var shapeType = new TensorType(DataType.Int32, new[] { 2 });
                var input = _context.CreateValue(inputType);
                var shape = _context.CreateValue(shapeType);
                var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
                var result = _context.CreateValue(resultType);

                var reshapeOp = new DynamicReshapeOp(input, shape, result);
                Assert.DoesNotThrow(() => reshapeOp.Validate());
            }
        }

        [Test]
        public void DynamicReshapeOp_Validate_ThrowsForNon1DShape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Int32, new[] { 3, 2 });  // 2D shape
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(shapeType);
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType);

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.Throws<System.InvalidOperationException>(() => reshapeOp.Validate());
        }

        [Test]
        public void DynamicReshapeOp_Validate_ThrowsForNonIntegerShape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Float32, new[] { 3 });  // Wrong type
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(shapeType);
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType);

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.Throws<System.InvalidOperationException>(() => reshapeOp.Validate());
        }

        [Test]
        public void DynamicReshapeOp_Validate_SupportsInt64Shape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Int64, new[] { 3 });
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(shapeType);
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType);

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.DoesNotThrow(() => reshapeOp.Validate());
        }

        [Test]
        public void DynamicReshapeOp_Validate_ThrowsForMismatchedElementType()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Int32, new[] { 3 });
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(shapeType);
            var resultType = new TensorType(DataType.Int32, new[] { 6, 4 });  // Wrong type
            var result = _context.CreateValue(resultType);

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.Throws<System.InvalidOperationException>(() => reshapeOp.Validate());
        }

        [Test]
        public void DynamicReshapeOp_Validate_ThrowsForNonTensorInput()
        {
            var input = _context.CreateValue(new Attributes.ScalarType(DataType.Float32));
            var shapeType = new TensorType(DataType.Int32, new[] { 3 });
            var shape = _context.CreateValue(shapeType);
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType);

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.Throws<System.InvalidOperationException>(() => reshapeOp.Validate());
        }

        [Test]
        public void DynamicReshapeOp_Validate_ThrowsForNonTensorShape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(new Attributes.ScalarType(DataType.Int32));
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType);

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.Throws<System.InvalidOperationException>(() => reshapeOp.Validate());
        }

        [Test]
        public void DynamicReshapeOp_Validate_ThrowsForNonTensorResult()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Int32, new[] { 3 });
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(shapeType);
            var result = _context.CreateValue(new Attributes.ScalarType(DataType.Float32));

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.Throws<System.InvalidOperationException>(() => reshapeOp.Validate());
        }

        [Test]
        public void DynamicReshapeOp_Construction_ThrowsForNullInput()
        {
            var shapeType = new TensorType(DataType.Int32, new[] { 3 });
            var shape = _context.CreateValue(shapeType);
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new DynamicReshapeOp(null, shape, result));
        }

        [Test]
        public void DynamicReshapeOp_Construction_ThrowsForNullShape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var input = _context.CreateValue(inputType);
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new DynamicReshapeOp(input, null, result));
        }

        [Test]
        public void DynamicReshapeOp_Construction_ThrowsForNullResult()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Int32, new[] { 3 });
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(shapeType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new DynamicReshapeOp(input, shape, null));
        }

        [Test]
        public void DynamicReshapeOp_Clone_CreatesIndependentCopy()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Int32, new[] { 3 });
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(shapeType);
            var resultType = new TensorType(DataType.Float32, new[] { 6, 4 });
            var result = _context.CreateValue(resultType);

            var original = new DynamicReshapeOp(input, shape, result);
            var cloned = (DynamicReshapeOp)original.Clone();

            Assert.AreEqual(original.Input, cloned.Input);
            Assert.AreEqual(original.Shape, cloned.Shape);
            Assert.AreEqual(original.Result, cloned.Result);
        }

        [Test]
        public void DynamicReshapeOp_Validate_SupportsDifferentDataTypes()
        {
            var dataTypes = new[] { DataType.Float16, DataType.Float32, DataType.Float64,
                                   DataType.Int8, DataType.Int16, DataType.Int32, DataType.Int64 };

            foreach (var dataType in dataTypes)
            {
                var inputType = new TensorType(dataType, new[] { 2, 3, 4 });
                var shapeType = new TensorType(DataType.Int32, new[] { 3 });
                var input = _context.CreateValue(inputType);
                var shape = _context.CreateValue(shapeType);
                var resultType = new TensorType(dataType, new[] { 6, 4 });
                var result = _context.CreateValue(resultType);

                var reshapeOp = new DynamicReshapeOp(input, shape, result);
                Assert.DoesNotThrow(() => reshapeOp.Validate());
            }
        }

        [Test]
        public void DynamicReshapeOp_Validate_PreservesElementShape()
        {
            // Verify that element-wise reshaping is allowed (same number of elements)
            var inputType = new TensorType(DataType.Float32, new[] { 2, 3, 4 });
            var shapeType = new TensorType(DataType.Int32, new[] { 4 });
            var input = _context.CreateValue(inputType);
            var shape = _context.CreateValue(shapeType);
            var resultType = new TensorType(DataType.Float32, new[] { 1, 24 });
            var result = _context.CreateValue(resultType);

            var reshapeOp = new DynamicReshapeOp(input, shape, result);

            Assert.DoesNotThrow(() => reshapeOp.Validate());
        }
    }
}
