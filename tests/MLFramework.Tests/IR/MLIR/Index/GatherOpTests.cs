using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.MLIR.Index;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.MLIR.Index
{
    [TestFixture]
    public class GatherOpTests
    {
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _context = new IRContext();
        }

        [Test]
        public void GatherOp_CreatesCorrectOperation()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Int32, new[] { 5 });
            var input = _context.CreateValue(inputType, "input");
            var indices = _context.CreateValue(indicesType, "indices");
            var resultType = new TensorType(DataType.Float32, new[] { 5, 20, 30 });
            var result = _context.CreateValue(resultType, "result");

            var gatherOp = new GatherOp(input, indices, result, 0);

            Assert.AreEqual(input, gatherOp.Input);
            Assert.AreEqual(indices, gatherOp.Indices);
            Assert.AreEqual(result, gatherOp.Result);
            Assert.AreEqual(0, gatherOp.Axis);
        }

        [Test]
        public void GatherOp_Validate_DoesNotThrowForValidGather()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Int32, new[] { 5 });
            var input = _context.CreateValue(inputType);
            var indices = _context.CreateValue(indicesType);
            var resultType = new TensorType(DataType.Float32, new[] { 5, 20, 30 });
            var result = _context.CreateValue(resultType);

            var gatherOp = new GatherOp(input, indices, result, 0);

            Assert.DoesNotThrow(() => gatherOp.Validate());
        }

        [Test]
        public void GatherOp_Validate_SupportsDifferentAxes()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Int32, new[] { 5 });

            for (int axis = 0; axis < 3; axis++)
            {
                var input = _context.CreateValue(inputType);
                var indices = _context.CreateValue(indicesType);
                var resultType = new TensorType(DataType.Float32,
                    axis == 0 ? new[] { 5, 20, 30 } : axis == 1 ? new[] { 10, 5, 30 } : new[] { 10, 20, 5 });
                var result = _context.CreateValue(resultType);

                var gatherOp = new GatherOp(input, indices, result, axis);
                Assert.DoesNotThrow(() => gatherOp.Validate());
            }
        }

        [Test]
        public void GatherOp_Validate_ThrowsForAxisOutOfBounds()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Int32, new[] { 5 });
            var input = _context.CreateValue(inputType);
            var indices = _context.CreateValue(indicesType);
            var resultType = new TensorType(DataType.Float32, new[] { 5, 20, 30 });
            var result = _context.CreateValue(resultType);

            var gatherOp = new GatherOp(input, indices, result, 5);  // Out of bounds

            Assert.Throws<System.InvalidOperationException>(() => gatherOp.Validate());
        }

        [Test]
        public void GatherOp_Validate_ThrowsForNegativeAxis()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Int32, new[] { 5 });
            var input = _context.CreateValue(inputType);
            var indices = _context.CreateValue(indicesType);
            var resultType = new TensorType(DataType.Float32, new[] { 5, 20, 30 });
            var result = _context.CreateValue(resultType);

            var gatherOp = new GatherOp(input, indices, result, -1);

            Assert.Throws<System.InvalidOperationException>(() => gatherOp.Validate());
        }

        [Test]
        public void GatherOp_Validate_ThrowsForNonIntegerIndices()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Float32, new[] { 5 });  // Wrong type
            var input = _context.CreateValue(inputType);
            var indices = _context.CreateValue(indicesType);
            var resultType = new TensorType(DataType.Float32, new[] { 5, 20, 30 });
            var result = _context.CreateValue(resultType);

            var gatherOp = new GatherOp(input, indices, result, 0);

            Assert.Throws<System.InvalidOperationException>(() => gatherOp.Validate());
        }

        [Test]
        public void GatherOp_Validate_SupportsInt64Indices()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Int64, new[] { 5 });
            var input = _context.CreateValue(inputType);
            var indices = _context.CreateValue(indicesType);
            var resultType = new TensorType(DataType.Float32, new[] { 5, 20, 30 });
            var result = _context.CreateValue(resultType);

            var gatherOp = new GatherOp(input, indices, result, 0);

            Assert.DoesNotThrow(() => gatherOp.Validate());
        }

        [Test]
        public void GatherOp_Validate_ThrowsForMismatchedResultShape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Int32, new[] { 5 });
            var input = _context.CreateValue(inputType);
            var indices = _context.CreateValue(indicesType);
            var resultType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });  // Wrong shape
            var result = _context.CreateValue(resultType);

            var gatherOp = new GatherOp(input, indices, result, 0);

            Assert.Throws<System.InvalidOperationException>(() => gatherOp.Validate());
        }

        [Test]
        public void GatherOp_Construction_ThrowsForNullInput()
        {
            var indicesType = new TensorType(DataType.Int32, new[] { 5 });
            var indices = _context.CreateValue(indicesType);
            var resultType = new TensorType(DataType.Float32, new[] { 5, 20, 30 });
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new GatherOp(null, indices, result, 0));
        }

        [Test]
        public void GatherOp_Construction_ThrowsForNullIndices()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var input = _context.CreateValue(inputType);
            var resultType = new TensorType(DataType.Float32, new[] { 5, 20, 30 });
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new GatherOp(input, null, result, 0));
        }

        [Test]
        public void GatherOp_Construction_ThrowsForNullResult()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var input = _context.CreateValue(inputType);
            var indicesType = new TensorType(DataType.Int32, new[] { 5 });
            var indices = _context.CreateValue(indicesType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new GatherOp(input, indices, null, 0));
        }

        [Test]
        public void GatherOp_Clone_CreatesIndependentCopy()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Int32, new[] { 5 });
            var input = _context.CreateValue(inputType);
            var indices = _context.CreateValue(indicesType);
            var resultType = new TensorType(DataType.Float32, new[] { 5, 20, 30 });
            var result = _context.CreateValue(resultType);

            var original = new GatherOp(input, indices, result, 0);
            var cloned = (GatherOp)original.Clone();

            Assert.AreEqual(original.Input, cloned.Input);
            Assert.AreEqual(original.Indices, cloned.Indices);
            Assert.AreEqual(original.Result, cloned.Result);
            Assert.AreEqual(original.Axis, cloned.Axis);
        }

        [Test]
        public void GatherOp_Validate_SupportsMultiDimensionalIndices()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 10, 20, 30 });
            var indicesType = new TensorType(DataType.Int32, new[] { 5, 3 });
            var input = _context.CreateValue(inputType);
            var indices = _context.CreateValue(indicesType);
            var resultType = new TensorType(DataType.Float32, new[] { 5, 3, 20, 30 });
            var result = _context.CreateValue(resultType);

            var gatherOp = new GatherOp(input, indices, result, 0);

            Assert.DoesNotThrow(() => gatherOp.Validate());
        }
    }
}
