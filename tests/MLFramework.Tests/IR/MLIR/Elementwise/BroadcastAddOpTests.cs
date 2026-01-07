using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.MLIR.Elementwise;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.MLIR.Elementwise
{
    [TestFixture]
    public class BroadcastAddOpTests
    {
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _context = new IRContext();
        }

        [Test]
        public void BroadcastAddOp_CreatesCorrectOperation()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 1, 64 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var lhs = _context.CreateValue(lhsType, "lhs");
            var rhs = _context.CreateValue(rhsType, "rhs");
            var result = _context.CreateValue(resultType, "result");

            var broadcastAddOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);

            Assert.AreEqual(lhs, broadcastAddOp.Lhs);
            Assert.AreEqual(rhs, broadcastAddOp.Rhs);
            Assert.AreEqual(result, broadcastAddOp.Result);
            CollectionAssert.AreEqual(broadcastShape, broadcastAddOp.BroadcastShape);
        }

        [Test]
        public void BroadcastAddOp_Validate_DoesNotThrowForValidBroadcast()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 1, 64 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var broadcastAddOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);

            Assert.DoesNotThrow(() => broadcastAddOp.Validate());
        }

        [Test]
        public void BroadcastAddOp_Validate_ThrowsForIncompatibleShapes()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var broadcastAddOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);

            Assert.Throws<System.InvalidOperationException>(() => broadcastAddOp.Validate());
        }

        [Test]
        public void BroadcastAddOp_Validate_ThrowsForDifferentElementTypes()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Int32, new[] { 1, 64 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var broadcastAddOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);

            Assert.Throws<System.InvalidOperationException>(() => broadcastAddOp.Validate());
        }

        [Test]
        public void BroadcastAddOp_Validate_ThrowsForMismatchedResultShape()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 1, 64 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var broadcastAddOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);

            Assert.Throws<System.InvalidOperationException>(() => broadcastAddOp.Validate());
        }

        [Test]
        public void BroadcastAddOp_Construction_ThrowsForNullLhs()
        {
            var rhsType = new TensorType(DataType.Float32, new[] { 1, 64 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new BroadcastAddOp(null, rhs, result, broadcastShape));
        }

        [Test]
        public void BroadcastAddOp_Construction_ThrowsForNullRhs()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var lhs = _context.CreateValue(lhsType);
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new BroadcastAddOp(lhs, null, result, broadcastShape));
        }

        [Test]
        public void BroadcastAddOp_Construction_ThrowsForNullResult()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 1, 64 });
            var broadcastShape = new[] { 32, 64 };
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new BroadcastAddOp(lhs, rhs, null, broadcastShape));
        }

        [Test]
        public void BroadcastAddOp_Construction_ThrowsForNullBroadcastShape()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 1, 64 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new BroadcastAddOp(lhs, rhs, result, null));
        }

        [Test]
        public void BroadcastAddOp_Clone_CreatesIndependentCopy()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 1, 64 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var original = new BroadcastAddOp(lhs, rhs, result, broadcastShape);
            var cloned = (BroadcastAddOp)original.Clone();

            Assert.AreEqual(original.Lhs, cloned.Lhs);
            Assert.AreEqual(original.Rhs, cloned.Rhs);
            Assert.AreEqual(original.Result, cloned.Result);
            CollectionAssert.AreEqual(original.BroadcastShape, cloned.BroadcastShape);
            Assert.AreNotSame(original.BroadcastShape, cloned.BroadcastShape);
        }

        [Test]
        public void BroadcastAddOp_Validate_SupportsScalarBroadcast()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 1 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var broadcastAddOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);

            Assert.DoesNotThrow(() => broadcastAddOp.Validate());
        }

        [Test]
        public void BroadcastAddOp_Validate_ThrowsForNonTensorLhs()
        {
            var lhs = _context.CreateValue(new Attributes.ScalarType(DataType.Float32));
            var rhsType = new TensorType(DataType.Float32, new[] { 1, 64 });
            var broadcastShape = new[] { 1, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var broadcastAddOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);

            Assert.Throws<System.InvalidOperationException>(() => broadcastAddOp.Validate());
        }

        [Test]
        public void BroadcastAddOp_Validate_ThrowsForNonTensorRhs()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var broadcastShape = new[] { 32, 64 };
            var resultType = new TensorType(DataType.Float32, broadcastShape);
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(new Attributes.ScalarType(DataType.Float32));
            var result = _context.CreateValue(resultType);

            var broadcastAddOp = new BroadcastAddOp(lhs, rhs, result, broadcastShape);

            Assert.Throws<System.InvalidOperationException>(() => broadcastAddOp.Validate());
        }
    }
}
