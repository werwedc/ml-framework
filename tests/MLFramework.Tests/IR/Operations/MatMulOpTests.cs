using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.HLIR.Matrix;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.Operations
{
    [TestFixture]
    public class MatMulOpTests
    {
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _context = new IRContext();
        }

        [Test]
        public void MatMulOp_CreatesCorrectOperation()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var matMulOp = new MatMulOp(lhs, rhs, result);

            Assert.AreEqual(lhs, matMulOp.Lhs);
            Assert.AreEqual(rhs, matMulOp.Rhs);
            Assert.AreEqual(result, matMulOp.Result);
            Assert.IsFalse(matMulOp.TransposeA);
            Assert.IsFalse(matMulOp.TransposeB);
        }

        [Test]
        public void MatMulOp_Validate_DoesNotThrowForValidShapes()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var matMulOp = new MatMulOp(lhs, rhs, result);

            Assert.DoesNotThrow(() => matMulOp.Validate());
        }

        [Test]
        public void MatMulOp_Validate_ThrowsForIncompatibleInnerDimensions()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 128, 64 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var matMulOp = new MatMulOp(lhs, rhs, result);

            Assert.Throws<System.InvalidOperationException>(() => matMulOp.Validate());
        }

        [Test]
        public void MatMulOp_Validate_ThrowsForDifferentElementTypes()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Int32, new[] { 64, 128 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var matMulOp = new MatMulOp(lhs, rhs, result);

            Assert.Throws<System.InvalidOperationException>(() => matMulOp.Validate());
        }

        [Test]
        public void MatMulOp_WithTransposeA_RespectsTransposeFlag()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 64, 32 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var matMulOp = new MatMulOp(lhs, rhs, result, transposeA: true);

            Assert.IsTrue(matMulOp.TransposeA);
            Assert.DoesNotThrow(() => matMulOp.Validate());
        }

        [Test]
        public void MatMulOp_WithTransposeB_RespectsTransposeFlag()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 128, 64 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var matMulOp = new MatMulOp(lhs, rhs, result, transposeB: true);

            Assert.IsTrue(matMulOp.TransposeB);
            Assert.DoesNotThrow(() => matMulOp.Validate());
        }

        [Test]
        public void MatMulOp_Validate_ThrowsForIncompatibleTransposeDimensions()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var matMulOp = new MatMulOp(lhs, rhs, result, transposeA: true);

            Assert.Throws<System.InvalidOperationException>(() => matMulOp.Validate());
        }

        [Test]
        public void MatMulOp_Create_CreatesOperationWithAutoGeneratedResult()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);

            var result = MatMulOp.Create(_context, lhs, rhs, name: "result");

            Assert.IsNotNull(result);
            Assert.AreEqual("result", result.Name);

            // Verify operation was registered
            var ops = _context.GetAllOperations();
            Assert.AreEqual(1, ops.Count);
        }

        [Test]
        public void MatMulOp_Create_InfersCorrectOutputShape()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);

            var result = MatMulOp.Create(_context, lhs, rhs);

            Assert.IsInstanceOf<TensorType>(result.Type);
            var resultType = (TensorType)result.Type;
            Assert.AreEqual(new[] { 32, 128 }, resultType.Shape);
        }

        [Test]
        public void MatMulOp_Create_WithTranspose_InfersCorrectShape()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 64, 32 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);

            var result = MatMulOp.Create(_context, lhs, rhs, transposeA: true);

            Assert.IsInstanceOf<TensorType>(result.Type);
            var resultType = (TensorType)result.Type;
            Assert.AreEqual(new[] { 32, 128 }, resultType.Shape);
        }

        [Test]
        public void MatMulOp_Create_WithBatchDim_InfersCorrectShape()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 4, 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);

            var result = MatMulOp.Create(_context, lhs, rhs);

            Assert.IsInstanceOf<TensorType>(result.Type);
            var resultType = (TensorType)result.Type;
            Assert.AreEqual(new[] { 4, 32, 128 }, resultType.Shape);
        }

        [Test]
        public void MatMulOp_Create_ThrowsForIncompatibleDimensions()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 128, 64 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);

            Assert.Throws<System.InvalidOperationException>(() =>
                MatMulOp.Create(_context, lhs, rhs));
        }

        [Test]
        public void MatMulOp_Create_ThrowsForLessThan2D()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);

            Assert.Throws<System.InvalidOperationException>(() =>
                MatMulOp.Create(_context, lhs, rhs));
        }

        [Test]
        public void MatMulOp_Clone_CreatesIndependentCopy()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);
            var result = _context.CreateValue(resultType);

            var original = new MatMulOp(lhs, rhs, result, true, false);
            var cloned = (MatMulOp)original.Clone();

            Assert.AreEqual(original.Lhs, cloned.Lhs);
            Assert.AreEqual(original.Rhs, cloned.Rhs);
            Assert.AreEqual(original.Result, cloned.Result);
            Assert.AreEqual(original.TransposeA, cloned.TransposeA);
            Assert.AreEqual(original.TransposeB, cloned.TransposeB);
        }

        [Test]
        public void MatMulOp_Construction_ThrowsForNullLhs()
        {
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var rhs = _context.CreateValue(rhsType);
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new MatMulOp(null, rhs, result));
        }

        [Test]
        public void MatMulOp_Construction_ThrowsForNullRhs()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var lhs = _context.CreateValue(lhsType);
            var resultType = new TensorType(DataType.Float32, new[] { 32, 128 });
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new MatMulOp(lhs, null, result));
        }

        [Test]
        public void MatMulOp_Construction_ThrowsForNullResult()
        {
            var lhsType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var rhsType = new TensorType(DataType.Float32, new[] { 64, 128 });
            var lhs = _context.CreateValue(lhsType);
            var rhs = _context.CreateValue(rhsType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new MatMulOp(lhs, rhs, null));
        }
    }
}
