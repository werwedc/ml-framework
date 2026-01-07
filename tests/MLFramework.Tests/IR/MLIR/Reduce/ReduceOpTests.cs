using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.MLIR.Reduce;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.MLIR.Reduce
{
    [TestFixture]
    public class ReduceOpTests
    {
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _context = new IRContext();
        }

        [Test]
        public void ReduceOp_CreatesCorrectOperation()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var outputType = new TensorType(DataType.Float32, new[] { 32, 1, 128 });
            var input = _context.CreateValue(inputType, "input");
            var result = _context.CreateValue(outputType, "result");
            var axes = new[] { 1 };

            var reduceOp = new ReduceOp(input, result, ReductionKind.Sum, axes, true);

            Assert.AreEqual(input, reduceOp.Input);
            Assert.AreEqual(result, reduceOp.Result);
            Assert.AreEqual(ReductionKind.Sum, reduceOp.Kind);
            CollectionAssert.AreEqual(axes, reduceOp.Axes);
            Assert.IsTrue(reduceOp.KeepDims);
        }

        [Test]
        public void ReduceOp_CreatesOperationWithMultipleAxes()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var outputType = new TensorType(DataType.Float32, new[] { 32 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(outputType);
            var axes = new[] { 1, 2 };

            var reduceOp = new ReduceOp(input, result, ReductionKind.Mean, axes, false);

            CollectionAssert.AreEqual(axes, reduceOp.Axes);
            Assert.IsFalse(reduceOp.KeepDims);
        }

        [Test]
        public void ReduceOp_CreatesOperationWithEmptyAxes()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var outputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(outputType);

            var reduceOp = new ReduceOp(input, result, ReductionKind.Max, System.Array.Empty<int>(), false);

            Assert.AreEqual(0, reduceOp.Axes.Length);
        }

        [Test]
        public void ReduceOp_SupportsAllReductionKinds()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var outputType = new TensorType(DataType.Float32, new[] { 32, 1 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(outputType);
            var axes = new[] { 1 };

            var kinds = new[] { ReductionKind.Sum, ReductionKind.Mean, ReductionKind.Max,
                              ReductionKind.Min, ReductionKind.Prod, ReductionKind.Any, ReductionKind.All };

            foreach (var kind in kinds)
            {
                var reduceOp = new ReduceOp(input, result, kind, axes, true);
                Assert.AreEqual(kind, reduceOp.Kind);
            }
        }

        [Test]
        public void ReduceOp_Construction_ThrowsForNullInput()
        {
            var outputType = new TensorType(DataType.Float32, new[] { 32, 1 });
            var result = _context.CreateValue(outputType);
            var axes = new[] { 1 };

            Assert.Throws<System.ArgumentNullException>(() =>
                new ReduceOp(null, result, ReductionKind.Sum, axes, true));
        }

        [Test]
        public void ReduceOp_Construction_ThrowsForNullResult()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);
            var axes = new[] { 1 };

            Assert.Throws<System.ArgumentNullException>(() =>
                new ReduceOp(input, null, ReductionKind.Sum, axes, true));
        }

        [Test]
        public void ReduceOp_Construction_AcceptsNullAxes()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var outputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(outputType);

            var reduceOp = new ReduceOp(input, result, ReductionKind.Sum, null, false);

            Assert.IsNotNull(reduceOp.Axes);
            Assert.AreEqual(0, reduceOp.Axes.Length);
        }

        [Test]
        public void ReduceOp_Clone_CreatesIndependentCopy()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var outputType = new TensorType(DataType.Float32, new[] { 32, 1, 128 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(outputType);
            var axes = new[] { 1 };

            var original = new ReduceOp(input, result, ReductionKind.Sum, axes, true);
            var cloned = (ReduceOp)original.Clone();

            Assert.AreEqual(original.Input, cloned.Input);
            Assert.AreEqual(original.Result, cloned.Result);
            Assert.AreEqual(original.Kind, cloned.Kind);
            CollectionAssert.AreEqual(original.Axes, cloned.Axes);
            Assert.AreEqual(original.KeepDims, cloned.KeepDims);
            Assert.AreNotSame(original.Axes, cloned.Axes);
        }

        [Test]
        public void ReduceOp_Clone_WithEmptyAxes()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var outputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(outputType);

            var original = new ReduceOp(input, result, ReductionKind.Max, System.Array.Empty<int>(), false);
            var cloned = (ReduceOp)original.Clone();

            Assert.AreEqual(0, cloned.Axes.Length);
        }

        [Test]
        public void ReduceOp_Validate_DoesNotThrowByDefault()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64 });
            var outputType = new TensorType(DataType.Float32, new[] { 32, 1 });
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(outputType);
            var axes = new[] { 1 };

            var reduceOp = new ReduceOp(input, result, ReductionKind.Sum, axes, true);

            Assert.DoesNotThrow(() => reduceOp.Validate());
        }
    }
}
