using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.MLIR.Index;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.MLIR.Index
{
    [TestFixture]
    public class SliceOpTests
    {
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _context = new IRContext();
        }

        [Test]
        public void SliceOp_CreatesCorrectOperation()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType, "input");
            var starts = new[] { 0, 10, 20 };
            var ends = new[] { 16, 50, 100 };
            var strides = new[] { 1, 1, 1 };
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType, "result");

            var sliceOp = new SliceOp(input, result, starts, ends, strides);

            Assert.AreEqual(input, sliceOp.Input);
            Assert.AreEqual(result, sliceOp.Result);
            CollectionAssert.AreEqual(starts, sliceOp.Starts);
            CollectionAssert.AreEqual(ends, sliceOp.Ends);
            CollectionAssert.AreEqual(strides, sliceOp.Strides);
        }

        [Test]
        public void SliceOp_Validate_DoesNotThrowForValidSlice()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 0, 10, 20 };
            var ends = new[] { 16, 50, 100 };
            var strides = new[] { 1, 1, 1 };
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);

            var sliceOp = new SliceOp(input, result, starts, ends, strides);

            Assert.DoesNotThrow(() => sliceOp.Validate());
        }

        [Test]
        public void SliceOp_Validate_ThrowsForMismatchedStartsLength()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 0, 10 };  // Wrong length
            var ends = new[] { 16, 50, 100 };
            var strides = new[] { 1, 1, 1 };
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);

            var sliceOp = new SliceOp(input, result, starts, ends, strides);

            Assert.Throws<System.InvalidOperationException>(() => sliceOp.Validate());
        }

        [Test]
        public void SliceOp_Validate_ThrowsForMismatchedEndsLength()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 0, 10, 20 };
            var ends = new[] { 16, 50 };  // Wrong length
            var strides = new[] { 1, 1, 1 };
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);

            var sliceOp = new SliceOp(input, result, starts, ends, strides);

            Assert.Throws<System.InvalidOperationException>(() => sliceOp.Validate());
        }

        [Test]
        public void SliceOp_Validate_ThrowsForZeroStride()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 0, 10, 20 };
            var ends = new[] { 16, 50, 100 };
            var strides = new[] { 1, 0, 1 };  // Zero stride
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);

            var sliceOp = new SliceOp(input, result, starts, ends, strides);

            Assert.Throws<System.InvalidOperationException>(() => sliceOp.Validate());
        }

        [Test]
        public void SliceOp_Validate_ThrowsForStartOutOfBounds()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 0, 70, 20 };  // 70 > 64
            var ends = new[] { 16, 50, 100 };
            var strides = new[] { 1, 1, 1 };
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);

            var sliceOp = new SliceOp(input, result, starts, ends, strides);

            Assert.Throws<System.InvalidOperationException>(() => sliceOp.Validate());
        }

        [Test]
        public void SliceOp_Validate_ThrowsForEndOutOfBounds()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 0, 10, 20 };
            var ends = new[] { 16, 50, 130 };  // 130 > 128
            var strides = new[] { 1, 1, 1 };
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);

            var sliceOp = new SliceOp(input, result, starts, ends, strides);

            Assert.Throws<System.InvalidOperationException>(() => sliceOp.Validate());
        }

        [Test]
        public void SliceOp_Validate_ThrowsForStartGreaterThanEndWithPositiveStride()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 0, 50, 20 };  // 50 > 10
            var ends = new[] { 16, 10, 100 };
            var strides = new[] { 1, 1, 1 };
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);

            var sliceOp = new SliceOp(input, result, starts, ends, strides);

            Assert.Throws<System.InvalidOperationException>(() => sliceOp.Validate());
        }

        [Test]
        public void SliceOp_Validate_SupportsNegativeStride()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 15, 49, 99 };
            var ends = new[] { 0, 10, 20 };
            var strides = new[] { -1, -1, -1 };
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);

            var sliceOp = new SliceOp(input, result, starts, ends, strides);

            Assert.DoesNotThrow(() => sliceOp.Validate());
        }

        [Test]
        public void SliceOp_Construction_ThrowsForNullInput()
        {
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);
            var starts = new[] { 0, 10, 20 };
            var ends = new[] { 16, 50, 100 };
            var strides = new[] { 1, 1, 1 };

            Assert.Throws<System.ArgumentNullException>(() =>
                new SliceOp(null, result, starts, ends, strides));
        }

        [Test]
        public void SliceOp_Construction_ThrowsForNullResult()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 0, 10, 20 };
            var ends = new[] { 16, 50, 100 };
            var strides = new[] { 1, 1, 1 };

            Assert.Throws<System.ArgumentNullException>(() =>
                new SliceOp(input, null, starts, ends, strides));
        }

        [Test]
        public void SliceOp_Construction_ThrowsForNullStarts()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);
            var ends = new[] { 16, 50, 100 };
            var strides = new[] { 1, 1, 1 };

            Assert.Throws<System.ArgumentNullException>(() =>
                new SliceOp(input, result, null, ends, strides));
        }

        [Test]
        public void SliceOp_Construction_ThrowsForNullEnds()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);
            var starts = new[] { 0, 10, 20 };
            var strides = new[] { 1, 1, 1 };

            Assert.Throws<System.ArgumentNullException>(() =>
                new SliceOp(input, result, starts, null, strides));
        }

        [Test]
        public void SliceOp_Construction_ThrowsForNullStrides()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);
            var starts = new[] { 0, 10, 20 };
            var ends = new[] { 16, 50, 100 };

            Assert.Throws<System.ArgumentNullException>(() =>
                new SliceOp(input, result, starts, ends, null));
        }

        [Test]
        public void SliceOp_Clone_CreatesIndependentCopy()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 32, 64, 128 });
            var input = _context.CreateValue(inputType);
            var starts = new[] { 0, 10, 20 };
            var ends = new[] { 16, 50, 100 };
            var strides = new[] { 1, 1, 1 };
            var resultType = new TensorType(DataType.Float32, new[] { 16, 40, 80 });
            var result = _context.CreateValue(resultType);

            var original = new SliceOp(input, result, starts, ends, strides);
            var cloned = (SliceOp)original.Clone();

            Assert.AreEqual(original.Input, cloned.Input);
            Assert.AreEqual(original.Result, cloned.Result);
            CollectionAssert.AreEqual(original.Starts, cloned.Starts);
            CollectionAssert.AreEqual(original.Ends, cloned.Ends);
            CollectionAssert.AreEqual(original.Strides, cloned.Strides);
            Assert.AreNotSame(original.Starts, cloned.Starts);
            Assert.AreNotSame(original.Ends, cloned.Ends);
            Assert.AreNotSame(original.Strides, cloned.Strides);
        }
    }
}
