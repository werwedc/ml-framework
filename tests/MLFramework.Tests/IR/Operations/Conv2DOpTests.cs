using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.HLIR.Conv;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.Tests.IR.Operations
{
    [TestFixture]
    public class Conv2DOpTests
    {
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _context = new IRContext();
        }

        [Test]
        public void Conv2DOp_CreatesCorrectOperation()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType, "input");
            var weight = _context.CreateValue(weightType, "weight");
            var result = _context.CreateValue(resultType, "result");

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride);

            Assert.AreEqual(input, convOp.Input);
            Assert.AreEqual(weight, convOp.Weight);
            Assert.IsNull(convOp.Bias);
            Assert.AreEqual(result, convOp.Result);
            Assert.AreEqual(kernelSize, convOp.KernelSize);
            Assert.AreEqual(stride, convOp.Stride);
        }

        [Test]
        public void Conv2DOp_Validate_DoesNotThrowForValidShapes()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride);

            Assert.DoesNotThrow(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_ThrowsForNon4DInput()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride);

            Assert.Throws<System.InvalidOperationException>(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_ThrowsForNon4DWeight()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride);

            Assert.Throws<System.InvalidOperationException>(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_ThrowsForInvalidKernelSizeLength()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride);

            Assert.Throws<System.InvalidOperationException>(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_ThrowsForInvalidStrideLength()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride);

            Assert.Throws<System.InvalidOperationException>(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_ThrowsForInvalidPaddingLength()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var padding = new[] { 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride, padding);

            Assert.Throws<System.InvalidOperationException>(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_ThrowsForWeightChannelMismatch()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 5, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride);

            Assert.Throws<System.InvalidOperationException>(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_ThrowsForWeightKernelMismatch()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 5, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride);

            Assert.Throws<System.InvalidOperationException>(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_DoesNotThrowForValidBias()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var biasType = new TensorType(DataType.Float32, new[] { 16 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var bias = _context.CreateValue(biasType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, bias, result, kernelSize, stride);

            Assert.DoesNotThrow(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_ThrowsForInvalidBiasSize()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var biasType = new TensorType(DataType.Float32, new[] { 32 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var bias = _context.CreateValue(biasType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, bias, result, kernelSize, stride);

            Assert.Throws<System.InvalidOperationException>(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Validate_ThrowsForNon1DBias()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var biasType = new TensorType(DataType.Float32, new[] { 16, 1 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var bias = _context.CreateValue(biasType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, bias, result, kernelSize, stride);

            Assert.Throws<System.InvalidOperationException>(() => convOp.Validate());
        }

        [Test]
        public void Conv2DOp_Create_CreatesOperationWithAutoGeneratedResult()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);

            var result = Conv2DOp.Create(_context, input, weight, null, kernelSize, stride, name: "result");

            Assert.IsNotNull(result);
            Assert.AreEqual("result", result.Name);

            var ops = _context.GetAllOperations();
            Assert.AreEqual(1, ops.Count);
        }

        [Test]
        public void Conv2DOp_Create_ComputesCorrectOutputShape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);

            var result = Conv2DOp.Create(_context, input, weight, null, kernelSize, stride);

            Assert.IsInstanceOf<TensorType>(result.Type);
            var resultType = (TensorType)result.Type;
            Assert.AreEqual(new[] { 4, 16, 26, 26 }, resultType.Shape);
        }

        [Test]
        public void Conv2DOp_Create_WithPadding_ComputesCorrectOutputShape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var padding = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);

            var result = Conv2DOp.Create(_context, input, weight, null, kernelSize, stride, padding);

            Assert.IsInstanceOf<TensorType>(result.Type);
            var resultType = (TensorType)result.Type;
            Assert.AreEqual(new[] { 4, 16, 28, 28 }, resultType.Shape);
        }

        [Test]
        public void Conv2DOp_Create_WithStride_ComputesCorrectOutputShape()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 2, 2 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);

            var result = Conv2DOp.Create(_context, input, weight, null, kernelSize, stride);

            Assert.IsInstanceOf<TensorType>(result.Type);
            var resultType = (TensorType)result.Type;
            Assert.AreEqual(new[] { 4, 16, 13, 13 }, resultType.Shape);
        }

        [Test]
        public void Conv2DOp_Clone_CreatesIndependentCopy()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var biasType = new TensorType(DataType.Float32, new[] { 16 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var padding = new[] { 1, 1 };
            var dilation = new[] { 2, 2 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var bias = _context.CreateValue(biasType);
            var result = _context.CreateValue(resultType);

            var original = new Conv2DOp(input, weight, bias, result, kernelSize, stride, padding, dilation, groups: 2);
            var cloned = (Conv2DOp)original.Clone();

            Assert.AreEqual(original.Input, cloned.Input);
            Assert.AreEqual(original.Weight, cloned.Weight);
            Assert.AreEqual(original.Bias, cloned.Bias);
            Assert.AreEqual(original.Result, cloned.Result);
            Assert.AreEqual(original.KernelSize, cloned.KernelSize);
            Assert.AreEqual(original.Stride, cloned.Stride);
            Assert.AreEqual(original.Padding, cloned.Padding);
            Assert.AreEqual(original.Dilation, cloned.Dilation);
            Assert.AreEqual(original.Groups, cloned.Groups);
        }

        [Test]
        public void Conv2DOp_Construction_ThrowsForNullInput()
        {
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new Conv2DOp(null, weight, null, result, kernelSize, stride));
        }

        [Test]
        public void Conv2DOp_Construction_ThrowsForNullWeight()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var result = _context.CreateValue(resultType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new Conv2DOp(input, null, null, result, kernelSize, stride));
        }

        [Test]
        public void Conv2DOp_Construction_ThrowsForNullResult()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);

            Assert.Throws<System.ArgumentNullException>(() =>
                new Conv2DOp(input, weight, null, null, kernelSize, stride));
        }

        [Test]
        public void Conv2DOp_Construction_UsesDefaultsForPaddingDilationGroups()
        {
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 16, 3, 3, 3 });
            var resultType = new TensorType(DataType.Float32, new[] { 4, 16, 26, 26 });
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var input = _context.CreateValue(inputType);
            var weight = _context.CreateValue(weightType);
            var result = _context.CreateValue(resultType);

            var convOp = new Conv2DOp(input, weight, null, result, kernelSize, stride);

            Assert.AreEqual(new[] { 0, 0 }, convOp.Padding);
            Assert.AreEqual(new[] { 1, 1 }, convOp.Dilation);
            Assert.AreEqual(1, convOp.Groups);
        }
    }
}
