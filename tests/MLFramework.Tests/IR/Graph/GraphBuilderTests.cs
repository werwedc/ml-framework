using NUnit.Framework;
using MLFramework.IR;
using MLFramework.IR.HLIR;
using MLFramework.IR.HLIR.Elementwise;
using MLFramework.IR.HLIR.Activation;
using MLFramework.IR.HLIR.Matrix;
using MLFramework.IR.HLIR.Conv;
using MLFramework.IR.HLIR.Pool;
using MLFramework.IR.Types;
using MLFramework.IR.Values;
using MLFramework.IR.Attributes;

namespace MLFramework.Tests.IR.Graph
{
    [TestFixture]
    public class GraphBuilderTests
    {
        private HLIRModule _module;
        private IRContext _context;

        [SetUp]
        public void Setup()
        {
            _module = new HLIRModule("TestModule");
            _context = new IRContext();
        }

        [Test]
        public void IRBlock_CreateWithValidName()
        {
            var block = new IRBlock("TestBlock");

            Assert.AreEqual("TestBlock", block.Name);
            Assert.IsNotNull(block.Operations);
            Assert.IsNotNull(block.Arguments);
            Assert.IsNotNull(block.Returns);
            Assert.AreEqual(0, block.Operations.Count);
            Assert.AreEqual(0, block.Arguments.Count);
            Assert.AreEqual(0, block.Returns.Count);
        }

        [Test]
        public void IRBlock_CreateWithNullName_Throws()
        {
            Assert.Throws<System.ArgumentNullException>(() =>
                new IRBlock(null));
        }

        [Test]
        public void IRBlock_AddOperation_AddsOperationToBlock()
        {
            var block = new IRBlock("TestBlock");
            var inputType = new TensorType(DataType.Float32, new[] { 32, 784 });
            var input = _context.CreateValue(inputType, "x");
            var result = AddOp.Create(_context, input, input);

            block.AddOperation(result.DefiningOperation);

            Assert.AreEqual(1, block.Operations.Count);
        }

        [Test]
        public void IRBlock_AddOperation_ThrowsForNullOperation()
        {
            var block = new IRBlock("TestBlock");

            Assert.Throws<System.ArgumentNullException>(() =>
                block.AddOperation(null));
        }

        [Test]
        public void IRBlock_AddArgument_AddsArgumentToBlock()
        {
            var block = new IRBlock("TestBlock");
            var inputType = new TensorType(DataType.Float32, new[] { 32, 784 });
            var input = _context.CreateValue(inputType, "x");

            block.AddArgument(input);

            Assert.AreEqual(1, block.Arguments.Count);
            Assert.AreEqual(input, block.Arguments[0]);
        }

        [Test]
        public void IRBlock_AddArgument_ThrowsForNullArgument()
        {
            var block = new IRBlock("TestBlock");

            Assert.Throws<System.ArgumentNullException>(() =>
                block.AddArgument(null));
        }

        [Test]
        public void IRBlock_AddReturn_AddsReturnValueToBlock()
        {
            var block = new IRBlock("TestBlock");
            var inputType = new TensorType(DataType.Float32, new[] { 32, 784 });
            var input = _context.CreateValue(inputType, "x");
            var result = AddOp.Create(_context, input, input);

            block.AddReturn(result);

            Assert.AreEqual(1, block.Returns.Count);
            Assert.AreEqual(result, block.Returns[0]);
        }

        [Test]
        public void IRBlock_AddReturn_ThrowsForNullReturnValue()
        {
            var block = new IRBlock("TestBlock");

            Assert.Throws<System.ArgumentNullException>(() =>
                block.AddReturn(null));
        }

        [Test]
        public void BuildSimpleLinearLayer_CreatesCorrectGraph()
        {
            var func = new HIRFunction { Name = "Linear" };
            var inputType = new TensorType(DataType.Float32, new[] { 32, 784 });
            var weightType = new TensorType(DataType.Float32, new[] { 784, 256 });
            var input = _context.CreateValue(inputType, "x");

            func.Parameters.Add(input);

            var weight = _context.CreateValue(weightType, "w1");
            var matmul = MatMulOp.Create(_context, input, weight, "matmul");
            var h1 = ReLUOp.Create(_context, matmul, "h1");

            func.Result = h1;

            Assert.AreEqual(1, func.Parameters.Count);
            Assert.IsNotNull(func.Result);
            Assert.AreEqual("Linear", func.Name);
        }

        [Test]
        public void BuildMLP_CreatesCorrectGraph()
        {
            var func = new HIRFunction { Name = "MLP" };
            var inputType = new TensorType(DataType.Float32, new[] { 32, 784 });
            var weight1Type = new TensorType(DataType.Float32, new[] { 784, 256 });
            var weight2Type = new TensorType(DataType.Float32, new[] { 256, 10 });
            var input = _context.CreateValue(inputType, "x");

            func.Parameters.Add(input);

            var w1 = _context.CreateValue(weight1Type, "w1");
            var h1 = ReLUOp.Create(_context, MatMulOp.Create(_context, input, w1), "h1");

            var w2 = _context.CreateValue(weight2Type, "w2");
            var output = MatMulOp.Create(_context, h1, w2, "output");

            func.Result = output;

            Assert.AreEqual(1, func.Parameters.Count);
            Assert.IsNotNull(func.Result);
            Assert.AreEqual("MLP", func.Name);
        }

        [Test]
        public void BuildConvNet_CreatesCorrectGraph()
        {
            var func = new HIRFunction { Name = "ConvNet" };
            var inputType = new TensorType(DataType.Float32, new[] { 4, 1, 28, 28 });
            var weightType = new TensorType(DataType.Float32, new[] { 32, 1, 3, 3 });
            var input = _context.CreateValue(inputType, "x");

            func.Parameters.Add(input);

            var conv1Weight = _context.CreateValue(weightType, "conv1_w");
            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var padding = new[] { 1, 1 };

            var conv1 = Conv2DOp.Create(_context, input, conv1Weight, null, kernelSize, stride, padding, name: "conv1");
            var relu1 = ReLUOp.Create(_context, conv1, "relu1");

            var poolKernel = new[] { 2, 2 };
            var poolStride = new[] { 2, 2 };
            var pool1 = MaxPool2DOp.Create(_context, relu1, poolKernel, poolStride, "pool1");

            func.Result = pool1;

            Assert.AreEqual(1, func.Parameters.Count);
            Assert.IsNotNull(func.Result);
            Assert.AreEqual("ConvNet", func.Name);
        }

        [Test]
        public void BuildResidualBlock_CreatesCorrectGraph()
        {
            var func = new HIRFunction { Name = "ResidualBlock" };
            var inputType = new TensorType(DataType.Float32, new[] { 4, 64, 56, 56 });
            var conv1WeightType = new TensorType(DataType.Float32, new[] { 64, 64, 3, 3 });
            var conv2WeightType = new TensorType(DataType.Float32, new[] { 64, 64, 3, 3 });
            var input = _context.CreateValue(inputType, "x");

            func.Parameters.Add(input);

            var kernelSize = new[] { 3, 3 };
            var stride = new[] { 1, 1 };
            var padding = new[] { 1, 1 };

            var w1 = _context.CreateValue(conv1WeightType, "w1");
            var conv1 = Conv2DOp.Create(_context, input, w1, null, kernelSize, stride, padding, name: "conv1");
            var bn1 = conv1; // Simplified - would normally be batch norm

            var relu1 = ReLUOp.Create(_context, bn1, "relu1");

            var w2 = _context.CreateValue(conv2WeightType, "w2");
            var conv2 = Conv2DOp.Create(_context, relu1, w2, null, kernelSize, stride, padding, name: "conv2");

            var relu2 = ReLUOp.Create(_context, AddOp.Create(_context, conv2, input), "relu2");

            func.Result = relu2;

            Assert.AreEqual(1, func.Parameters.Count);
            Assert.IsNotNull(func.Result);
        }

        [Test]
        public void BuildComplexModel_WithMultipleOperations()
        {
            var func = new HIRFunction { Name = "ComplexModel" };
            var inputType = new TensorType(DataType.Float32, new[] { 4, 3, 32, 32 });
            var input = _context.CreateValue(inputType, "x");

            func.Parameters.Add(input);

            // Conv1
            var conv1WType = new TensorType(DataType.Float32, new[] { 32, 3, 3, 3 });
            var conv1W = _context.CreateValue(conv1WType, "conv1_w");
            var conv1 = Conv2DOp.Create(_context, input, conv1W, null, new[] { 3, 3 }, new[] { 1, 1 }, new[] { 1, 1 }, name: "conv1");
            var relu1 = ReLUOp.Create(_context, conv1, "relu1");

            // Pool1
            var pool1 = MaxPool2DOp.Create(_context, relu1, new[] { 2, 2 }, new[] { 2, 2 }, "pool1");

            // Conv2
            var conv2WType = new TensorType(DataType.Float32, new[] { 64, 32, 3, 3 });
            var conv2W = _context.CreateValue(conv2WType, "conv2_w");
            var conv2 = Conv2DOp.Create(_context, pool1, conv2W, null, new[] { 3, 3 }, new[] { 1, 1 }, new[] { 1, 1 }, name: "conv2");
            var relu2 = ReLUOp.Create(_context, conv2, "relu2");

            // Pool2
            var pool2 = MaxPool2DOp.Create(_context, relu2, new[] { 2, 2 }, new[] { 2, 2 }, "pool2");

            func.Result = pool2;

            Assert.AreEqual(1, func.Parameters.Count);
            Assert.IsNotNull(func.Result);
        }

        [Test]
        public void IRBlock_ToString_ReturnsCorrectFormat()
        {
            var block = new IRBlock("TestBlock");

            var str = block.ToString();

            Assert.IsTrue(str.Contains("TestBlock"));
            Assert.IsTrue(str.Contains("0 ops"));
        }

        [Test]
        public void IRBlock_WithOperations_ReturnsCorrectCount()
        {
            var block = new IRBlock("TestBlock");
            var inputType = new TensorType(DataType.Float32, new[] { 32, 784 });
            var input = _context.CreateValue(inputType, "x");

            var op1 = AddOp.Create(_context, input, input).DefiningOperation;
            var op2 = ReLUOp.Create(_context, input).DefiningOperation;

            block.AddOperation(op1);
            block.AddOperation(op2);

            var str = block.ToString();

            Assert.IsTrue(str.Contains("2 ops"));
        }

        [Test]
        public void IRBlock_MultipleArguments()
        {
            var block = new IRBlock("TestBlock");
            var type1 = new TensorType(DataType.Float32, new[] { 32, 784 });
            var type2 = new TensorType(DataType.Float32, new[] { 32, 256 });

            var arg1 = _context.CreateValue(type1, "x");
            var arg2 = _context.CreateValue(type2, "y");

            block.AddArgument(arg1);
            block.AddArgument(arg2);

            Assert.AreEqual(2, block.Arguments.Count);
            Assert.AreEqual(arg1, block.Arguments[0]);
            Assert.AreEqual(arg2, block.Arguments[1]);
        }

        [Test]
        public void IRBlock_MultipleReturns()
        {
            var block = new IRBlock("TestBlock");
            var inputType = new TensorType(DataType.Float32, new[] { 32, 784 });
            var input = _context.CreateValue(inputType, "x");

            var result1 = AddOp.Create(_context, input, input);
            var result2 = ReLUOp.Create(_context, input);

            block.AddReturn(result1);
            block.AddReturn(result2);

            Assert.AreEqual(2, block.Returns.Count);
            Assert.AreEqual(result1, block.Returns[0]);
            Assert.AreEqual(result2, block.Returns[1]);
        }

        [Test]
        public void BuildLayerWithBias()
        {
            var func = new HIRFunction { Name = "LinearWithBias" };
            var inputType = new TensorType(DataType.Float32, new[] { 32, 784 });
            var weightType = new TensorType(DataType.Float32, new[] { 784, 256 });
            var biasType = new TensorType(DataType.Float32, new[] { 256 });
            var input = _context.CreateValue(inputType, "x");

            func.Parameters.Add(input);

            var weight = _context.CreateValue(weightType, "w");
            var bias = _context.CreateValue(biasType, "b");

            var matmul = MatMulOp.Create(_context, input, weight, "matmul");
            var biased = AddOp.Create(_context, matmul, bias, "biased");
            var output = ReLUOp.Create(_context, biased, "output");

            func.Result = output;

            Assert.AreEqual(1, func.Parameters.Count);
            Assert.IsNotNull(func.Result);
        }
    }
}
